import uuid
from datetime import datetime

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from utils.config import Config


# Same embedding model as RAG — no extra download needed

def get_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name    = "sentence-transformers/all-mpnet-base-v2",
        model_kwargs  = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True}
    )
    return embeddings


# Separate Chroma collection from RAG, stored in the same vector_store folder
def get_memory_store():
    memory_store = Chroma(
        collection_name    = "math_memory",
        embedding_function = get_embeddings(),
        persist_directory  = Config.VECTOR_DB_PATH
    )
    return memory_store


# ─────────────────────────────────────────────────────────────
# Save memory after pipeline finishes
# ─────────────────────────────────────────────────────────────

def save_memory(state):
    """
    Save a solved problem into memory.
    Called from app.py after graph.invoke() completes.
    """

    try:
        parsed_question = state.get("input_text", "")

        if parsed_question.strip() == "":
            return

        input_mode       = state.get("input_mode",       "text")
        raw_input        = state.get("raw_input",         "")
        topic            = state.get("topic",             "unknown")
        solution         = state.get("solution",          "")
        is_correct       = state.get("is_correct",        False)
        confidence       = state.get("confidence",        0.0)
        feedback         = state.get("feedback",          "")
        feedback_comment = state.get("feedback_comment",  "")

        # Cap long strings so metadata doesn't get too large
        if len(raw_input) > 500:
            raw_input = raw_input[:500]

        if len(solution) > 1000:
            solution = solution[:1000]

        # Chroma only supports string metadata values
        metadata = {
            "id"              : str(uuid.uuid4()),
            "timestamp"       : datetime.now().isoformat(),
            "input_mode"      : input_mode,
            "raw_input"       : raw_input,
            "topic"           : topic,
            "solution"        : solution,
            "is_correct"      : str(is_correct),
            "confidence"      : str(confidence),
            "feedback"        : feedback,
            "feedback_comment": feedback_comment,
        }

        doc = Document(
            page_content = parsed_question,
            metadata     = metadata
        )

        memory_store = get_memory_store()
        memory_store.add_documents([doc])

        print(f"Memory saved: {parsed_question[:80]}...")

    except Exception as e:
        print(f"Memory save failed (non-critical): {e}")


def retrieve_similar(problem, topic, k=4):
    """
    Search memory for semantically similar past solved problems.
    Called from solver_node.py before the LLM solve call.

    Returns a formatted string of past solutions to add to solver context.
    Returns empty string if nothing useful found.
    """

    try:
        memory_store = get_memory_store()

        results = memory_store.similarity_search(problem, k=k)

        if not results:
            return ""

        useful_results = []

        for doc in results:
            meta = doc.metadata

            doc_topic    = meta.get("topic",      "")
            doc_correct  = meta.get("is_correct", "False")
            doc_solution = meta.get("solution",   "")

            same_topic   = doc_topic   == topic
            was_correct  = doc_correct == "True"
            has_solution = doc_solution.strip() != ""

            # Skip if this is the exact same problem being solved again
            is_same_problem = doc.page_content.strip().lower() == problem.strip().lower()

            if same_topic and was_correct and has_solution and not is_same_problem:
                entry = "Similar problem: " + doc.page_content + "\n"
                entry = entry + "Solution pattern: " + doc_solution[:300]
                useful_results.append(entry)

        if len(useful_results) == 0:
            return ""

        memory_context = "\n\n---\n\n".join(useful_results)
        return memory_context

    except Exception as e:
        print(f"Memory retrieval failed (non-critical): {e}")
        return ""