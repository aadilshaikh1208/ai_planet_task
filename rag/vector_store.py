# Embed chunks and save to Chroma. Load when already built.

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rag.ingest import load_and_chunk
from utils.config import Config
import os

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name   = "sentence-transformers/all-mpnet-base-v2",
        model_kwargs = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True}
    )

def build_vector_store():
    chunks = load_and_chunk()
    embeddings = get_embeddings()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="math_mentor",
        persist_directory=Config.VECTOR_DB_PATH
    )

    print(f"Vector store saved to {Config.VECTOR_DB_PATH}/")
    return vector_store

def load_vector_store():
    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name="math_mentor",
        embedding_function=embeddings,
        persist_directory=Config.VECTOR_DB_PATH
    )

    print("Vector store loaded")
    return vector_store

def get_vector_store():
    # Check for actual Chroma DB file to confirm if vector store exists
    chroma_db_file = os.path.join(Config.VECTOR_DB_PATH, "chroma.sqlite3")

    if os.path.exists(chroma_db_file):
        print("Existing vector store found. Loading...")
        return load_vector_store()
    else:
        print("No vector store found. Building from scratch...")
        return build_vector_store()