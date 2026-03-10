# graph/nodes/solver_node.py
#
# What this node does:
#   1. Retrieves relevant chunks from RAG knowledge base
#   2. Retrieves similar past solved problems from memory
#   3. Tries SymPy for symbolic problems first
#   4. Single LLM call with RAG + memory context combined
#   5. Stores solution in state

import re
import sympy as sp

from utils.llm import call_llm
from rag.retriever import retrieve
from memory.memory_store import retrieve_similar
from graph.state import AgentState


# ─────────────────────────────────────────────────────────────
# SymPy Solver
# ─────────────────────────────────────────────────────────────

def try_sympy(problem: str):
    """
    Attempt to solve simple algebra/calculus expressions with SymPy.
    Returns result string or None if not solvable.
    """

    x = sp.symbols("x")

    try:

        # Detect equation
        if "=" in problem:
            left, right = problem.split("=", 1)
            expr   = sp.sympify(left) - sp.sympify(right)
            result = sp.solve(expr, x)
            return f"SymPy result: {result}"

        # Detect derivative
        if "derivative" in problem.lower() or "differentiate" in problem.lower():
            expr_match = re.findall(r"\((.*?)\)", problem)
            if expr_match:
                expr   = sp.sympify(expr_match[0])
                result = sp.diff(expr, x)
                return f"SymPy result: {result}"

        # Detect integration
        if "integrate" in problem.lower() or "∫" in problem:
            expr_match = re.findall(r"\((.*?)\)", problem)
            if expr_match:
                expr   = sp.sympify(expr_match[0])
                result = sp.integrate(expr, x)
                return f"SymPy result: {result}"

    except Exception:
        return None

    return None


# ─────────────────────────────────────────────────────────────
# LLM Solve — now accepts memory_context too
# ─────────────────────────────────────────────────────────────

def solve_with_llm(problem: str, topic: str, context: str,
                   sympy_result=None, memory_context: str = ""):

    # Build memory section only if we have past similar problems
    memory_section = ""
    if memory_context:
        memory_section = f"""
Similar problems solved before (use as reference):
{memory_context}
"""

    if sympy_result:
        prompt = f"""
You are a JEE math tutor.

Problem:
{problem}

A symbolic math engine computed this result:
{sympy_result}

Relevant knowledge:
{context}
{memory_section}
Explain the full step-by-step solution clearly for a JEE student.
Show the reasoning and end with the final answer.
"""

    else:
        prompt = f"""
You are a JEE math solver.

Topic: {topic}

Relevant knowledge:
{context}
{memory_section}
Problem:
{problem}

Solve step by step clearly and give the final answer at the end.
"""

    return call_llm(prompt, max_tokens=800)


# ─────────────────────────────────────────────────────────────
# Solver Node
# ─────────────────────────────────────────────────────────────

def solver_node(state: AgentState) -> AgentState:

    problem = state["input_text"]
    topic   = state["topic"]

    # Step 1: Retrieve from RAG knowledge base
    chunks = retrieve(problem)

    retrieved_docs = [doc.page_content for doc in chunks]
    sources        = list(set([doc.metadata.get("source", "unknown") for doc in chunks]))
    context        = "\n\n".join(retrieved_docs) if retrieved_docs else "No relevant context found."

    # Step 2: Retrieve similar past solved problems from memory
    memory_context = retrieve_similar(problem, topic)

    # Step 3: Try SymPy first for symbolic problems
    sympy_result = try_sympy(problem)

    # Step 4: Single LLM call with RAG + memory context
    solution  = solve_with_llm(problem, topic, context, sympy_result, memory_context)
    tool_used = "sympy + llm" if sympy_result else "llm"

    # Log whether memory was used
    memory_used = "yes" if memory_context else "no"

    # Step 5: Update state
    state["retrieved_docs"] = retrieved_docs
    state["sources"]        = sources
    state["solution"]       = solution
    state["agent_trace"].append(
        f"Solver: used {tool_used}, "
        f"{len(chunks)} RAG chunks from {sources}, "
        f"memory_used={memory_used}"
    )

    return state