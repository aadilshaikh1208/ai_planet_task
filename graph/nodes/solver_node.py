import re
import sympy as sp

from utils.llm import call_llm
from rag.retriever import retrieve
from memory.memory_store import retrieve_similar
from graph.state import AgentState


def try_sympy(problem: str):
    """
    Try to solve the problem symbolically using SymPy.
    Returns a result string if successful, None otherwise.
    """

    x = sp.symbols("x")

    try:
        if "=" in problem:
            left, right = problem.split("=", 1)
            expr   = sp.sympify(left) - sp.sympify(right)
            result = sp.solve(expr, x)
            return f"SymPy result: {result}"

        if "derivative" in problem.lower() or "differentiate" in problem.lower():
            expr_match = re.findall(r"\((.*?)\)", problem)
            if expr_match:
                expr   = sp.sympify(expr_match[0])
                result = sp.diff(expr, x)
                return f"SymPy result: {result}"

        if "integrate" in problem.lower() or "∫" in problem:
            expr_match = re.findall(r"\((.*?)\)", problem)
            if expr_match:
                expr   = sp.sympify(expr_match[0])
                result = sp.integrate(expr, x)
                return f"SymPy result: {result}"

    except Exception:
        return None

    return None


def solve_with_llm(problem: str, topic: str, context: str,
                   sympy_result=None, memory_context: str = ""):

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


def solver_node(state: AgentState) -> AgentState:

    problem = state["input_text"]
    topic   = state["topic"]

    chunks = retrieve(problem)
    retrieved_docs = [doc.page_content for doc in chunks]
    sources        = list(set([doc.metadata.get("source", "unknown") for doc in chunks]))
    context        = "\n\n".join(retrieved_docs) if retrieved_docs else "No relevant context found."

    memory_context = retrieve_similar(problem, topic)
    sympy_result = try_sympy(problem)
    solution  = solve_with_llm(problem, topic, context, sympy_result, memory_context)

    tool_used = "sympy + llm" if sympy_result else "llm"
    memory_used = "yes" if memory_context else "no"

    state["retrieved_docs"] = retrieved_docs
    state["sources"]        = sources
    state["solution"]       = solution
    state["agent_trace"].append(
        f"Solver: used {tool_used}, "
        f"{len(chunks)} RAG chunks from {sources}, "
        f"memory_used={memory_used}"
    )

    return state