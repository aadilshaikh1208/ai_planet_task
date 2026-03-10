import re
import json
import sympy as sp

from utils.llm import call_llm
from graph.state import AgentState

# If final confidence is below this threshold, flag for human review
CONFIDENCE_THRESHOLD = 0.7


# First SymPy verification
def sympy_verify(problem: str, solution: str, variables: list) -> dict:
    """
    Try to extract the answer from solution text and substitute
    back into the original equation to verify mathematically.

    Returns:
        {
            "ran": True/False,      # whether SymPy could actually run
            "passed": True/False,   # whether back-substitution passed
            "detail": "..."         # what happened
        }
    """

    try:
        # Only works if problem has an equation (has "=")
        if "=" not in problem:
            return {"ran": False, "passed": False, "detail": "Problem has no equation to substitute into."}

        # Build symbols from variables list (e.g. ["x", "y"] → x, y symbols)
        symbol_names = variables if variables else ["x"]
        symbols = {name: sp.Symbol(name) for name in symbol_names}
        main_var = list(symbols.values())[0]

        # Split equation into left and right side
        left_str, right_str = problem.split("=", 1)
        left_expr  = sp.sympify(left_str,  locals=symbols)
        right_expr = sp.sympify(right_str, locals=symbols)

        # Try to extract numeric answer from solution text
        answer_pattern = rf"{main_var}\s*=\s*([-\d/\.]+)"
        matches = re.findall(answer_pattern, solution)

        if not matches:
            return {"ran": False, "passed": False, "detail": "Could not extract a numeric answer from solution text."}

        # Check each extracted answer value
        all_passed = True
        checked    = []

        for match in matches:
            value = sp.sympify(match)

            # Substitute into both sides
            lhs = left_expr.subs(main_var, value)
            rhs = right_expr.subs(main_var, value)

            diff = sp.simplify(lhs - rhs)

            if diff == 0:
                checked.append(f"{main_var}={value} ✅")
            else:
                checked.append(f"{main_var}={value} ❌ (LHS-RHS={diff})")
                all_passed = False

        return {
            "ran":    True,
            "passed": all_passed,
            "detail": f"Back-substitution results: {', '.join(checked)}"
        }

    except Exception as e:
        return {"ran": False, "passed": False, "detail": f"SymPy error: {str(e)}"}


# LLM Review
def llm_verify(problem: str, solution: str, topic: str, sympy_check: dict) -> dict:
    """
    Ask LLM to review the solution.
    If SymPy already verified it, tell the LLM so it can focus
    on reasoning quality rather than recomputing the answer.

    Returns parsed dict with is_correct, confidence, reason.
    """

    # Tell LLM what SymPy found so it doesn't contradict a hard check
    if sympy_check["ran"] and sympy_check["passed"]:
        sympy_context = f"A symbolic math engine already verified the answer is correct: {sympy_check['detail']}. Focus on checking the reasoning and steps."

    elif sympy_check["ran"] and not sympy_check["passed"]:
        sympy_context = f"A symbolic math engine found the answer is WRONG: {sympy_check['detail']}. The solution is likely incorrect."

    else:
        sympy_context = "Symbolic verification was not possible for this problem. Do a full review yourself."

    prompt = f"""
You are a JEE math verifier.

Topic: {topic}

Problem:
{problem}

Solution to verify:
{solution}

Symbolic check result:
{sympy_context}

Your job:
1. Check if the solution is correct and reasoning is valid
2. Assign a confidence score between 0.0 and 1.0

Return ONLY a JSON object in this exact format:
{{
  "is_correct": true,
  "confidence": 0.95,
  "reason": "one sentence explaining your verdict"
}}

Return ONLY the JSON. No explanation. No markdown.
"""

    response = call_llm(prompt,max_tokens=150)

    try:
        result = json.loads(response)
        return {
            "is_correct": bool(result.get("is_correct", False)),
            "confidence": float(result.get("confidence", 0.0)),
            "reason":     result.get("reason", "")
        }

    except (json.JSONDecodeError, ValueError):
        return {
            "is_correct": False,
            "confidence": 0.0,
            "reason":     "Verifier could not parse LLM response."
        }


# Combine Both Tools into Final Verdict
def combine_verdict(sympy_check: dict, llm_check: dict) -> tuple:
    """
    Combine SymPy and LLM results into final is_correct + confidence.

    Rules:
    - If SymPy ran and FAILED → is_correct=False, cap confidence at 0.4
    - If SymPy ran and PASSED → boost LLM confidence by 0.1 (max 1.0)
    - If SymPy didn't run     → trust LLM confidence as-is

    Returns: (is_correct, confidence)
    """

    is_correct = llm_check["is_correct"]
    confidence = llm_check["confidence"]

    if sympy_check["ran"] and sympy_check["passed"]:
        # Hard mathematical proof — boost confidence
        confidence = min(1.0, confidence + 0.1)

    elif sympy_check["ran"] and not sympy_check["passed"]:
        # Hard mathematical disproof — override
        is_correct = False
        confidence = min(confidence, 0.4)

    return is_correct, confidence


# Verifier Node
def verifier_node(state: AgentState) -> AgentState:

    problem   = state["input_text"]
    solution  = state["solution"]
    topic     = state["topic"]
    variables = state["parsed_problem"].get("variables", [])

    # SymPy back-substitution (no LLM call)
    sympy_check = sympy_verify(problem, solution, variables)

    # LLM review (informed by SymPy result)
    llm_check = llm_verify(problem, solution, topic, sympy_check)

    # Combine both into final verdict
    is_correct, confidence = combine_verdict(sympy_check, llm_check)

    # Trigger HITL if confidence is too low
    if confidence < CONFIDENCE_THRESHOLD:
        state["needs_human_review"] = True
        state["hitl_reason"]        = (
            f"Verifier low confidence ({confidence:.2f}): {llm_check['reason']} "
            f"| SymPy: {sympy_check['detail']}"
        )

    # Update state
    state["is_correct"]  = is_correct
    state["confidence"]  = confidence
    state["agent_trace"].append(
        f"Verifier: is_correct={is_correct}, confidence={confidence:.2f}, "
        f"sympy_ran={sympy_check['ran']}, sympy_passed={sympy_check['passed']}"
    )

    return state