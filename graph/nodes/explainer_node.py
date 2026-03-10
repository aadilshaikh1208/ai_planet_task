from utils.llm import call_llm
from graph.state import AgentState


def explainer_node(state: AgentState) -> AgentState:

    problem     = state["input_text"]
    solution    = state["solution"]
    topic       = state["topic"]
    is_correct  = state["is_correct"]
    confidence  = state["confidence"]

    parsed      = state.get("parsed_problem",{})
    variables   = parsed.get("variables",   [])
    constraints = parsed.get("constraints", [])

    # Skip explanation if verifier marked solution incorrect
    if not is_correct:
        state["explanation"] = (
            "The solution could not be verified as correct. "
            "Please review the problem or consult a teacher."
        )
        state["agent_trace"].append("Explainer: skipped — solution marked incorrect by verifier")
        return state

    # Build compact context
    variables_str   = ", ".join(variables)   if variables   else "not specified"
    constraints_str = ", ".join(constraints) if constraints else "none"


    # Tone based on verification confidence
    if confidence >= 0.9:
        tone = "Be thorough and confident in your explanation."
    elif confidence >= 0.7:
        tone = "Explain clearly but note that students should double-check the final answer."
    else:
        tone = "Explain cautiously and encourage the student to verify independently."


    prompt = f"""
You are a JEE math tutor explaining a solution to a student.

Topic: {topic}

Problem:
{problem}

Solution:
{solution}

Variables: {variables_str}
Constraints: {constraints_str}

Explanation style: {tone}

Structure the explanation like this:

Concept:
Key formula or idea used.

Steps:
Clear step-by-step reasoning.

Shortcut:
Mention any JEE trick if applicable.

Final Answer:
One-line final answer.

Write clearly for a JEE student.
"""

    explanation = call_llm(prompt,max_tokens=500)

    state["explanation"] = explanation
    state["agent_trace"].append(
        f"Explainer: explanation generated, confidence={confidence:.2f}, topic={topic}"
    )

    return state