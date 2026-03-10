import json
from utils.llm import call_llm
from graph.state import AgentState

def parser_node(state: AgentState) -> AgentState:

    raw = state["input_text"]

    prompt = f"""
You are a math problem parser.

Given this raw input (may be from OCR or audio, so it might have errors):
"{raw}"

Your job:
1. Clean any OCR/audio errors
2. Identify the math problem clearly
3. Return ONLY a JSON object in this exact format:

{{
  "problem_text": "cleaned version of the problem",
  "topic": "one of: algebra / calculus / probability / linear_algebra",
  "variables": ["list", "of", "variables"],
  "constraints": ["any constraints like x > 0"],
  "needs_clarification": false
}}

Set needs_clarification to true ONLY if the problem is too ambiguous to solve.
Return ONLY the JSON. No explanation.
"""

    response = call_llm(prompt, max_tokens=200)

    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        parsed = {
            "problem_text": raw,
            "topic": "unknown",
            "variables": [],
            "constraints": [],
            "needs_clarification": True
        }

    needs_review = parsed.get("needs_clarification", False)

    state["parsed_problem"] = parsed
    state["input_text"] = parsed["problem_text"]
    state["needs_human_review"] = needs_review
    state["hitl_reason"] = "Parser could not understand the problem clearly." if needs_review else ""
    state["agent_trace"].append(f"Parser: topic={parsed['topic']}, needs_clarification={needs_review}")

    return state