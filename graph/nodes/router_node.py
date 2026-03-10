from utils.llm import call_llm
from graph.state import AgentState

VALID_TOPICS = ["algebra", "calculus", "probability", "linear_algebra"]

def router_node(state: AgentState) -> AgentState:

    parsed = state["parsed_problem"]
    problem_text = parsed.get("problem_text", "")
    parser_topic = parsed.get("topic", "")

    # If parser already gave a valid topic, trust it, Otherwise ask LLM to classify
    if parser_topic in VALID_TOPICS:
        state["topic"] = parser_topic
        state["agent_trace"].append(f"Router: routed to {parser_topic} (from parser)")
        return state

    # 
    prompt = f"""
You are a math topic classifier.

Problem: "{problem_text}"

Classify this into exactly one of these topics:
- algebra
- calculus
- probability
- linear_algebra

Reply with just the topic name. Nothing else.
"""

    response = call_llm(prompt, max_tokens=10).strip().lower()

    # Validate LLM response
    if response in VALID_TOPICS:
        topic = response
        state["needs_human_review"] = False
        state["hitl_reason"] = ""
    else:
        topic = "unknown"
        state["needs_human_review"] = True
        state["hitl_reason"] = f"Router could not classify the topic. LLM returned: '{response}'"

    state["topic"] = topic
    state["agent_trace"].append(f"Router: routed to {topic}")

    return state