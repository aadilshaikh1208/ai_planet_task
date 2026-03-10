# This is the shared "memory" that passes between all agents.
# Every agent reads from this and adds its own output to it.

from typing import TypedDict, List

class AgentState(TypedDict):

    # Input 
    input_mode: str          # "text" / "image" / "audio"
    raw_input: str           # original OCR / transcript / typed text (before cleaning)
    input_text: str          # cleaned input sent to parser

    # Parser Agent 
    parsed_problem: dict     # { problem_text, topic, variables, constraints, needs_clarification }

    # Router Agent
    topic: str               # algebra / calculus / probability / linear_algebra

    # RAG 
    retrieved_docs: List[str]   # actual chunk content
    sources: List[str]          # source file names shown in UI

    # Solver Agent 
    solution: str            # final answer

    # Verifier Agent 
    is_correct: bool         # did verifier approve?
    confidence: float        # 0.0 to 1.0 — shown as indicator in UI

    # Explainer Agent 
    explanation: str         # step-by-step explanation for student

    # HITL 
    needs_human_review: bool    # true when OCR low / parser ambiguous / verifier unsure
    hitl_reason: str            # why HITL was triggered (shown to human reviewer)

    # UI / Trace 
    agent_trace: List[str]   # e.g. ["Parser: done", "Router: calculus", ...]

    # Memory / Feedback 
    feedback: str            # "correct" / "incorrect" / "" — from user feedback buttons
    feedback_comment: str    # optional comment when user clicks ❌