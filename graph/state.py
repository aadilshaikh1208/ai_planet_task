from typing import TypedDict, List

# Shared state passed between all agents in the pipeline.
class AgentState(TypedDict):

    # Input
    input_mode: str          # "text" / "image" / "audio"
    raw_input: str           # original text before any cleaning
    input_text: str          # cleaned version sent to parser

    # Parser
    parsed_problem: dict     # { problem_text, topic, variables, constraints, needs_clarification }

    # Router
    topic: str               # algebra / calculus / probability / linear_algebra

    # RAG
    retrieved_docs: List[str]
    sources: List[str]

    # Solver
    solution: str

    # Verifier
    is_correct: bool
    confidence: float        # 0.0 to 1.0

    # Explainer
    explanation: str

    # HITL
    needs_human_review: bool
    hitl_reason: str         # shown in UI when HITL triggers

    # Trace
    agent_trace: List[str]

    # Feedback
    feedback: str            # "correct" / "incorrect" / ""
    feedback_comment: str