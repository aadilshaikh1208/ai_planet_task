# utils/llm.py
# Single function all agents will use to call the LLM.
# FreeFlow automatically switches between Groq → Gemini → GitHub
# when rate limits are hit.

from freeflow_llm import FreeFlowClient, NoProvidersAvailableError

def call_llm(prompt: str, system: str = "You are a helpful math assistant.",max_tokens: int = 500) -> str:
    """
    Call the LLM using FreeFlow.
    Automatically falls back across Groq, Gemini, GitHub if rate limited.

    Args:
        prompt: the user message
        system: system instruction (optional)

    Returns:
        LLM response as a string
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": prompt}
    ]

    try:
        with FreeFlowClient() as client:
            response = client.chat(messages=messages,max_tokens=max_tokens)
            return response.content

    except NoProvidersAvailableError:
        return "Error: All LLM providers are currently rate limited. Please try again in a minute."