from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChatbotResponse:
    text: str
    blocked: bool = False


WATER_DOMAIN_SYSTEM_RULES = (
    "You are an assistant embedded in a web tool for seawater desalination research "
    "focused on single-layer graphene nanopore membranes. "
    "Only answer questions related to water desalination, membrane science, "
    "nanoporous graphene, and interpreting model outputs. "
    "If the user asks about unrelated topics (sports, politics, etc.), refuse briefly "
    "and steer back to desalination/membranes. "
    "You must support Arabic, French, and English. Answer in the same language as the user."
)


def answer_user_message(message: str, *, api_key: str | None = None, model: str = "gemini-2.0-flash") -> ChatbotResponse:
    """
    Gemini-backed chatbot (preferred). If api_key is missing, falls back to a local rule-based response.
    """
    lower = message.lower().strip()
    blocked_keywords = ["football", "soccer", "politic", "election", "gym", "workout", "basketball"]
    if any(k in lower for k in blocked_keywords):
        return ChatbotResponse(
            text="I can only help with seawater desalination and graphene nanopore membrane topics inside this tool.",
            blocked=True,
        )

    if not api_key:
        return ChatbotResponse(
            text=(
                "AI chatbot is not configured yet. Add `GEMINI_API_KEY` in Streamlit secrets, "
                "then I can answer desalination/graphene membrane questions in Arabic/French/English."
            ),
            blocked=False,
        )

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        m = genai.GenerativeModel(model_name=model, system_instruction=WATER_DOMAIN_SYSTEM_RULES)
        resp = m.generate_content(message)
        text = (getattr(resp, "text", None) or "").strip()
        if not text:
            text = "I couldn't generate a response. Please rephrase your desalination/membrane question."
        return ChatbotResponse(text=text, blocked=False)
    except Exception:
        # Fallback if library/model fails at runtime
        return ChatbotResponse(
            text=(
                "Chatbot is temporarily unavailable. Please try again later, or ask a specific question "
                "about salt rejection, water flux, or graphene nanopore membranes."
            ),
            blocked=False,
        )

