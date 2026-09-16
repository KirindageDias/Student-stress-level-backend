import os
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from app.core.config import settings
from app.schemas.chat import ChatHistoryItem


CRISIS_TERMS = [
    "suicide",
    "kill myself",
    "end my life",
    "self harm",
    "hurt myself",
    "i want to die",
    "marenna",
    "marennam",
    "marenawa",
]


def detect_safety_level(message: str) -> str:
    lowered = message.lower()
    if any(term in lowered for term in CRISIS_TERMS):
        return "urgent"
    return "supportive"


def build_context(result: Optional[Dict[str, Any]]) -> str:
    if not result:
        return "No assessment result has been provided yet."

    factors = result.get("top_factors", [])
    factor_text = ", ".join(
        f"{factor.get('name')}: {factor.get('score')} ({factor.get('level')})"
        for factor in factors[:6]
        if isinstance(factor, dict)
    )

    return "\n".join(
        [
            f"ML predicted stress level: {result.get('ml_predicted_stress_level') or result.get('predicted_stress_level')}",
            f"Screening stress level: {result.get('screening_stress_level') or result.get('predicted_stress_level')}",
            f"Stress score: {result.get('calculated_stress_score')}/100",
            f"Anxiety score: {result.get('calculated_anxiety_score')}/100",
            f"Well-being score: {result.get('wellbeing_score')}/100",
            f"Model confidence: {result.get('model_confidence')}%",
            f"Top factors: {factor_text or 'Not available'}",
        ]
    )


def build_prompt(message: str, result: Optional[Dict[str, Any]], history: List[ChatHistoryItem]) -> str:
    history_text = "\n".join(f"{item.role}: {item.content}" for item in history[-6:])
    return f"""
You are MindfulFlow Assistant, a student well-being support assistant for an undergraduate research prototype.

Rules:
- Explain the provided assessment result in simple Sinhala, English, or mixed Sinhala-English matching the user's message.
- Do not diagnose medical or mental-health conditions.
- Do not claim the model is clinically validated.
- Give practical, low-risk suggestions about study planning, sleep, breathing, physical activity, social support, and university support.
- Keep answers concise: 4 to 8 short bullet points or short paragraphs.
- If the user mentions self-harm, suicide, danger, or being unable to stay safe, tell them to contact immediate human/emergency support now and encourage a trusted person to stay with them.

Assessment context:
{build_context(result)}

Recent conversation:
{history_text or "No previous chat."}

User message:
{message}
""".strip()


def fallback_reply(message: str, result: Optional[Dict[str, Any]], safety_level: str) -> str:
    if safety_level == "urgent":
        return (
            "I am really sorry you are feeling this much pressure. Please contact emergency support or a trusted person immediately, "
            "and do not stay alone if you feel unsafe. This app is not a crisis service, but your safety matters right now."
        )

    if result:
        level = result.get("ml_predicted_stress_level") or result.get("predicted_stress_level") or "your current"
        anxiety = result.get("calculated_anxiety_score", "N/A")
        wellbeing = result.get("wellbeing_score", "N/A")
        return (
            f"Your result suggests {level}. Anxiety is {anxiety}/100 and well-being is {wellbeing}/100. "
            "A good next step is to choose one small action today: plan the next study block, protect sleep time, "
            "try two minutes of slow breathing, and speak with a trusted person or university support if stress feels heavy."
        )

    return (
        "I can help explain your stress check-in result and suggest a simple action plan. "
        "Complete the assessment first for a more personalized answer."
    )


def generate_chat_reply(message: str, result: Optional[Dict[str, Any]], history: List[ChatHistoryItem]) -> Dict[str, str]:
    safety_level = detect_safety_level(message)
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        return {"reply": fallback_reply(message, result, safety_level), "safety_level": safety_level}

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=build_prompt(message, result, history),
            config=types.GenerateContentConfig(
                temperature=0.35,
                max_output_tokens=650,
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                ],
            ),
        )
        reply = (response.text or "").strip()
        if not reply:
            reply = fallback_reply(message, result, safety_level)
        return {"reply": reply, "safety_level": safety_level}
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Gemini assistant failed: {error}") from error
