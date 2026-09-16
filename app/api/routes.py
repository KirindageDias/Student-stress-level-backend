from typing import Any, Dict

from fastapi import APIRouter

from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.prediction import PredictionResponse, QuestionnaireInput
from app.services.gemini_service import generate_chat_reply
from app.services.model_service import predict_student_stress


router = APIRouter()


@router.get("/")
def health_check() -> Dict[str, str]:
    return {"message": "Student Stress Prediction API is running"}


@router.post("/predict", response_model=PredictionResponse)
def predict_stress_level(data: QuestionnaireInput) -> Dict[str, Any]:
    return predict_student_stress(data)


@router.post("/chat", response_model=ChatResponse)
def chat_with_assistant(data: ChatRequest) -> Dict[str, str]:
    return generate_chat_reply(data.message, data.result, data.history)
