from typing import Any, Dict

from fastapi import APIRouter

from app.schemas.prediction import PredictionResponse, QuestionnaireInput
from app.services.model_service import predict_student_stress


router = APIRouter()


@router.get("/")
def health_check() -> Dict[str, str]:
    return {"message": "Student Stress Prediction API is running"}


@router.post("/predict", response_model=PredictionResponse)
def predict_stress_level(data: QuestionnaireInput) -> Dict[str, Any]:
    return predict_student_stress(data)
