from typing import List

from pydantic import BaseModel, Field


class QuestionnaireInput(BaseModel):
    age_group: int = Field(..., ge=1, le=4)
    year_of_study: int = Field(..., ge=1, le=4)
    working_while_studying: int = Field(..., ge=0, le=1)
    academic_performance: int = Field(..., ge=1, le=5)
    study_hours: int = Field(..., ge=1, le=4)
    deadline_stress: int = Field(..., ge=1, le=5)
    worried: int = Field(..., ge=1, le=5)
    exam_nervous: int = Field(..., ge=1, le=5)
    worry_control: int = Field(..., ge=1, le=5)
    sleep_hours: int = Field(..., ge=1, le=4)
    sleep_quality: int = Field(..., ge=1, le=5)
    sleep_stress: int = Field(..., ge=1, le=5)
    self_esteem: int = Field(..., ge=1, le=5)
    life_satisfaction: int = Field(..., ge=1, le=5)
    overwhelmed: int = Field(..., ge=1, le=5)
    physical_activity: int = Field(..., ge=1, le=5)
    social_time: int = Field(..., ge=1, le=5)
    financial_stress: int = Field(..., ge=1, le=5)


class FactorScore(BaseModel):
    name: str
    score: int
    level: str


class Recommendation(BaseModel):
    title: str
    category: str
    priority: str
    detail: str


class PredictionResponse(BaseModel):
    predicted_stress_level: str
    calculated_stress_score: int
    calculated_anxiety_score: int
    wellbeing_score: int
    model_confidence: int
    top_factors: List[FactorScore]
    recommendations: List[Recommendation]
    safety_note: str
