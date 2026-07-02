from typing import List

from pydantic import BaseModel, Field


class QuestionnaireInput(BaseModel):
    age_group: int = Field(..., ge=1, le=4)
    year_of_study: int = Field(..., ge=1, le=4)
    working_while_studying: int = Field(..., ge=0, le=1)
    academic_performance: int = Field(..., ge=1, le=5)
    study_hours: int = Field(..., ge=1, le=4)
    sleep_hours: int = Field(..., ge=1, le=4)
    sleep_quality: int = Field(..., ge=1, le=5)
    physical_activity: int = Field(..., ge=1, le=5)
    social_time: int = Field(..., ge=1, le=5)
    financial_pressure: int = Field(..., ge=1, le=5)

    gad_1: int = Field(..., ge=0, le=3)
    gad_2: int = Field(..., ge=0, le=3)
    gad_3: int = Field(..., ge=0, le=3)
    gad_4: int = Field(..., ge=0, le=3)
    gad_5: int = Field(..., ge=0, le=3)
    gad_6: int = Field(..., ge=0, le=3)
    gad_7: int = Field(..., ge=0, le=3)

    pss_1: int = Field(..., ge=0, le=4)
    pss_2: int = Field(..., ge=0, le=4)
    pss_3: int = Field(..., ge=0, le=4)
    pss_4: int = Field(..., ge=0, le=4)
    pss_5: int = Field(..., ge=0, le=4)
    pss_6: int = Field(..., ge=0, le=4)
    pss_7: int = Field(..., ge=0, le=4)
    pss_8: int = Field(..., ge=0, le=4)
    pss_9: int = Field(..., ge=0, le=4)
    pss_10: int = Field(..., ge=0, le=4)

    who_1: int = Field(..., ge=0, le=5)
    who_2: int = Field(..., ge=0, le=5)
    who_3: int = Field(..., ge=0, le=5)
    who_4: int = Field(..., ge=0, le=5)
    who_5: int = Field(..., ge=0, le=5)


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
