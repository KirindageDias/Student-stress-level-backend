from typing import List, Optional

from pydantic import BaseModel, Field


class QuestionnaireInput(BaseModel):
    undergraduate_student: int = Field(..., ge=0, le=1)
    age_group: int = Field(..., ge=1, le=4)
    gender: int = Field(..., ge=1, le=3)
    year_of_study: int = Field(..., ge=1, le=4)
    working_while_studying: int = Field(..., ge=0, le=1)
    academic_performance: int = Field(..., ge=1, le=5)
    study_hours: int = Field(..., ge=1, le=4)
    deadline_pressure: int = Field(..., ge=1, le=5)

    anxiety_1: int = Field(..., ge=0, le=3)
    anxiety_2: int = Field(..., ge=0, le=3)

    stress_1: int = Field(..., ge=0, le=4)
    stress_2: int = Field(..., ge=0, le=4)
    stress_3: int = Field(..., ge=0, le=4)
    stress_4: int = Field(..., ge=0, le=4)

    wellbeing_1: int = Field(..., ge=0, le=5)
    wellbeing_2: int = Field(..., ge=0, le=5)
    wellbeing_3: int = Field(..., ge=0, le=5)
    wellbeing_4: int = Field(..., ge=0, le=5)
    wellbeing_5: int = Field(..., ge=0, le=5)

    sleep_hours: int = Field(..., ge=1, le=4)
    sleep_rested: int = Field(..., ge=1, le=5)
    physical_activity: int = Field(..., ge=1, le=5)
    social_time: int = Field(..., ge=1, le=5)
    financial_pressure: int = Field(..., ge=1, le=5)
    main_stress_reason: Optional[str] = None


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
