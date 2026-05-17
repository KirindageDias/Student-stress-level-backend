from pydantic import BaseModel, Field


class QuestionnaireInput(BaseModel):
    # These fields are the 11 questionnaire answers sent by the frontend.
    academic_performance: int = Field(..., ge=1, le=5)
    study_hours: int = Field(..., ge=1, le=4)
    deadline_stress: int = Field(..., ge=1, le=5)
    worried: int = Field(..., ge=1, le=5)
    exam_nervous: int = Field(..., ge=1, le=5)
    worry_control: int = Field(..., ge=1, le=5)
    sleep_hours: int = Field(..., ge=1, le=5)
    sleep_stress: int = Field(..., ge=1, le=5)
    self_esteem: int = Field(..., ge=1, le=5)
    overwhelmed: int = Field(..., ge=1, le=5)
    financial_stress: int = Field(..., ge=1, le=5)


class PredictionResponse(BaseModel):
    calculated_anxiety_level: int
    predicted_stress_level: str
