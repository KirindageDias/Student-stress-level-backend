from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from fastapi import HTTPException

from app.core.config import settings
from app.schemas.prediction import QuestionnaireInput


model: Optional[Any] = None
label_encoder: Optional[Any] = None
metadata: Optional[Dict[str, Any]] = None

RAW_COLUMNS = [
    "undergraduate_student",
    "age_group",
    "gender",
    "year_of_study",
    "working_while_studying",
    "academic_performance",
    "study_hours",
    "deadline_pressure",
    "anxiety_1",
    "anxiety_2",
    "wellbeing_1",
    "wellbeing_2",
    "wellbeing_3",
    "wellbeing_4",
    "wellbeing_5",
    "sleep_hours",
    "sleep_rested",
    "physical_activity",
    "social_time",
    "financial_pressure",
]

ENGINEERED_COLUMNS = [
    "anxiety_score",
    "anxiety_percent",
    "wellbeing_raw",
    "wellbeing_percent",
    "academic_pressure_score",
    "sleep_risk_score",
    "lifestyle_support_score",
    "financial_pressure_score",
]

DEFAULT_FEATURE_COLUMNS = RAW_COLUMNS + ENGINEERED_COLUMNS


def load_ml_files() -> Tuple[Any, Any, Dict[str, Any]]:
    global model, label_encoder, metadata

    missing_files = [
        path.name
        for path in (settings.best_model_path, settings.label_encoder_path)
        if not path.exists()
    ]
    if missing_files:
        raise HTTPException(
            status_code=503,
            detail=f"Missing model file(s): {', '.join(missing_files)}. Train models and copy them into backend/models.",
        )

    if model is None:
        model = joblib.load(settings.best_model_path)
    if label_encoder is None:
        label_encoder = joblib.load(settings.label_encoder_path)
    if metadata is None:
        metadata = (
            joblib.load(settings.metadata_path)
            if settings.metadata_path.exists()
            else {"feature_columns": DEFAULT_FEATURE_COLUMNS}
        )

    return model, label_encoder, metadata


def normalize(value: float, maximum: float) -> float:
    return (value / maximum) * 100


def normalize_1_to_5(value: float) -> float:
    return ((value - 1) / 4) * 100


def reverse_stress(value: int) -> int:
    return 4 - value


def stress_level(score: int) -> str:
    if score <= 5:
        return "Low Stress"
    if score <= 10:
        return "Moderate Stress"
    return "High Stress"


def score_level_0_100(score: float, reverse: bool = False) -> str:
    value = 100 - score if reverse else score
    if value < 34:
        return "Low"
    if value < 67:
        return "Moderate"
    return "High"


def calculate_features(data: QuestionnaireInput) -> Dict[str, float]:
    raw = data.dict()

    anxiety_score = data.anxiety_1 + data.anxiety_2
    stress_score = data.stress_1 + reverse_stress(data.stress_2) + reverse_stress(data.stress_3) + data.stress_4
    wellbeing_raw = sum(
        [
            data.wellbeing_1,
            data.wellbeing_2,
            data.wellbeing_3,
            data.wellbeing_4,
            data.wellbeing_5,
        ]
    )

    academic_pressure_score = np.mean(
        [
            normalize_1_to_5(data.study_hours),
            normalize_1_to_5(6 - data.academic_performance),
            normalize_1_to_5(data.deadline_pressure),
            55 if data.year_of_study >= 3 else 30,
        ]
    )
    sleep_duration_risk = {1: 100, 2: 70, 3: 20, 4: 35}[data.sleep_hours]
    sleep_risk_score = np.mean([sleep_duration_risk, normalize_1_to_5(6 - data.sleep_rested)])
    lifestyle_support_score = np.mean(
        [normalize_1_to_5(data.physical_activity), normalize_1_to_5(data.social_time)]
    )

    raw.update(
        {
            "anxiety_score": anxiety_score,
            "anxiety_percent": round(normalize(anxiety_score, 6), 2),
            "stress_score": stress_score,
            "stress_percent": round(normalize(stress_score, 16), 2),
            "stress_level": stress_level(stress_score),
            "wellbeing_raw": wellbeing_raw,
            "wellbeing_percent": wellbeing_raw * 4,
            "academic_pressure_score": round(academic_pressure_score, 2),
            "sleep_risk_score": round(sleep_risk_score, 2),
            "lifestyle_support_score": round(lifestyle_support_score, 2),
            "financial_pressure_score": round(normalize_1_to_5(data.financial_pressure), 2),
        }
    )
    return raw


def get_model_confidence(current_model: Any, model_input: np.ndarray, target_label: str) -> int:
    if not hasattr(current_model, "predict_proba"):
        return 70

    probabilities = current_model.predict_proba(model_input)[0]
    model_classes = current_model.classes_
    try:
        encoded_target = label_encoder.transform([target_label])[0] if label_encoder else None
        class_index = list(model_classes).index(encoded_target)
        return int(round(float(probabilities[class_index]) * 100))
    except (ValueError, TypeError):
        return int(round(float(np.max(probabilities)) * 100))


def build_top_factors(features: Dict[str, float]) -> List[Dict[str, Any]]:
    factor_scores = [
        ("Anxiety", features["anxiety_percent"], False),
        ("Academic pressure", features["academic_pressure_score"], False),
        ("Sleep risk", features["sleep_risk_score"], False),
        ("Financial pressure", features["financial_pressure_score"], False),
        ("Well-being", features["wellbeing_percent"], True),
        ("Lifestyle and social support", features["lifestyle_support_score"], True),
    ]
    ranked = sorted(factor_scores, key=lambda item: 100 - item[1] if item[2] else item[1], reverse=True)
    return [
        {"name": name, "score": int(round(score)), "level": score_level_0_100(score, reverse=reverse)}
        for name, score, reverse in ranked
    ]


def add_recommendation(items: List[Dict[str, str]], title: str, category: str, priority: str, detail: str):
    items.append({"title": title, "category": category, "priority": priority, "detail": detail})


def build_recommendations(features: Dict[str, float]) -> List[Dict[str, str]]:
    recommendations: List[Dict[str, str]] = []

    if features["anxiety_score"] >= 3:
        add_recommendation(
            recommendations,
            "Use a short anxiety reset before demanding tasks",
            "Anxiety",
            "High",
            "Try box breathing for two minutes before exams, presentations, work shifts, or difficult study blocks.",
        )
    if features["stress_score"] >= 11:
        add_recommendation(
            recommendations,
            "Speak to a human support service",
            "Stress",
            "Urgent",
            "Your stress score is in the high range. Consider contacting a university counselor, trusted lecturer, doctor, or family member soon.",
        )
    if features["sleep_risk_score"] >= 55:
        add_recommendation(
            recommendations,
            "Protect a consistent sleep window",
            "Sleep",
            "High",
            "Keep a similar sleep and wake time on most days, reduce caffeine late evening, and write tomorrow's top tasks before bed.",
        )
    if features["academic_pressure_score"] >= 55:
        add_recommendation(
            recommendations,
            "Break academic pressure into visible tasks",
            "Academic",
            "Medium",
            "Plan three must-do tasks per day. Use 50-minute focus blocks with 10-minute breaks and start assignments with a 15-minute first step.",
        )
    if features["financial_pressure"] >= 4 or features["working_while_studying"] == 1:
        add_recommendation(
            recommendations,
            "Balance work shifts with recovery blocks",
            "Work and finance",
            "Medium",
            "Reserve at least one non-work recovery block weekly and track food, transport, and course expenses in a simple weekly budget.",
        )
    if features["physical_activity"] <= 2:
        add_recommendation(
            recommendations,
            "Start with movement that fits student life",
            "Exercise",
            "Medium",
            "Walk briskly for 10-15 minutes after lectures or work 4 days a week. Add two short bodyweight sessions when possible.",
        )
    if features["social_time"] <= 2:
        add_recommendation(
            recommendations,
            "Schedule one support contact",
            "Social support",
            "Medium",
            "Message or call one trusted friend or family member twice a week. Keep it simple: one meal, walk, or 10-minute call.",
        )
    if features["wellbeing_raw"] < 13:
        add_recommendation(
            recommendations,
            "Add one well-being recovery habit",
            "Well-being",
            "Medium",
            "At the end of each day, write one completed task and one next small step. Low well-being scores are a signal to increase support and recovery.",
        )

    add_recommendation(
        recommendations,
        "Use a simple food pattern during busy weeks",
        "Food",
        "Low",
        "Aim for rice or whole grains plus dhal, beans, eggs, fish, chicken, or soya with vegetables. Keep water nearby and avoid relying on energy drinks.",
    )

    return recommendations[:7]


def predict_student_stress(data: QuestionnaireInput) -> Dict[str, Any]:
    current_model, _, current_metadata = load_ml_files()
    features = calculate_features(data)
    feature_columns = current_metadata.get("feature_columns", DEFAULT_FEATURE_COLUMNS)
    model_input = np.array([[features[column] for column in feature_columns]])
    target_label = features["stress_level"].replace(" Stress", "")

    try:
        current_model.predict(model_input)
        confidence = get_model_confidence(current_model, model_input, target_label)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed. Check that backend models match the current feature schema. Error: {error}",
        ) from error

    return {
        "predicted_stress_level": features["stress_level"],
        "calculated_stress_score": int(round(features["stress_percent"])),
        "calculated_anxiety_score": int(round(features["anxiety_percent"])),
        "wellbeing_score": int(round(features["wellbeing_percent"])),
        "model_confidence": confidence,
        "top_factors": build_top_factors(features),
        "recommendations": build_recommendations(features),
        "safety_note": "This tool is for research and self-care support, not clinical diagnosis. If stress feels unmanageable or unsafe, contact a qualified professional or emergency support.",
    }
