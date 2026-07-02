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
    "age_group",
    "year_of_study",
    "working_while_studying",
    "academic_performance",
    "study_hours",
    "deadline_stress",
    "worried",
    "exam_nervous",
    "worry_control",
    "sleep_hours",
    "sleep_quality",
    "sleep_stress",
    "self_esteem",
    "life_satisfaction",
    "overwhelmed",
    "physical_activity",
    "social_time",
    "financial_stress",
]

ENGINEERED_COLUMNS = [
    "anxiety_score",
    "academic_pressure_score",
    "sleep_risk_score",
    "wellbeing_score",
    "lifestyle_support_score",
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
        missing_text = ", ".join(missing_files)
        raise HTTPException(
            status_code=503,
            detail=f"Missing model file(s): {missing_text}. Train models and copy them into backend/models.",
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


def normalize_1_to_5(value: float) -> float:
    return ((value - 1) / 4) * 100


def level_from_score(score: float, reverse: bool = False) -> str:
    value = 100 - score if reverse else score
    if value < 32:
        return "Low"
    if value < 56:
        return "Moderate"
    if value < 76:
        return "High"
    return "Extreme"


def calculate_features(data: QuestionnaireInput) -> Dict[str, float]:
    raw = data.dict()
    anxiety_score = float(
        np.mean(
            [
                normalize_1_to_5(data.worried),
                normalize_1_to_5(data.exam_nervous),
                normalize_1_to_5(data.worry_control),
                normalize_1_to_5(data.overwhelmed),
            ]
        )
    )
    academic_pressure_score = float(
        np.mean(
            [
                normalize_1_to_5(data.deadline_stress),
                normalize_1_to_5(data.study_hours),
                normalize_1_to_5(6 - data.academic_performance),
            ]
        )
    )
    sleep_duration_risk = {1: 100, 2: 70, 3: 25, 4: 35}[data.sleep_hours]
    sleep_risk_score = float(
        np.mean(
            [
                sleep_duration_risk,
                normalize_1_to_5(6 - data.sleep_quality),
                normalize_1_to_5(data.sleep_stress),
            ]
        )
    )
    wellbeing_score = float(
        np.mean(
            [
                normalize_1_to_5(data.self_esteem),
                normalize_1_to_5(data.life_satisfaction),
            ]
        )
    )
    lifestyle_support_score = float(
        np.mean(
            [
                normalize_1_to_5(data.physical_activity),
                normalize_1_to_5(data.social_time),
            ]
        )
    )

    raw.update(
        {
            "anxiety_score": round(anxiety_score, 2),
            "academic_pressure_score": round(academic_pressure_score, 2),
            "sleep_risk_score": round(sleep_risk_score, 2),
            "wellbeing_score": round(wellbeing_score, 2),
            "lifestyle_support_score": round(lifestyle_support_score, 2),
        }
    )
    return raw


def calculate_stress_score(features: Dict[str, float]) -> int:
    score = (
        features["anxiety_score"] * 0.32
        + features["academic_pressure_score"] * 0.20
        + features["sleep_risk_score"] * 0.18
        + normalize_1_to_5(features["financial_stress"]) * 0.14
        + normalize_1_to_5(features["overwhelmed"]) * 0.10
        + (100 - features["wellbeing_score"]) * 0.04
        + (100 - features["lifestyle_support_score"]) * 0.02
    )
    if features["working_while_studying"] == 1:
        score += 4
    if features["year_of_study"] >= 3:
        score += 2
    return int(np.clip(round(score), 0, 100))


def format_stress_label(raw_label: Any) -> str:
    normalized = str(raw_label).strip().lower()
    label_map = {
        "low": "Low Stress",
        "moderate": "Moderate Stress",
        "medium": "Moderate Stress",
        "high": "High Stress",
        "extreme": "Extreme Stress",
    }
    return label_map.get(normalized, str(raw_label))


def get_model_confidence(current_model: Any, model_input: np.ndarray, prediction: Any) -> int:
    if hasattr(current_model, "predict_proba"):
        probabilities = current_model.predict_proba(model_input)[0]
        return int(round(float(np.max(probabilities)) * 100))

    if hasattr(current_model, "named_steps") and hasattr(current_model.named_steps.get("model"), "predict_proba"):
        probabilities = current_model.predict_proba(model_input)[0]
        return int(round(float(np.max(probabilities)) * 100))

    return 70 if prediction is not None else 0


def build_top_factors(features: Dict[str, float]) -> List[Dict[str, Any]]:
    factor_scores = [
        ("Anxiety", features["anxiety_score"], False),
        ("Academic pressure", features["academic_pressure_score"], False),
        ("Sleep risk", features["sleep_risk_score"], False),
        ("Financial pressure", normalize_1_to_5(features["financial_stress"]), False),
        ("Well-being protection", features["wellbeing_score"], True),
        ("Lifestyle and social support", features["lifestyle_support_score"], True),
    ]
    ranked = sorted(factor_scores, key=lambda item: 100 - item[1] if item[2] else item[1], reverse=True)
    return [
        {
            "name": name,
            "score": int(round(score)),
            "level": level_from_score(score, reverse=reverse),
        }
        for name, score, reverse in ranked
    ]


def add_recommendation(items: List[Dict[str, str]], title: str, category: str, priority: str, detail: str):
    items.append({"title": title, "category": category, "priority": priority, "detail": detail})


def build_recommendations(features: Dict[str, float], stress_score: int) -> List[Dict[str, str]]:
    recommendations: List[Dict[str, str]] = []

    if features["anxiety_score"] >= 55:
        add_recommendation(
            recommendations,
            "Use a two-minute reset before study or work",
            "Anxiety",
            "High",
            "Try box breathing: inhale 4 seconds, hold 4, exhale 4, hold 4. Repeat 4 rounds before lectures, shifts, exams, or presentations.",
        )
    if features["sleep_risk_score"] >= 50:
        add_recommendation(
            recommendations,
            "Protect a consistent sleep window",
            "Sleep",
            "High",
            "Keep the same sleep and wake time on most days, reduce screens and caffeine late evening, and write tomorrow's top tasks before bed to reduce overthinking.",
        )
    if features["academic_pressure_score"] >= 50:
        add_recommendation(
            recommendations,
            "Convert deadlines into small visible tasks",
            "Academic",
            "High",
            "Plan three must-do tasks per day. Use 50-minute focus blocks with 10-minute breaks, and start assignments with the easiest 15-minute section.",
        )
    if features["financial_stress"] >= 4 or features["working_while_studying"] == 1:
        add_recommendation(
            recommendations,
            "Balance work shifts with recovery blocks",
            "Work and finance",
            "Medium",
            "Reserve at least one non-work recovery block each week. Track transport, food, and course expenses weekly so money stress becomes visible and manageable.",
        )
    if features["physical_activity"] <= 2:
        add_recommendation(
            recommendations,
            "Start with movement that fits student life",
            "Exercise",
            "Medium",
            "Walk briskly for 10-15 minutes after lectures or work 4 days a week. Add two short bodyweight sessions: squats, wall push-ups, and planks.",
        )
    if features["social_time"] <= 2:
        add_recommendation(
            recommendations,
            "Schedule one support contact",
            "Social support",
            "Medium",
            "Message or call one trusted friend or family member twice a week. Keep it simple: one meal, walk, or 10-minute call is enough to reduce isolation.",
        )
    if features["wellbeing_score"] < 45:
        add_recommendation(
            recommendations,
            "Add one confidence-building routine",
            "Well-being",
            "Medium",
            "At the end of each day, write one completed task and one next small step. This helps rebuild control when responsibilities feel heavy.",
        )
    if stress_score >= 76:
        add_recommendation(
            recommendations,
            "Speak to a human support service",
            "Safety",
            "Urgent",
            "Because the stress estimate is very high, consider contacting a university counselor, trusted lecturer, doctor, or family member soon.",
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
    current_model, current_label_encoder, current_metadata = load_ml_files()
    features = calculate_features(data)
    feature_columns = current_metadata.get("feature_columns", DEFAULT_FEATURE_COLUMNS)
    model_input = np.array([[features[column] for column in feature_columns]])

    try:
        prediction = current_model.predict(model_input)
        decoded_prediction = current_label_encoder.inverse_transform(prediction)[0]
        confidence = get_model_confidence(current_model, model_input, prediction)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed. Check that backend models match the current feature schema. Error: {error}",
        ) from error

    stress_score = calculate_stress_score(features)

    return {
        "predicted_stress_level": format_stress_label(decoded_prediction),
        "calculated_stress_score": stress_score,
        "calculated_anxiety_score": int(round(features["anxiety_score"])),
        "wellbeing_score": int(round(features["wellbeing_score"])),
        "model_confidence": confidence,
        "top_factors": build_top_factors(features),
        "recommendations": build_recommendations(features, stress_score),
        "safety_note": "This tool is a research and self-care support prototype, not a medical diagnosis. If stress feels unmanageable or unsafe, contact a qualified professional or emergency support.",
    }
