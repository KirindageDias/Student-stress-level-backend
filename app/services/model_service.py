from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
from fastapi import HTTPException

from app.core.config import settings
from app.schemas.prediction import QuestionnaireInput


model: Optional[Any] = None
scaler: Optional[Any] = None
label_encoder: Optional[Any] = None


def load_ml_files() -> Tuple[Any, Any, Any]:
    """Load the saved ML files only once, then reuse them for predictions."""
    global model, scaler, label_encoder

    missing_files = [
        path.name
        for path in (
            settings.decision_tree_model_path,
            settings.scaler_path,
            settings.label_encoder_path,
        )
        if not path.exists()
    ]

    if missing_files:
        missing_text = ", ".join(missing_files)
        raise HTTPException(
            status_code=503,
            detail=f"Missing model file(s): {missing_text}. Place them inside backend/models.",
        )

    if model is None:
        model = joblib.load(settings.decision_tree_model_path)
    if scaler is None:
        scaler = joblib.load(settings.scaler_path)
    if label_encoder is None:
        label_encoder = joblib.load(settings.label_encoder_path)

    return model, scaler, label_encoder


def calculate_anxiety_level(data: QuestionnaireInput) -> int:
    anxiety_score = (
        data.worried
        + data.exam_nervous
        + data.worry_control
        + data.sleep_stress
        + data.overwhelmed
    )
    return round((anxiety_score / 25) * 10)


def format_stress_label(raw_label: Any) -> str:
    """Convert model output into labels that are easy for students to read."""
    normalized = str(raw_label).strip().lower()
    label_map = {
        "low": "Low Stress",
        "low stress": "Low Stress",
        "medium": "Moderate Stress",
        "moderate": "Moderate Stress",
        "moderate stress": "Moderate Stress",
        "high": "High Stress",
        "high stress": "High Stress",
    }
    return label_map.get(normalized, str(raw_label))


def predict_student_stress(data: QuestionnaireInput) -> Dict[str, Any]:
    current_model, current_scaler, current_label_encoder = load_ml_files()
    calculated_anxiety_level = calculate_anxiety_level(data)

    # This feature order must match the order used when training the model.
    model_input = np.array(
        [
            [
                calculated_anxiety_level,
                data.sleep_hours,
                data.academic_performance,
                data.study_hours,
                data.self_esteem,
            ]
        ]
    )

    try:
        scaled_input = current_scaler.transform(model_input)
        prediction = current_model.predict(scaled_input)
        try:
            decoded_prediction = current_label_encoder.inverse_transform(prediction)[0]
        except (TypeError, ValueError):
            # Some saved models already return text labels instead of encoded numbers.
            decoded_prediction = prediction[0]
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed. Please check that the saved model files match the expected features. Error: {error}",
        ) from error

    return {
        "calculated_anxiety_level": calculated_anxiety_level,
        "predicted_stress_level": format_stress_label(decoded_prediction),
    }
