import os
from pathlib import Path


class Settings:
    # backend/app/core/config.py -> backend
    backend_dir = Path(__file__).resolve().parents[2]
    model_dir = backend_dir / "models"

    decision_tree_model_path = model_dir / "decision_tree_model.pkl"
    scaler_path = model_dir / "scaler.pkl"
    label_encoder_path = model_dir / "label_encoder.pkl"

    # Comma-separated values can be used later if another frontend URL is needed.
    cors_origins = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
        if origin.strip()
    ]


settings = Settings()
