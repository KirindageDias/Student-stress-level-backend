import os
from pathlib import Path

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BACKEND_DIR / ".env")


class Settings:
    # backend/app/core/config.py -> backend
    backend_dir = BACKEND_DIR
    model_dir = backend_dir / "models"

    best_model_path = model_dir / "best_stress_model.pkl"
    decision_tree_model_path = model_dir / "decision_tree_model.pkl"
    scaler_path = model_dir / "scaler.pkl"
    label_encoder_path = model_dir / "label_encoder.pkl"
    metadata_path = model_dir / "model_metadata.pkl"
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

    # Comma-separated values can be used later if another frontend URL is needed.
    cors_origins = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
        if origin.strip()
    ]
    cors_origin_regex = os.getenv("CORS_ORIGIN_REGEX")


settings = Settings()
