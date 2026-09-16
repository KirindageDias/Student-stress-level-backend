import os
from pathlib import Path

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BACKEND_DIR / ".env")

DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "https://student-stress-level-frontend-9vf6oj8ms-sewmini-s-projects.vercel.app",
]


def parse_cors_origins(value: str) -> list[str]:
    return [
        origin.strip().rstrip("/")
        for origin in value.split(",")
        if origin.strip()
    ]


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
    cors_origins = parse_cors_origins(os.getenv("CORS_ORIGINS", ",".join(DEFAULT_CORS_ORIGINS)))
    cors_origin_regex = os.getenv("CORS_ORIGIN_REGEX", r"https://.*\.vercel\.app")


settings = Settings()
