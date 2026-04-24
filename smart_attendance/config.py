import os
from pathlib import Path


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODELS_DIR = Path(os.getenv("SMART_ATTENDANCE_MODELS_DIR", str(BASE_DIR / "models")))
    DATA_DIR = Path(os.getenv("SMART_ATTENDANCE_DATA_DIR", str(BASE_DIR / "data")))
    EXPORTS_DIR = DATA_DIR / "exports"

    CLASSIFIER_PATH = Path(os.getenv("SMART_ATTENDANCE_CLASSIFIER_PATH", str(MODELS_DIR / "face_classifier.pkl")))
    LABEL_ENCODER_PATH = Path(os.getenv("SMART_ATTENDANCE_LABEL_ENCODER_PATH", str(MODELS_DIR / "label_encoder.pkl")))
    ATTENDANCE_FILE = Path(os.getenv("SMART_ATTENDANCE_FILE", str(DATA_DIR / "attendance.csv")))

    MODEL_NAME = os.getenv("SMART_ATTENDANCE_MODEL_NAME", "ArcFace")
    DETECTOR_BACKEND = os.getenv("SMART_ATTENDANCE_DETECTOR_BACKEND", "opencv")
    ENFORCE_DETECTION = _get_bool("SMART_ATTENDANCE_ENFORCE_DETECTION", False)
    CONFIDENCE_THRESHOLD = _get_float("SMART_ATTENDANCE_CONFIDENCE_THRESHOLD", 0.55)
    CHECKIN_COOLDOWN_MINUTES = _get_int("SMART_ATTENDANCE_COOLDOWN_MINUTES", 120)

    MAX_CONTENT_LENGTH = _get_int("SMART_ATTENDANCE_MAX_UPLOAD_MB", 5) * 1024 * 1024
    SECRET_KEY = os.getenv("SMART_ATTENDANCE_SECRET_KEY", "smart-attendance-local-dev")
    APP_ENV = os.getenv("SMART_ATTENDANCE_ENV", "development")
    HOST = os.getenv("SMART_ATTENDANCE_HOST", "127.0.0.1")
    PORT = _get_int("SMART_ATTENDANCE_PORT", 5000)
    DEBUG = _get_bool("SMART_ATTENDANCE_DEBUG", True)
