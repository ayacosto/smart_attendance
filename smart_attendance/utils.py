import base64
import binascii
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def image_bytes_from_data_url(data_url: str) -> bytes:
    if "," not in data_url:
        raise ValueError("Invalid image payload.")
    _, encoded = data_url.split(",", 1)
    try:
        return base64.b64decode(encoded)
    except binascii.Error as exc:
        raise ValueError("Invalid image payload.") from exc


def decode_image(image_bytes: bytes) -> np.ndarray:
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode the uploaded image.")
    return image


def now_strings() -> tuple[str, str]:
    now = datetime.now()
    return now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")


def format_confidence(value: float) -> str:
    return f"{value:.2%}"


def discover_person_folders(base_dir: Path, names: list[str]) -> dict[str, Path | None]:
    mapping: dict[str, Path | None] = {}

    for name in names:
        root = base_dir / name
        if not root.exists() or not root.is_dir():
            mapping[name] = None
            continue

        subfolders = sorted([path for path in root.iterdir() if path.is_dir()])
        mapping[name] = subfolders[0] if subfolders else root

    return mapping


def count_images(folder: Path | None) -> int:
    if folder is None:
        return 0
    return sum(1 for file in folder.iterdir() if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS)
