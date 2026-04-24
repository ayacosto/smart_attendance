"""
Retrain the face recognition model.

Usage:
    python retrain.py

Add a new person by creating a folder with their name at the project root,
place their photos inside (or in a subfolder), then run this script.
"""

import os
import pickle
from pathlib import Path

import numpy as np
from deepface import DeepFace
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC

# ── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
CLASSIFIER_PATH = MODELS_DIR / "face_classifier.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"

MODEL_NAME = "ArcFace"
DETECTOR = "opencv"

# Folders that are NOT person folders
SKIP_DIRS = {
    "smart_attendance", "models", "data", "static", "templates",
    "tests", ".git", "__pycache__", ".venv", "venv", "node_modules",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# ── Discover person folders ───────────────────────────────────────────────────

def find_person_folders() -> dict[str, Path]:
    people = {}
    for entry in sorted(BASE_DIR.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith(".") or entry.name in SKIP_DIRS:
            continue
        subfolders = sorted([p for p in entry.iterdir() if p.is_dir()])
        img_folder = subfolders[0] if subfolders else entry
        images = [f for f in img_folder.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
        if images:
            people[entry.name] = img_folder
    return people

# ── Extract embeddings ────────────────────────────────────────────────────────

def extract_embeddings(person_folders: dict[str, Path]) -> tuple[np.ndarray, list[str]]:
    X, y = [], []
    for person, folder in person_folders.items():
        images = [f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
        print(f"\nProcessing {person} ({len(images)} images from {folder.name}/)...")
        ok = 0
        for img_path in images:
            try:
                result = DeepFace.represent(
                    img_path=str(img_path),
                    model_name=MODEL_NAME,
                    detector_backend=DETECTOR,
                    enforce_detection=False,
                )
                emb = result[0]["embedding"]
                if emb and len(emb) > 0:
                    X.append(emb)
                    y.append(person)
                    ok += 1
            except Exception as exc:
                print(f"  skipped {img_path.name}: {exc}")
        print(f"  {ok}/{len(images)} embeddings extracted")
    return np.array(X), y

# ── Train ─────────────────────────────────────────────────────────────────────

def train(X: np.ndarray, y: list[str]):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    pipeline = Pipeline([
        ("normalizer", Normalizer(norm="l2")),
        ("svm", SVC(kernel="linear", C=1.0, probability=True)),
    ])

    n_classes = len(set(y))
    min_per_class = min(np.bincount(y_enc))

    if X.shape[0] >= 4 and min_per_class >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )
        pipeline.fit(X_train, y_train)
        print(f"\nTraining accuracy : {pipeline.score(X_train, y_train):.2%}")
        print(f"Test accuracy     : {pipeline.score(X_test, y_test):.2%}")
    else:
        print("\nSmall dataset — training on all samples (no test split)")
        pipeline.fit(X, y_enc)
        print(f"Training accuracy: {pipeline.score(X, y_enc):.2%}")

    return pipeline, le

# ── Save ──────────────────────────────────────────────────────────────────────

def save_models(pipeline, le):
    MODELS_DIR.mkdir(exist_ok=True)
    with open(CLASSIFIER_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)
    print(f"\nSaved: {CLASSIFIER_PATH}")
    print(f"Saved: {LABEL_ENCODER_PATH}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Smart Attendance — Retrain ===\n")

    person_folders = find_person_folders()
    if not person_folders:
        print("No person folders found. Create a folder with your name and add photos.")
        return

    print(f"Found {len(person_folders)} people: {', '.join(person_folders)}")

    X, y = extract_embeddings(person_folders)

    if len(X) == 0:
        print("\nNo embeddings extracted. Check your images.")
        return

    print(f"\nTotal embeddings: {len(X)} across {len(set(y))} people")

    pipeline, le = train(X, y)

    save_models(pipeline, le)

    print("\nDone! Restart the app for changes to take effect:")
    print("  python app.py")


if __name__ == "__main__":
    main()
