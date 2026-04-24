import pickle
from pathlib import Path

import numpy as np
from deepface import DeepFace

from .utils import count_images, decode_image, discover_person_folders, format_confidence


class RecognitionService:
    def __init__(self, config):
        self.config = config
        self.load_error = None
        self.classifier = None
        self.label_encoder = None

        try:
            self.classifier = self._load_pickle(Path(config["CLASSIFIER_PATH"]))
            self.label_encoder = self._load_pickle(Path(config["LABEL_ENCODER_PATH"]))
        except Exception as exc:
            self.load_error = str(exc)

        self.is_ready = self.classifier is not None and self.label_encoder is not None
        self.known_names = self._load_known_names()

    @staticmethod
    def _load_pickle(path: Path):
        with path.open("rb") as handle:
            return pickle.load(handle)

    def _load_known_names(self) -> list[str]:
        if hasattr(self.label_encoder, "classes_"):
            return [str(name) for name in self.label_encoder.classes_]
        return []

    def get_known_names(self) -> list[str]:
        return self.known_names

    def get_roster(self) -> list[dict]:
        folders = discover_person_folders(Path(self.config["BASE_DIR"]), self.known_names)
        roster = []
        for name in self.known_names:
            folder = folders.get(name)
            roster.append(
                {
                    "name": name,
                    "images": count_images(folder),
                    "folder": str(folder) if folder else None,
                }
            )
        return roster

    def predict(self, image_bytes: bytes) -> dict:
        if not self.is_ready:
            raise RuntimeError("Recognition models are not loaded.")

        image = decode_image(image_bytes)
        rgb_image = image[:, :, ::-1]

        result = DeepFace.represent(
            img_path=rgb_image,
            model_name=self.config["MODEL_NAME"],
            detector_backend=self.config["DETECTOR_BACKEND"],
            enforce_detection=self.config["ENFORCE_DETECTION"],
        )

        if not result:
            return {
                "status": "error",
                "name": None,
                "best_guess": None,
                "confidence": 0.0,
                "message": "No face embedding could be extracted.",
            }

        face_confidence = float(result[0].get("face_confidence", 0.0))
        if face_confidence < 0.5:
            return {
                "status": "unknown",
                "name": None,
                "best_guess": None,
                "confidence": 0.0,
                "message": "No face detected in the image.",
            }

        embedding = np.array(result[0]["embedding"], dtype=np.float32).reshape(1, -1)
        probabilities = self.classifier.predict_proba(embedding)[0]
        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])
        predicted_name = str(self.label_encoder.inverse_transform([best_index])[0])

        if confidence >= self.config["CONFIDENCE_THRESHOLD"]:
            return {
                "status": "recognized",
                "name": predicted_name,
                "best_guess": predicted_name,
                "confidence": confidence,
                "confidence_display": format_confidence(confidence),
                "message": f"{predicted_name} recognized successfully.",
            }

        return {
            "status": "unknown",
            "name": None,
            "best_guess": predicted_name,
            "confidence": confidence,
            "confidence_display": format_confidence(confidence),
            "message": f"Confidence below threshold. Best guess: {predicted_name}.",
        }
