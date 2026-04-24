import io
from pathlib import Path

from smart_attendance import create_app


class FakeRecognizer:
    is_ready = True
    load_error = None

    def get_roster(self):
        return [
            {"name": "Anas", "images": 10, "folder": "Anas"},
            {"name": "Mathias", "images": 10, "folder": "Mathias"},
            {"name": "Taha", "images": 10, "folder": "Taha"},
        ]

    def get_known_names(self):
        return ["Anas", "Mathias", "Taha"]

    def predict(self, _image_bytes):
        return {
            "status": "recognized",
            "name": "Anas",
            "best_guess": "Anas",
            "confidence": 0.98,
            "confidence_display": "98.00%",
            "message": "Anas recognized successfully.",
        }


class FakeStore:
    def __init__(self):
        self.records = []

    def get_records(self):
        return self.records

    def get_stats(self):
        return {
            "total_records": len(self.records),
            "today_records": len(self.records),
            "unique_people_today": len({record["Name"] for record in self.records}),
            "last_check_in": self.records[-1] if self.records else None,
            "by_person": [{"Name": "Anas", "days_present": 1}] if self.records else [],
        }

    def process_prediction(self, prediction):
        if prediction["status"] == "recognized":
            row = {
                "Name": prediction["name"],
                "Date": "2026-04-23",
                "Time": "14:00:00",
                "Confidence": "98.00%",
            }
            self.records.append(row)
            return {
                "status": "marked_present",
                "name": prediction["name"],
                "best_guess": prediction["name"],
                "confidence": prediction["confidence"],
                "message": f"Attendance marked for {prediction['name']}.",
                "attendance_marked": True,
            }
        return {
            "status": prediction["status"],
            "name": prediction["name"],
            "best_guess": prediction["best_guess"],
            "confidence": prediction["confidence"],
            "message": prediction["message"],
            "attendance_marked": False,
        }

    def clear(self):
        self.records = []

    def export_csv(self):
        export_path = Path("tests") / "fake_export.csv"
        export_path.write_text("Name,Date,Time,Confidence\n", encoding="utf-8")
        return export_path


def make_client():
    app = create_app(recognizer=FakeRecognizer(), store=FakeStore())
    app.config["TESTING"] = True
    return app.test_client()


def test_index_loads():
    client = make_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"Smart Attendance" in response.data


def test_bootstrap_returns_roster():
    client = make_client()
    response = client.get("/api/bootstrap")
    payload = response.get_json()
    assert response.status_code == 200
    assert payload["model_ready"] is True
    assert len(payload["roster"]) == 3


def test_recognize_marks_attendance():
    client = make_client()
    response = client.post(
        "/api/recognize",
        data={"image": (io.BytesIO(b"fake-image-bytes"), "sample.jpg")},
        content_type="multipart/form-data",
    )
    payload = response.get_json()
    assert response.status_code == 200
    assert payload["result"]["status"] == "marked_present"
    assert payload["stats"]["today_records"] == 1


def test_clear_attendance():
    client = make_client()
    client.post(
        "/api/recognize",
        data={"image": (io.BytesIO(b"fake-image-bytes"), "sample.jpg")},
        content_type="multipart/form-data",
    )
    response = client.post("/api/attendance/clear")
    payload = response.get_json()
    assert response.status_code == 200
    assert payload["attendance"] == []


def test_missing_payload_returns_bad_request():
    client = make_client()
    response = client.post("/api/recognize", json={})
    payload = response.get_json()
    assert response.status_code == 400
    assert payload["ok"] is False
