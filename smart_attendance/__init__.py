from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge

from .attendance_store import AttendanceStore
from .config import Config
from .recognition import RecognitionService
from .utils import ALLOWED_MIME_TYPES, image_bytes_from_data_url


def create_app(recognizer=None, store=None) -> Flask:
    base_dir = Path(__file__).resolve().parent.parent
    app = Flask(
        __name__,
        template_folder=str(base_dir / "templates"),
        static_folder=str(base_dir / "static"),
    )
    app.config.from_object(Config)

    recognizer = recognizer or RecognitionService(app.config)
    store = store or AttendanceStore(app.config)

    def json_error(message: str, status_code: int):
        return jsonify({"ok": False, "message": message}), status_code

    @app.errorhandler(RequestEntityTooLarge)
    def handle_large_upload(_error):
        limit_mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
        return json_error(f"Image is too large. Maximum upload size is {limit_mb} MB.", 413)

    @app.errorhandler(404)
    def handle_not_found(_error):
        if request.path.startswith("/api/"):
            return json_error("The requested API endpoint does not exist.", 404)
        return render_template("index.html", roster=recognizer.get_roster(), threshold=app.config["CONFIDENCE_THRESHOLD"], cooldown_minutes=app.config["CHECKIN_COOLDOWN_MINUTES"]), 404

    @app.errorhandler(500)
    def handle_server_error(_error):
        return json_error("An unexpected server error occurred.", 500)

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            roster=recognizer.get_roster(),
            threshold=app.config["CONFIDENCE_THRESHOLD"],
            cooldown_minutes=app.config["CHECKIN_COOLDOWN_MINUTES"],
        )

    @app.get("/api/bootstrap")
    def bootstrap():
        return jsonify(
            {
                "roster": recognizer.get_roster(),
                "stats": store.get_stats(),
                "attendance": store.get_records(),
                "threshold": app.config["CONFIDENCE_THRESHOLD"],
                "app_env": app.config["APP_ENV"],
                "model_ready": recognizer.is_ready,
            }
        )

    @app.get("/api/attendance")
    def attendance():
        return jsonify(
            {
                "attendance": store.get_records(),
                "stats": store.get_stats(),
            }
        )

    @app.post("/api/attendance/clear")
    def clear_attendance():
        store.clear()
        return jsonify(
            {
                "message": "Attendance log cleared.",
                "attendance": [],
                "stats": store.get_stats(),
            }
        )

    @app.get("/api/attendance/export")
    def export_attendance():
        export_path = store.export_csv()
        return send_file(export_path, as_attachment=True)

    @app.post("/api/recognize")
    def recognize():
        if not recognizer.is_ready:
            return (
                jsonify(
                    {
                        "ok": False,
                        "message": recognizer.load_error or "Recognition models are not loaded.",
                    }
                ),
                503,
            )

        image_bytes = None

        try:
            if "image" in request.files:
                uploaded_file = request.files["image"]
                if uploaded_file.mimetype not in ALLOWED_MIME_TYPES:
                    return json_error("Unsupported image type. Use JPG, PNG, or WEBP.", 415)
                image_bytes = uploaded_file.read()
            elif request.is_json:
                payload = request.get_json(silent=True) or {}
                image_data = payload.get("image_data")
                if image_data:
                    image_bytes = image_bytes_from_data_url(image_data)
        except Exception as exc:
            return json_error(str(exc), 400)

        if not image_bytes:
            return json_error("No image was provided.", 400)

        try:
            prediction = recognizer.predict(image_bytes)
            attendance_result = store.process_prediction(prediction)
        except Exception as exc:
            return json_error(str(exc), 500)

        return jsonify(
            {
                "ok": True,
                "prediction": prediction,
                "result": attendance_result,
                "attendance": store.get_records(),
                "stats": store.get_stats(),
            }
        )

    @app.get("/api/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "model_loaded": recognizer.is_ready,
                "known_people": recognizer.get_known_names(),
                "load_error": recognizer.load_error,
            }
        )

    return app
