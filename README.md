# Smart Attendance Web App

This project turns the original Kaggle notebook into a local face-recognition attendance application with a real browser UI, persistent CSV logging, and a production-ready Python server setup.

## Features

- Reuses your exported ArcFace + SVM model from `models/face_classifier.pkl`
- Reuses your saved label encoder from `models/label_encoder.pkl`
- Supports webcam capture and uploaded photos
- Prevents duplicate attendance entries for the same person on the same date
- Stores attendance in `data/attendance.csv`
- Exports attendance snapshots to `data/exports/`
- Shows live dashboard stats, roster counts, and per-person attendance summary
- Includes smoke tests for the Flask app
- Includes a production entrypoint with Waitress for Windows-friendly deployment

## Project structure

```text
Smart Attendence/
├── app.py
├── wsgi.py
├── requirements.txt
├── .env.example
├── README.md
├── models/
├── data/
├── templates/
├── static/
├── smart_attendance/
├── tests/
├── Anas/
├── Mathias/
└── Taha/
```

## Local development

1. Create and activate a virtual environment.
2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Start the development server:

```powershell
python app.py
```

4. Open [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Production run

Use Waitress instead of Flask's debug server:

```powershell
python wsgi.py
```

This uses the host and port from configuration and avoids running the Werkzeug debugger in production.

## Configuration

The app reads optional environment variables:

- `SMART_ATTENDANCE_ENV`
- `SMART_ATTENDANCE_DEBUG`
- `SMART_ATTENDANCE_HOST`
- `SMART_ATTENDANCE_PORT`
- `SMART_ATTENDANCE_CONFIDENCE_THRESHOLD`
- `SMART_ATTENDANCE_MAX_UPLOAD_MB`
- `SMART_ATTENDANCE_DATA_DIR`
- `SMART_ATTENDANCE_MODELS_DIR`
- `SMART_ATTENDANCE_CLASSIFIER_PATH`
- `SMART_ATTENDANCE_LABEL_ENCODER_PATH`

You can start from `.env.example` and set values manually in your shell.

## Testing

Run the smoke tests with:

```powershell
python -m pytest
```

## Recognition pipeline

The prediction path mirrors the notebook:

- `DeepFace.represent(...)`
- `model_name="ArcFace"`
- `detector_backend="opencv"`
- `enforce_detection=False`
- SVM `predict_proba(...)`
- confidence threshold defaulting to `0.55`

## Notes

- The first launch may take longer because DeepFace can initialize local cache files.
- TensorFlow warnings may still appear in the terminal; they do not usually block the app.
- The current project is organized so a future retraining script can be added cleanly.
