from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from .utils import ensure_dir, format_confidence, now_strings


class AttendanceStore:
    COLUMNS = ["Name", "Date", "Time", "Confidence"]

    def __init__(self, config):
        self.config = config
        ensure_dir(Path(config["DATA_DIR"]))
        ensure_dir(Path(config["EXPORTS_DIR"]))
        self.file_path = Path(config["ATTENDANCE_FILE"])
        self._ensure_file()

    def _ensure_file(self):
        if not self.file_path.exists():
            pd.DataFrame(columns=self.COLUMNS).to_csv(self.file_path, index=False)

    def _load(self) -> pd.DataFrame:
        df = pd.read_csv(self.file_path)
        for column in self.COLUMNS:
            if column not in df.columns:
                df[column] = ""
        return df[self.COLUMNS]

    def _save(self, df: pd.DataFrame):
        df.to_csv(self.file_path, index=False)

    def get_records(self) -> list[dict]:
        df = self._load()
        if df.empty:
            return []
        return df.sort_values(by=["Date", "Time"], ascending=False).to_dict(orient="records")

    def get_stats(self) -> dict:
        df = self._load()
        today = datetime.now().strftime("%Y-%m-%d")
        today_df = df[df["Date"] == today] if not df.empty else df
        latest_row = None
        by_person = []

        if not df.empty:
            latest_row = df.sort_values(by=["Date", "Time"], ascending=False).iloc[0].to_dict()
            by_person = (
                df.groupby("Name")
                .size()
                .sort_values(ascending=False)
                .reset_index(name="days_present")
                .to_dict(orient="records")
            )

        return {
            "total_records": int(len(df)),
            "today_records": int(len(today_df)),
            "unique_people_today": int(today_df["Name"].nunique()) if not today_df.empty else 0,
            "last_check_in": latest_row,
            "by_person": by_person,
        }

    def process_prediction(self, prediction: dict) -> dict:
        date, time = now_strings()

        if prediction["status"] != "recognized":
            return {
                "status": prediction["status"],
                "name": prediction["name"],
                "best_guess": prediction["best_guess"],
                "confidence": prediction["confidence"],
                "message": prediction["message"],
                "attendance_marked": False,
            }

        df = self._load()
        name = prediction["name"]
        cooldown_minutes = self.config.get("CHECKIN_COOLDOWN_MINUTES", 120)

        today_rows = df[(df["Name"] == name) & (df["Date"] == date)]
        if not today_rows.empty:
            last_time_str = today_rows.sort_values("Time").iloc[-1]["Time"]
            last_dt = datetime.strptime(f"{date} {last_time_str}", "%Y-%m-%d %H:%M:%S")
            minutes_since = (datetime.now() - last_dt).total_seconds() / 60
            if minutes_since < cooldown_minutes:
                return {
                    "status": "already_marked",
                    "name": name,
                    "best_guess": name,
                    "confidence": prediction["confidence"],
                    "message": f"{name} is already marked present for today.",
                    "attendance_marked": False,
                }

        new_row = pd.DataFrame(
            [
                {
                    "Name": name,
                    "Date": date,
                    "Time": time,
                    "Confidence": format_confidence(prediction["confidence"]),
                }
            ]
        )

        updated = pd.concat([df, new_row], ignore_index=True)
        self._save(updated)

        return {
            "status": "marked_present",
            "name": name,
            "best_guess": name,
            "confidence": prediction["confidence"],
            "message": f"Attendance marked for {name}.",
            "attendance_marked": True,
        }

    def export_csv(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = Path(self.config["EXPORTS_DIR"]) / f"attendance_export_{timestamp}.csv"
        self._load().to_csv(export_path, index=False)
        return export_path

    def clear(self):
        self._save(pd.DataFrame(columns=self.COLUMNS))
