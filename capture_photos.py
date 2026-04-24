"""
Quick photo capture tool.
- Press SPACE to save a photo
- Press Q to quit
"""

import os
import time
import cv2
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

PERSON_NAME = "Aya"
SAVE_DIR = Path(__file__).parent / PERSON_NAME / "photos"

# ─────────────────────────────────────────────────────────────────────────────

SAVE_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open camera.")
    exit(1)

count = len(list(SAVE_DIR.glob("*.jpg")))
print(f"Saving photos to: {SAVE_DIR}")
print(f"Already have {count} photos.")
print("Press SPACE to capture | Press Q to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    cv2.putText(display, f"Saved: {count}  |  SPACE=capture  Q=quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 100), 2)
    cv2.imshow("Capture Photos", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord(" "):
        count += 1
        filename = SAVE_DIR / f"capture_{int(time.time())}_{count}.jpg"
        cv2.imwrite(str(filename), frame)
        print(f"  Saved {filename.name}")

cap.release()
cv2.destroyAllWindows()
print(f"\nDone! {count} photos saved to {SAVE_DIR}")
print("Now run:  python retrain.py")
