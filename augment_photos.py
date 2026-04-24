"""
Balanced augmentation — makes every person have the same number of training images.
Deletes old augmented files first, then generates fresh ones.
Run once, then run retrain.py.
"""

import random
import cv2
import numpy as np
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SKIP_DIRS = {
    "smart_attendance", "models", "data", "static", "templates",
    "tests", ".git", "__pycache__", ".venv", "venv", "node_modules",
}

# ── Find all person folders ───────────────────────────────────────────────────

def find_all_persons() -> dict[str, Path]:
    base = Path(__file__).parent
    persons = {}
    for entry in sorted(base.iterdir()):
        if not entry.is_dir() or entry.name.startswith(".") or entry.name in SKIP_DIRS:
            continue
        subfolders = sorted([p for p in entry.iterdir() if p.is_dir()])
        folder = subfolders[0] if subfolders else entry
        originals = [f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS and not f.stem.startswith("aug_")]
        if originals:
            persons[entry.name] = folder
    return persons

def get_originals(folder: Path) -> list[Path]:
    return [f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS and not f.stem.startswith("aug_")]

# ── Random augmentation ───────────────────────────────────────────────────────

def random_augment(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    result = image.copy()

    # Random horizontal flip
    if random.random() > 0.5:
        result = cv2.flip(result, 1)

    # Random brightness (-50 to +50)
    beta = random.randint(-50, 50)
    result = cv2.convertScaleAbs(result, alpha=1.0, beta=beta)

    # Random contrast (0.75 to 1.35)
    alpha = random.uniform(0.75, 1.35)
    result = cv2.convertScaleAbs(result, alpha=alpha, beta=0)

    # Random rotation (-15° to +15°)
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    result = cv2.warpAffine(result, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Random zoom (0% to 12% crop)
    if random.random() > 0.5:
        margin = random.randint(0, int(min(h, w) * 0.12))
        if margin > 0:
            cropped = result[margin:h - margin, margin:w - margin]
            result = cv2.resize(cropped, (w, h))

    # Random slight blur
    if random.random() > 0.6:
        result = cv2.GaussianBlur(result, (3, 3), 0)

    return result

# ── Main ──────────────────────────────────────────────────────────────────────

persons = find_all_persons()
print(f"Found {len(persons)} people: {', '.join(persons)}\n")

# Count originals per person
originals_map = {name: get_originals(folder) for name, folder in persons.items()}
counts = {name: len(imgs) for name, imgs in originals_map.items()}

print("Original photo counts:")
for name, count in counts.items():
    print(f"  {name}: {count}")

# Fixed target per person
TARGET = 50
print(f"\nTarget per person: {TARGET} images\n")

# Delete old augmented files and generate fresh balanced set
for name, folder in persons.items():
    originals = originals_map[name]

    # Remove old aug_ files
    old_augs = [f for f in folder.iterdir() if f.stem.startswith("aug_")]
    for f in old_augs:
        f.unlink()

    needed = TARGET - len(originals)
    print(f"{name}: {len(originals)} originals → generating {needed} augmented images...")

    for i in range(needed):
        source = random.choice(originals)
        image = cv2.imread(str(source))
        if image is None:
            continue
        augmented = random_augment(image)
        out_path = folder / f"aug_{i:05d}.jpg"
        cv2.imwrite(str(out_path), augmented)

    total = len(get_originals(folder)) + needed
    print(f"  → {total} total images\n")

print(f"Done! Everyone now has {TARGET} images.")
print("\nNow run:  python retrain.py")
