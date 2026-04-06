"""
extract_landmarks.py — Extract MediaPipe hand landmarks from training images.

Run from the project root:
    python utils/extract_landmarks.py

Output:
    data/landmarks.csv
    Columns: filepath, label, x0, y0, x1, y1, ... x20, y20
    (coordinates normalised to the hand bounding box, range [0, 1])

Images where MediaPipe detects no hand are skipped; a per-class skip count
is printed at the end.
"""

import csv
import os
from pathlib import Path

import cv2
import mediapipe as mp

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRAIN_DIR   = "images"
OUTPUT_CSV  = "data/landmarks.csv"
N_LANDMARKS = 21
IMG_EXTS    = {".jpg", ".jpeg", ".png"}
# ---------------------------------------------------------------------------

# Column header: filepath, label, x0, y0, x1, y1, ..., x20, y20
HEADER = ["filepath", "label"] + [
    coord
    for i in range(N_LANDMARKS)
    for coord in (f"x{i}", f"y{i}")
]


def normalize_to_bbox(landmarks):
    """
    Re-normalise 21 landmark (x, y) pairs from image-relative [0,1] coordinates
    to hand-bounding-box-relative [0,1] coordinates.

    landmarks: list of (x, y) tuples, already in image-normalised space.
    Returns:   flat list [x0, y0, x1, y1, ..., x20, y20] in bbox space.
    """
    xs = [p[0] for p in landmarks]
    ys = [p[1] for p in landmarks]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Guard against degenerate (zero-size) bounding box
    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0

    flat = []
    for x, y in landmarks:
        flat.append((x - x_min) / x_range)
        flat.append((y - y_min) / y_range)
    return flat


def main():
    train_path  = Path(TRAIN_DIR)
    class_dirs  = sorted(d for d in train_path.iterdir() if d.is_dir())

    if not class_dirs:
        raise RuntimeError(f"No class sub-directories found under {TRAIN_DIR!r}.")

    os.makedirs(Path(OUTPUT_CSV).parent, exist_ok=True)

    mp_hands    = mp.solutions.hands
    hands_model = mp_hands.Hands(
        static_image_mode       = True,
        max_num_hands           = 1,
        min_detection_confidence= 0.3,   # lower threshold to catch harder images
    )

    total_written = 0
    total_skipped = 0
    skip_report   = {}   # class_name -> skip count

    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(HEADER)

        for class_dir in class_dirs:
            label = class_dir.name
            image_files = sorted(
                f for f in class_dir.iterdir()
                if f.suffix.lower() in IMG_EXTS
            )

            skipped  = 0
            written  = 0

            for img_path in image_files:
                # MediaPipe expects RGB
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    skipped += 1
                    continue

                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                result  = hands_model.process(img_rgb)

                if not result.multi_hand_landmarks:
                    skipped += 1
                    continue

                # Use first detected hand only
                hand_lms  = result.multi_hand_landmarks[0]
                landmarks = [(lm.x, lm.y) for lm in hand_lms.landmark]
                flat_coords = normalize_to_bbox(landmarks)

                row = [str(img_path), label] + flat_coords
                writer.writerow(row)
                written += 1

            total_written += written
            total_skipped += skipped
            skip_report[label] = skipped

            print(
                f"  [{label}]  written: {written:4d}  "
                f"skipped: {skipped:4d} / {len(image_files)}"
            )

    hands_model.close()

    print(f"\nCSV saved → {OUTPUT_CSV}")
    print(f"Total rows written : {total_written}")
    print(f"Total images skipped: {total_skipped}")

    # Classes with high skip rates (>20%) flagged for attention
    flagged = {
        k: v for k, v in skip_report.items()
        if v > 0
    }
    if flagged:
        print("\nSkip counts per class:")
        for label, count in sorted(flagged.items(), key=lambda x: -x[1]):
            print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
