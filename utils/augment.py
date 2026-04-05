"""
augment.py — Offline data augmentation for Canny-edge training images.

Run from the project root:
    python utils/augment.py

For each original image in data/train/{class}/ the script writes
AUGMENTS_PER_IMAGE new files named aug_{i}_{stem}.png into the same
folder.  Re-running is safe: files whose stem already starts with "aug_"
are skipped so originals are never re-augmented.
"""

import os
import random
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRAIN_DIR         = "data/train"
AUGMENTS_PER_IMAGE = 5
ROTATION_RANGE    = 15      # degrees ±
ZOOM_RANGE        = 0.10    # fraction ±
BRIGHTNESS_RANGE  = 40      # additive shift ±
NOISE_SIGMA       = 8       # Gaussian std-dev
SEED              = 42
# ---------------------------------------------------------------------------


def random_brightness(img: np.ndarray) -> np.ndarray:
    shift = random.uniform(-BRIGHTNESS_RANGE, BRIGHTNESS_RANGE)
    return np.clip(img.astype(np.float32) + shift, 0, 255).astype(np.uint8)


def random_rotation(img: np.ndarray) -> np.ndarray:
    angle = random.uniform(-ROTATION_RANGE, ROTATION_RANGE)
    h, w  = img.shape[:2]
    M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def random_flip(img: np.ndarray) -> np.ndarray:
    return cv2.flip(img, 1) if random.random() < 0.5 else img


def random_zoom(img: np.ndarray) -> np.ndarray:
    h, w   = img.shape[:2]
    factor = random.uniform(1.0 - ZOOM_RANGE, 1.0 + ZOOM_RANGE)
    nh, nw = max(1, int(h * factor)), max(1, int(w * factor))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    if factor >= 1.0:
        # Zoom in — crop the centre back to (h, w)
        y0 = (nh - h) // 2
        x0 = (nw - w) // 2
        return resized[y0:y0 + h, x0:x0 + w]
    else:
        # Zoom out — place on a black canvas of (h, w)
        canvas = np.zeros((h, w), dtype=np.uint8)
        y0 = (h - nh) // 2
        x0 = (w - nw) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = resized
        return canvas


def add_gaussian_noise(img: np.ndarray) -> np.ndarray:
    noise = np.random.normal(0, NOISE_SIGMA, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def augment(img: np.ndarray) -> np.ndarray:
    """Apply the full augmentation pipeline to a single grayscale image."""
    img = random_brightness(img)
    img = random_rotation(img)
    img = random_flip(img)
    img = random_zoom(img)
    img = add_gaussian_noise(img)
    return img


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    train_path  = Path(TRAIN_DIR)
    class_dirs  = sorted(d for d in train_path.iterdir() if d.is_dir())

    if not class_dirs:
        raise RuntimeError(f"No class sub-directories found under {TRAIN_DIR!r}.")

    total_written = 0

    for class_dir in class_dirs:
        # Collect originals only — skip any file already named aug_*
        originals = sorted(
            f for f in class_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".png"}
            and not f.stem.startswith("aug_")
        )

        if not originals:
            print(f"  [{class_dir.name}] no original images found, skipping.")
            continue

        written = 0
        for img_path in originals:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  Warning: could not read {img_path}, skipping.")
                continue

            for i in range(AUGMENTS_PER_IMAGE):
                aug_img  = augment(img)
                out_name = f"aug_{i}_{img_path.stem}.png"
                cv2.imwrite(str(class_dir / out_name), aug_img)
                written += 1

        total_written += written
        print(
            f"  [{class_dir.name}] {len(originals)} originals "
            f"→ {written} augmented images written"
        )

    print(f"\nDone. Total new images written: {total_written}")
    print(
        f"Expanded dataset size: "
        f"{total_written + sum(len(list((Path(TRAIN_DIR) / d.name).glob('*.jpg')) + list((Path(TRAIN_DIR) / d.name).glob('*.png'))) for d in class_dirs if d.is_dir())} images"
    )


if __name__ == "__main__":
    main()
