"""
retrain.py — Fine-tune the existing model on the augmented training set.

Run from the project root AFTER running utils/augment.py:
    python utils/retrain.py

Saves:
    Models/best_model.keras          — best checkpoint (overwrites)
    Models/best_model_inference.keras — same weights, augmentation layer stripped
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ---------------------------------------------------------------------------
# GPU setup
# ---------------------------------------------------------------------------
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f'Using GPU: {physical_devices[0]}')
    _BATCH_SIZE = 64
else:
    print('No GPU found, using CPU')
    _BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR     = "data/train"
MODEL_IN     = "NEW_Models/best_model.keras"
MODEL_OUT    = "NEW_Models/best_model.keras"
INFER_OUT    = "NEW_Models/best_model_inference.keras"
IMG_SIZE     = (100, 100)
BATCH_SIZE   = _BATCH_SIZE
SEED         = 1337
EPOCHS       = 30
LR           = 1e-4          # lower than original 1e-3 — fine-tuning regime
WEIGHT_DECAY = 1e-4
# ---------------------------------------------------------------------------


def load_data_to_numpy(data_dir, img_size=(100, 100), val_split=0.2, seed=1337):
    """
    Read all images from data_dir/{class}/ into numpy arrays using cv2.
    Returns X_train, y_train, X_val, y_val (float32), class_names.
    Images are stored as float32 [0-255] — the model's Rescaling(1./255)
    layer handles normalisation internally.
    """
    class_dirs  = sorted(d for d in Path(data_dir).iterdir() if d.is_dir())
    class_names = [d.name for d in class_dirs]
    n_classes   = len(class_names)

    X_list, y_list = [], []
    for class_idx, class_dir in enumerate(class_dirs):
        image_files = sorted(
            f for f in class_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".png"}
        )
        print(f"  Loading [{class_dir.name}] — {len(image_files)} images …")
        for img_path in image_files:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != img_size:
                img = cv2.resize(img, img_size, interpolation=cv2.INTER_LINEAR)
            X_list.append(img)
            y_list.append(class_idx)

    # Stack into arrays: (N, H, W, 1) float32
    X = np.array(X_list, dtype=np.float32)[..., np.newaxis]
    y = tf.keras.utils.to_categorical(y_list, num_classes=n_classes).astype(np.float32)

    print(f"\n  Total images: {len(X)}  |  shape: {X.shape}")
    print(f"  RAM for X: {X.nbytes / 1e9:.2f} GB")

    # Shuffle then split (reproducible)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    split      = int(len(X) * (1 - val_split))
    X_train, y_train = X[:split], y[:split]
    X_val,   y_val   = X[split:], y[split:]

    print(f"  Train: {len(X_train)}  |  Val: {len(X_val)}")
    return X_train, y_train, X_val, y_val, class_names


def export_inference_model(model: tf.keras.Model, save_path: str) -> None:
    """
    Save a copy of the model with the DataAugmentation layer (index 1) removed.
    Layer 0 is InputLayer; layer 1 is the data_augmentation Sequential block.
    """
    inp = layers.Input(shape=(100, 100, 1))
    x   = inp
    for layer in model.layers[2:]:   # skip InputLayer + data_augmentation
        x = layer(x)
    inf_model = models.Model(inp, x, name="inference_model")
    inf_model.save(save_path)
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"Inference model saved → {save_path}  ({size_mb:.1f} MB)")


def build_optimizer() -> tf.keras.optimizers.Optimizer:
    # tf.keras.optimizers.AdamW is built-in from TF 2.11+
    # fall back to Adam if the attribute is absent (older TF / tfa not installed)
    try:
        return tf.keras.optimizers.AdamW(learning_rate=LR, weight_decay=WEIGHT_DECAY)
    except AttributeError:
        print("AdamW not available in this TF version — falling back to Adam.")
        return tf.keras.optimizers.Adam(learning_rate=LR)


class EpochTimer(tf.keras.callbacks.Callback):
    """Prints epoch duration, current LR, and estimated time remaining."""

    def on_train_begin(self, logs=None):
        self._train_start = time.time()
        self._epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self._epoch_start
        self._epoch_times.append(elapsed)

        avg      = sum(self._epoch_times) / len(self._epoch_times)
        remaining = avg * (self.params["epochs"] - (epoch + 1))

        try:
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        except Exception:
            lr = float("nan")

        print(
            f"  ⏱  epoch {epoch + 1}/{self.params['epochs']} — "
            f"{elapsed:.1f}s  |  LR: {lr:.2e}  |  "
            f"ETA: {remaining / 60:.1f} min remaining"
        )


def main() -> None:
    print(f"TensorFlow {tf.__version__}")

    # ------------------------------------------------------------------
    # 1. Load datasets into memory (no disk I/O during training)
    # ------------------------------------------------------------------
    print("\nLoading all images into numpy arrays …")
    X_train, y_train, X_val, y_val, class_names = load_data_to_numpy(
        DATA_DIR, img_size=IMG_SIZE, val_split=0.2, seed=SEED
    )
    print(f"  Classes ({len(class_names)}): {class_names}")

    # Pin the full dataset tensors to CPU RAM so they never occupy GPU VRAM.
    # TF automatically transfers each batch to GPU during model.fit.
    AUTOTUNE = tf.data.AUTOTUNE
    with tf.device('/CPU:0'):
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        val_ds   = tf.data.Dataset.from_tensor_slices((X_val,   y_val))

    train_ds = (
        train_ds
        .shuffle(10000, seed=SEED, reshuffle_each_iteration=True)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        val_ds
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    # ------------------------------------------------------------------
    # 2. Load existing model
    # ------------------------------------------------------------------
    print(f"\nLoading model from {MODEL_IN} …")
    try:
        import tensorflow_addons as tfa
        model = tf.keras.models.load_model(
            MODEL_IN,
            custom_objects={"AdamW": tfa.optimizers.AdamW},
        )
    except Exception:
        model = tf.keras.models.load_model(MODEL_IN, compile=False)

    model.compile(
        optimizer = build_optimizer(),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"],
    )

    print(f"  Input  shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    print(f"  Trainable params: {model.count_params():,}")

    # ------------------------------------------------------------------
    # 3. Evaluate baseline before fine-tuning
    # ------------------------------------------------------------------
    print("\nBaseline evaluation (before fine-tuning) …")
    base_loss, base_acc = model.evaluate(val_ds, verbose=1)
    print(f"  Baseline val accuracy: {base_acc * 100:.2f}%")

    # ------------------------------------------------------------------
    # 4. Fine-tune
    # ------------------------------------------------------------------
    callbacks = [
        EpochTimer(),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_OUT,
            monitor        = "val_accuracy",
            save_best_only = True,
            verbose        = 1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor              = "val_accuracy",
            patience             = 5,
            restore_best_weights = True,
            verbose              = 1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = 0.5,
            patience = 3,
            min_lr   = 1e-7,
            verbose  = 1,
        ),
    ]

    print(f"\nFine-tuning (LR={LR}, up to {EPOCHS} epochs, 500 steps/epoch) …")
    history = model.fit(
        train_ds,
        validation_data  = val_ds,
        epochs           = EPOCHS,
        steps_per_epoch  = 500,
        verbose          = 1,
        callbacks        = callbacks,
    )

    best_val_acc = max(history.history["val_accuracy"])
    print(f"\nBest val accuracy reached: {best_val_acc * 100:.2f}%")

    # ------------------------------------------------------------------
    # 5. Export inference model (no augmentation layer)
    # ------------------------------------------------------------------
    print("\nExporting inference model …")
    best_model = tf.keras.models.load_model(MODEL_OUT, compile=False)
    export_inference_model(best_model, INFER_OUT)

    print("\nAll done.")


if __name__ == "__main__":
    main()
