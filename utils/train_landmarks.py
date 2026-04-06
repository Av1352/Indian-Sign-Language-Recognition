"""
train_landmarks.py — Train an MLPClassifier on hand landmark features.

Run from the project root:
    python utils/train_landmarks.py

Reads:  data/landmarks.csv  (produced by utils/extract_landmarks.py)
Saves:  NEW_Models/landmark_model.pkl
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CSV_PATH   = "data/landmarks.csv"
MODEL_OUT  = "NEW_Models/landmark_model.pkl"
VAL_SPLIT  = 0.2
SEED       = 42

MLP_PARAMS = dict(
    hidden_layer_sizes = (256, 128),
    activation         = "relu",
    solver             = "adam",
    max_iter           = 500,
    early_stopping     = True,
    validation_fraction= 0.1,       # internal val used by early_stopping
    n_iter_no_change   = 15,
    random_state       = SEED,
    verbose            = True,
)
# ---------------------------------------------------------------------------


def main():
    # ------------------------------------------------------------------
    # 1. Load CSV
    # ------------------------------------------------------------------
    print(f"Loading {CSV_PATH} …")
    df = pd.read_csv(CSV_PATH)

    feature_cols = [c for c in df.columns if c not in ("filepath", "label")]
    if len(feature_cols) != 42:
        print(f"  Warning: expected 42 feature columns, found {len(feature_cols)}")

    X = df[feature_cols].values.astype(np.float32)
    y_raw = df["label"].values

    le = LabelEncoder()
    y  = le.fit_transform(y_raw)

    print(f"  Samples  : {len(X)}")
    print(f"  Features : {X.shape[1]}")
    print(f"  Classes  : {len(le.classes_)}  → {list(le.classes_)}")

    # ------------------------------------------------------------------
    # 2. Train / val split  (stratified to keep class balance)
    # ------------------------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size    = VAL_SPLIT,
        stratify     = y,
        random_state = SEED,
    )
    print(f"\n  Train: {len(X_train)}  |  Val: {len(X_val)}")

    # ------------------------------------------------------------------
    # 3. Train
    # ------------------------------------------------------------------
    print("\nTraining MLPClassifier …")
    clf = MLPClassifier(**MLP_PARAMS)
    clf.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 4. Evaluate
    # ------------------------------------------------------------------
    y_pred    = clf.predict(X_val)
    val_acc   = accuracy_score(y_val, y_pred)
    print(f"\nValidation accuracy: {val_acc * 100:.2f}%")

    target_names = [str(c) for c in le.classes_]
    print("\nClassification report:")
    print(classification_report(y_val, y_pred, target_names=target_names))

    # ------------------------------------------------------------------
    # 5. Save model + label encoder together
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    payload = {"model": clf, "label_encoder": le}
    joblib.dump(payload, MODEL_OUT)
    size_kb = os.path.getsize(MODEL_OUT) / 1024
    print(f"Model saved → {MODEL_OUT}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
