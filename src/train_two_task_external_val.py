"""True external validation: train a phonationA+phonationI-only model on our
own VOC-ALS data, then score it on the independent Minsk2020 ALS database
(Belarus, 31 ALS + 33 HC, different recording equipment/population, never
seen during training) -- the genuine external-validation test noted as
missing elsewhere in this project (TORGO was ruled out separately -- see
README, "External validation").

Source: Vashkevich M., Rushkevich Yu. "Classification of ALS patients based
on acoustic analysis of sustained vowel phonations", Biomedical Signal
Processing and Control, 2021. https://doi.org/10.1016/j.bspc.2020.102350
https://github.com/Mak-Sim/Minsk2020_ALS_database (GPL-3.0)

This repo does NOT redistribute that dataset's audio. To run this script:
  1. Download data/ (ALS/ and HC/ folders of .wav files) from the GitHub repo
     above.
  2. Place them at data/external_minsk/als/ and data/external_minsk/hc/
     (gitignored -- never commit patient audio into this repo).
  3. Run: python train_two_task_external_val.py
"""
import os

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import extract_features

VOC_ALS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "features_multitask.csv")
MINSK_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "external_minsk")

CANDIDATES = {
    "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
}


def build_pipeline(estimator):
    base = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", estimator),
    ])
    return CalibratedClassifierCV(base, method="isotonic", cv=3)


def load_voc_als_two_task():
    """VOC-ALS features restricted to phonationA + phonationI columns only,
    matching what's available in the Minsk external set."""
    df = pd.read_csv(VOC_ALS_PATH)
    cols = [c for c in df.columns if c.startswith("phonationA__") or c.startswith("phonationI__")]
    X = df[cols]
    y = df["label"].to_numpy()
    return X, y, cols


def extract_minsk_features(cols):
    """Extract features from every Minsk .wav, in the same column format as
    VOC-ALS (task__feature), using our own features.py -- same extraction
    code as production, so this is a fair apples-to-apples comparison."""
    rows = []
    for label, subdir in [(1, "als"), (0, "hc")]:
        folder = os.path.join(MINSK_DIR, subdir)
        participants = {}
        for fname in sorted(os.listdir(folder)):
            if not fname.endswith(".wav"):
                continue
            participant_id, vowel = fname.replace(".wav", "").split("_")
            task = {"a": "phonationA", "i": "phonationI"}[vowel]
            path = os.path.join(folder, fname)
            try:
                feats = extract_features(path)
            except Exception as e:
                print(f"  Skipping {fname}: {e}")
                continue
            row = participants.setdefault(participant_id, {"label": label})
            for k, v in feats.items():
                row[f"{task}__{k}"] = v
        rows.extend(participants.values())

    df = pd.DataFrame(rows)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"  WARNING: {len(missing)} expected columns missing from Minsk features: {missing[:5]}...")
    X = df.reindex(columns=cols)
    y = df["label"].to_numpy()
    return X, y


def main():
    if not os.path.isdir(MINSK_DIR):
        print(f"External data not found at {MINSK_DIR}.")
        print("See this file's module docstring for download instructions.")
        return

    print("Loading VOC-ALS (phonationA + phonationI only) for training...")
    X_train, y_train, cols = load_voc_als_two_task()
    print(f"  {len(X_train)} participants ({int(y_train.sum())} ALS, {int((1 - y_train).sum())} HC), {len(cols)} features")

    print("\nCross-validated performance on VOC-ALS itself (sanity check, 2-task-only model)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, estimator in CANDIDATES.items():
        probs = cross_val_predict(build_pipeline(estimator), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        auc = roc_auc_score(y_train, probs)
        print(f"  {name}: AUC={auc:.3f}")

    print("\nExtracting features from Minsk2020 external audio (never seen during training)...")
    X_ext, y_ext = extract_minsk_features(cols)
    print(f"  {len(X_ext)} participants ({int(y_ext.sum())} ALS, {int((1 - y_ext).sum())} HC)")

    print("\nFitting each candidate on ALL of VOC-ALS (2-task), scoring on Minsk external set...")
    for name, estimator in CANDIDATES.items():
        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        probs = pipeline.predict_proba(X_ext)[:, 1]
        preds = (probs >= 0.5).astype(int)
        auc = roc_auc_score(y_ext, probs)
        bal_acc = balanced_accuracy_score(y_ext, preds)
        cm = confusion_matrix(y_ext, preds)
        print(f"\n=== {name} (trained on VOC-ALS, tested on Minsk) ===")
        print(f"  External ROC-AUC: {auc:.3f}  Balanced accuracy: {bal_acc:.3f}")
        print(f"  Confusion matrix:\n{cm}")

    print("\n\nFor reference, this project's other validated numbers:")
    print("  VOC-ALS single-task (phonationA only), 5-fold CV:        AUC ~0.62")
    print("  VOC-ALS 8-task multi-task, held-out:                     AUC ~0.76")


if __name__ == "__main__":
    main()
