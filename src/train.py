"""Train a classifier on data/features.csv and report cross-validated metrics.

Uses participant-grouped CV (StratifiedGroupKFold) so that multiple recordings
from the same person never span both the train and test side of a fold --
random k-fold CV on a file-level voice dataset leaks identity and inflates
reported accuracy.
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "features.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "als_detector.joblib")

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
    # Wrap in calibration so predict_proba outputs are meaningful probabilities,
    # not just rank scores -- important since predict.py reports a "risk score".
    return CalibratedClassifierCV(base, method="isotonic", cv=3)


def main():
    df = pd.read_csv(DATA_PATH)
    y = df["label"].to_numpy()
    groups = df["participant_id"].to_numpy()
    X = df.drop(columns=["label", "file", "participant_id"])

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    for name, estimator in CANDIDATES.items():
        pipeline = build_pipeline(estimator)
        probs = cross_val_predict(pipeline, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
        preds = (probs >= 0.5).astype(int)
        auc = roc_auc_score(y, probs)
        results[name] = {"pipeline": pipeline, "probs": probs, "preds": preds, "auc": auc}
        print(f"\n=== {name} (grouped 5-fold CV) ===")
        print(classification_report(y, preds))
        print("ROC AUC:", round(auc, 3))
        print("Confusion matrix:")
        print(confusion_matrix(y, preds))

    best_name = max(results, key=lambda k: results[k]["auc"])
    print(f"\nSelected best model: {best_name} (ROC AUC {results[best_name]['auc']:.3f})")

    best_pipeline = build_pipeline(CANDIDATES[best_name])
    best_pipeline.fit(X, y)
    joblib.dump(
        {"pipeline": best_pipeline, "feature_names": list(X.columns), "model_name": best_name},
        MODEL_PATH,
    )
    print(f"Saved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
