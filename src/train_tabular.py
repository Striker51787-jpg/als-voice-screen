"""Train an ALS classifier on the real VOC-ALS clinical spreadsheet
(data/VOC-ALS.xlsx) using only the precomputed acoustic features -- no audio
download required for this pass.

IMPORTANT: ALSFRS-R subscores, FVC%, OnsetRegion, DiseaseDuration, etc. are
recorded as "-" placeholders for every healthy-control row in this dataset.
Including them as model inputs would be label leakage (their mere presence
reveals the diagnosis). Only acoustic features (F0/jitter/shimmer/HNR per
task) and basic demographics (Age, Sex) are used here.
"""
import os
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "VOC-ALS.xlsx")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "als_tabular_detector.joblib")

ACOUSTIC_PREFIXES = ("meanF0Hz_", "stdevF0Hz_", "HNR_", "localJitter_", "localShimmer_")

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


def load_features():
    df = pd.read_excel(DATA_PATH, sheet_name="VOC-ALS_Data", header=1)
    y = (df["Category"] == "ALS").astype(int)

    acoustic_cols = [c for c in df.columns if c.startswith(ACOUSTIC_PREFIXES)]
    X = df[acoustic_cols].apply(pd.to_numeric, errors="coerce").copy()
    X["Age"] = pd.to_numeric(df["Age (years)"], errors="coerce")
    X["Sex_M"] = (df["Sex"] == "M").astype(int)

    return X, y


def main():
    X, y = load_features()
    print(f"Loaded {len(X)} participants ({y.sum()} ALS, {(1 - y).sum()} HC), {X.shape[1]} features.")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    for name, estimator in CANDIDATES.items():
        pipeline = build_pipeline(estimator)
        probs = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
        preds = (probs >= 0.5).astype(int)
        auc = roc_auc_score(y, probs)
        results[name] = {"probs": probs, "preds": preds, "auc": auc}
        print(f"\n=== {name} (5-fold CV) ===")
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
