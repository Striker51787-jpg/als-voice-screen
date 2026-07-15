"""Train an ALS classifier on data/features_multitask.csv (one row per
participant, features from all 8 VOC-ALS speech tasks concatenated).
"""
import os
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, balanced_accuracy_score

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "features_multitask.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "als_multitask_detector.joblib")

CANDIDATES = {
    "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
}


def build_pipeline(estimator, k_best=40):
    base = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=k_best)),
        ("clf", estimator),
    ])
    return CalibratedClassifierCV(base, method="isotonic", cv=3)


def main():
    df = pd.read_csv(DATA_PATH)
    y = df["label"].to_numpy()
    X = df.drop(columns=["label", "participant_id"])
    print(f"Loaded {len(X)} participants, {X.shape[1]} raw features (312 expected from 8 tasks x 39 features).")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    for name, estimator in CANDIDATES.items():
        pipeline = build_pipeline(estimator)
        probs = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
        preds = (probs >= 0.5).astype(int)
        auc = roc_auc_score(y, probs)
        bal_acc = balanced_accuracy_score(y, preds)
        results[name] = {"auc": auc, "bal_acc": bal_acc}
        print(f"\n=== {name} (5-fold CV, top-40 features) ===")
        print(classification_report(y, preds, zero_division=0))
        print("ROC AUC:", round(auc, 3), "| Balanced accuracy:", round(bal_acc, 3))
        print("Confusion matrix:")
        print(confusion_matrix(y, preds))

    # Select by balanced accuracy, not just AUC -- a model that collapses to
    # predicting one class can still post a deceptively OK-looking AUC.
    best_name = max(results, key=lambda k: results[k]["bal_acc"])
    print(f"\nSelected best model: {best_name} (balanced acc {results[best_name]['bal_acc']:.3f}, AUC {results[best_name]['auc']:.3f})")

    best_pipeline = build_pipeline(CANDIDATES[best_name])
    best_pipeline.fit(X, y)
    joblib.dump(
        {"pipeline": best_pipeline, "feature_names": list(X.columns), "model_name": best_name},
        MODEL_PATH,
    )
    print(f"Saved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
