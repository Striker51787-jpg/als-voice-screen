"""Bootstrap a 95% CI on the multi-task model's ROC-AUC.

Correct approach: compute out-of-fold CV predictions ONCE on the real,
non-duplicated data, then bootstrap-resample those (label, probability) pairs.
Resampling the raw rows with replacement BEFORE re-running cross-validation
(an earlier mistake here) lets duplicate copies of the same participant land
in both the train and test side of a fold, which leaks information and
inflates the score -- e.g. it produced a bogus AUC of 0.86 instead of the
correct ~0.70.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "features_multitask.csv")
N_BOOTSTRAP = 2000


def build_pipeline():
    base = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=40)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    return CalibratedClassifierCV(base, method="isotonic", cv=3)


def main():
    df = pd.read_csv(DATA_PATH)
    y = df["label"].to_numpy()
    X = df.drop(columns=["label", "participant_id"])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probs = cross_val_predict(build_pipeline(), X, y, cv=cv, method="predict_proba")[:, 1]
    point_auc = roc_auc_score(y, probs)
    print("Point estimate AUC:", round(point_auc, 3))

    rng = np.random.RandomState(42)
    n = len(y)
    boot_aucs = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.randint(0, n, size=n)
        yb, pb = y[idx], probs[idx]
        if len(np.unique(yb)) < 2:
            continue
        boot_aucs.append(roc_auc_score(yb, pb))

    boot_aucs = np.array(boot_aucs)
    lo, hi = np.percentile(boot_aucs, [2.5, 97.5])
    print(f"Bootstrap resamples: {len(boot_aucs)}")
    print(f"95% CI: [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
