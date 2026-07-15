"""Rigorous internal validation of the multi-task ALS model on VOC-ALS.

The headline 5-fold CV number in train_multitask.py is mildly optimistic: the
model type (and feature count k) were chosen on the same CV that reports the
score. This script removes that optimism two ways:

  1. Nested CV on a train split -- the inner loop re-runs the exact model
     selection procedure (pick the best of the 3 candidate models by balanced
     accuracy at k=40, matching how the production model was chosen), and the
     outer loop scores it on data the selection never saw. This gives an
     unbiased estimate of the *whole pipeline including selection*.
  2. A locked held-out test set (20%, never touched during selection) scored
     exactly once for a clean headline number, plus a calibration curve and a
     sensitivity-focused operating point.

Reports on the same pipeline shipped in models/als_multitask_detector.joblib;
it does not retrain or replace that model. Writes models/evaluation_report.json
and models/calibration_curve.png.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from train_multitask import CANDIDATES, build_pipeline

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "features_multitask.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORT_PATH = os.path.join(MODELS_DIR, "evaluation_report.json")
PLOT_PATH = os.path.join(MODELS_DIR, "calibration_curve.png")

K_BEST = 40
RANDOM_STATE = 42


def select_and_fit(X, y, inner_cv):
    """Reproduce the production selection: pick the candidate model with the
    best balanced accuracy under inner CV (k fixed at 40), then fit it on all
    of (X, y). Returns (fitted_pipeline, best_name)."""
    from sklearn.model_selection import cross_val_predict

    best_name, best_score = None, -np.inf
    for name, estimator in CANDIDATES.items():
        probs = cross_val_predict(
            build_pipeline(estimator, k_best=K_BEST), X, y,
            cv=inner_cv, method="predict_proba",
        )[:, 1]
        score = balanced_accuracy_score(y, (probs >= 0.5).astype(int))
        if score > best_score:
            best_name, best_score = name, score

    fitted = build_pipeline(CANDIDATES[best_name], k_best=K_BEST)
    fitted.fit(X, y)
    return fitted, best_name


def nested_cv(X, y):
    """Unbiased estimate of the selection procedure via nested CV."""
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_prob = np.zeros(len(y))
    picks = []
    for train_idx, test_idx in outer.split(X, y):
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        model, name = select_and_fit(X.iloc[train_idx], y[train_idx], inner)
        oof_prob[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
        picks.append(name)
    preds = (oof_prob >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, oof_prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
        "selected_models_per_fold": picks,
    }


def sensitivity_threshold(y_true, prob, target_sensitivity=0.85):
    """Lowest threshold whose sensitivity (recall on the positive/ALS class)
    is at least target_sensitivity, derived from the ROC on the TRAIN split."""
    fpr, tpr, thresholds = roc_curve(y_true, prob)
    ok = np.where(tpr >= target_sensitivity)[0]
    # roc_curve thresholds are descending; the last qualifying index is the
    # highest-specificity threshold that still meets the sensitivity target.
    return float(thresholds[ok[-1]]) if len(ok) else 0.5


def metrics_at(y_true, prob, threshold):
    preds = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "threshold": round(float(threshold), 3),
        "sensitivity": round(float(sens), 3),
        "specificity": round(float(spec), 3),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, preds)), 3),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main():
    df = pd.read_csv(DATA_PATH)
    y = df["label"].to_numpy()
    X = df.drop(columns=["label", "participant_id"])
    print(f"Loaded {len(X)} participants ({int(y.sum())} ALS, {int((1 - y).sum())} HC).")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    print(f"Train: {len(X_train)}  Locked test: {len(X_test)}")

    print("\nRunning nested CV (unbiased estimate of model selection)...")
    nested = nested_cv(X_train, y_train)
    print(f"  Nested-CV ROC-AUC: {nested['roc_auc']:.3f} | "
          f"balanced acc: {nested['balanced_accuracy']:.3f}")
    print(f"  Models picked per outer fold: {nested['selected_models_per_fold']}")

    print("\nFitting final model on train split, scoring locked test set once...")
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    final_model, final_name = select_and_fit(X_train, y_train, inner)
    test_prob = final_model.predict_proba(X_test)[:, 1]
    heldout_auc = float(roc_auc_score(y_test, test_prob))
    print(f"  Selected model: {final_name}")
    print(f"  Held-out ROC-AUC: {heldout_auc:.3f}")

    # Operating points: default 0.5 vs a sensitivity-focused threshold picked on
    # the TRAIN split (never on the test set), then reported on the test set.
    train_prob = final_model.predict_proba(X_train)[:, 1]
    sens_thr = sensitivity_threshold(y_train, train_prob, target_sensitivity=0.85)
    op_default = metrics_at(y_test, test_prob, 0.5)
    op_sensitive = metrics_at(y_test, test_prob, sens_thr)
    print(f"\n  Held-out @ 0.50 cutoff:      "
          f"sens={op_default['sensitivity']}, spec={op_default['specificity']}")
    print(f"  Held-out @ {sens_thr:.2f} (screening): "
          f"sens={op_sensitive['sensitivity']}, spec={op_sensitive['specificity']}")

    # Calibration curve on held-out probabilities.
    n_bins = 5
    frac_pos, mean_pred = calibration_curve(y_test, test_prob, n_bins=n_bins, strategy="uniform")
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
    plt.plot(mean_pred, frac_pos, "o-", label=f"{final_name} (held-out)")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction ALS")
    plt.title(f"Calibration (VOC-ALS held-out, n={len(y_test)})")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=120)
    plt.close()
    print(f"\nSaved calibration curve to {PLOT_PATH}")

    report = {
        "dataset": "VOC-ALS multi-task features",
        "n_total": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "k_best": K_BEST,
        "nested_cv": nested,
        "heldout": {
            "selected_model": final_name,
            "roc_auc": round(heldout_auc, 3),
            "operating_point_default_0.5": op_default,
            "operating_point_screening_0.85_sensitivity": op_sensitive,
        },
        "notes": (
            "Nested CV estimates the full selection procedure; the held-out set is "
            "scored exactly once. Small n (test set ~31) means all figures carry wide "
            "uncertainty -- see bootstrap_auc.py for the CV-level CI. The screening "
            "threshold is chosen on the train split only; the app still uses 0.5."
        ),
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved evaluation report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
