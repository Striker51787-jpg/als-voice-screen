"""Generate every presentation-ready figure from the actual saved data and
evaluation results -- nothing here is hand-copied into slides, so figures
never drift from models/evaluation_report.json or the underlying CSVs.

Recomputes the single-task and tabular-baseline CV numbers fresh (rather than
hardcoding remembered values) so the comparison chart can't go stale. Reuses
evaluate.py's exact train/test split (same RANDOM_STATE) and selection logic
for the multi-task ROC curve, confusion matrices, and calibration curve, so
those figures are guaranteed consistent with models/evaluation_report.json.

Run with: python generate_figures.py (from the src/ directory, matching the
convention of every other script in this project).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import evaluate
import train as train_single
import train_tabular
import train_multitask

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORT_PATH = os.path.join(MODELS_DIR, "evaluation_report.json")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "presentation", "figures")

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "legend.fontsize": 10,
})


def _best_auc(candidates, build_pipeline_fn, X, y, cv, groups=None):
    """Best held-in-CV ROC-AUC across a script's candidate models -- mirrors
    how each train_*.py script picks its own winner."""
    best = -np.inf
    for _, estimator in candidates.items():
        pipeline = build_pipeline_fn(estimator)
        kwargs = {"groups": groups} if groups is not None else {}
        probs = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba", **kwargs)[:, 1]
        best = max(best, roc_auc_score(y, probs))
    return best


def compute_comparison_aucs():
    """Recompute (not hardcode) the single-task and tabular-baseline CV AUCs,
    so the comparison chart can never drift from what those scripts actually
    produce. Multi-task numbers come straight from evaluation_report.json."""
    # Single-task (phonationA only), participant-grouped CV.
    df = pd.read_csv(train_single.DATA_PATH)
    y = df["label"].to_numpy()
    groups = df["participant_id"].to_numpy()
    X = df.drop(columns=["label", "file", "participant_id"])
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    single_auc = _best_auc(train_single.CANDIDATES, train_single.build_pipeline, X, y, cv, groups=groups)

    # Tabular baseline (spreadsheet's precomputed acoustic features).
    X, y = train_tabular.load_features()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tabular_auc = _best_auc(train_tabular.CANDIDATES, train_tabular.build_pipeline, X, y, cv)

    with open(REPORT_PATH) as f:
        report = json.load(f)

    return {
        "single_task": single_auc,
        "tabular_baseline": tabular_auc,
        "multitask_nested_cv": report["nested_cv"]["roc_auc"],
        "multitask_heldout": report["heldout"]["roc_auc"],
    }


def fig_model_comparison(aucs):
    labels = ["Single task\n(phonationA only)", "Tabular baseline\n(spreadsheet features)",
              "Multi-task\n(nested CV)", "Multi-task\n(locked held-out)"]
    values = [aucs["single_task"], aucs["tabular_baseline"],
              aucs["multitask_nested_cv"], aucs["multitask_heldout"]]
    colors = ["#9ca3af", "#9ca3af", "#2563eb", "#1d4ed8"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(labels, values, color=colors, width=0.6)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Chance (0.5)")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.2f}",
                ha="center", fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Combining all 8 speech tasks beats single-task and\ntabular-only baselines")
    ax.legend(loc="upper left")
    fig.text(0.5, -0.02,
             "Single-task/tabular bars: plain 5-fold CV. Multi-task bars: nested CV\n"
             "(unbiased selection estimate) and a locked held-out test set, never used for model selection.",
             ha="center", fontsize=9, style="italic", color="#555555")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "model_comparison.png"), bbox_inches="tight")
    plt.close()


def fig_roc_curve(y_test, test_prob, heldout_auc, nested_auc):
    fpr, tpr, _ = roc_curve(y_test, test_prob)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#1d4ed8", linewidth=2.5,
            label=f"Held-out ROC (AUC = {heldout_auc:.2f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance (AUC = 0.50)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve: multi-task model\non locked held-out set")
    ax.legend(loc="lower right")
    ax.text(0.55, 0.15, f"Nested-CV estimate: {nested_auc:.2f}\n(unbiased selection estimate)",
            fontsize=9, style="italic", color="#555555",
            bbox=dict(boxstyle="round", facecolor="#f3f4f6", edgecolor="#d1d5db"))
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "roc_curve.png"))
    plt.close()


def fig_calibration(y_test, test_prob, model_name):
    frac_pos, mean_pred = calibration_curve(y_test, test_prob, n_bins=5, strategy="uniform")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", color="#1d4ed8", linewidth=2, markersize=8,
            label=f"{model_name} (held-out)")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction actually ALS")
    ax.set_title(f"Calibration (VOC-ALS held-out, n={len(y_test)})")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "calibration_curve.png"))
    plt.close()


def fig_confusion_matrices(report):
    ops = [
        ("Default threshold (0.5)", report["heldout"]["operating_point_default_0.5"]),
        ("Screening threshold\n(tuned for sensitivity)", report["heldout"]["operating_point_screening_0.85_sensitivity"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (title, op) in zip(axes, ops):
        cm = op["confusion_matrix"]
        matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        im = ax.imshow(matrix, cmap="Blues", vmin=0)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        fontsize=16, fontweight="bold",
                        color="white" if matrix[i, j] > matrix.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Control", "ALS"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Control", "ALS"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"{title}\nsens={op['sensitivity']:.2f}, spec={op['specificity']:.2f}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrices.png"))
    plt.close()


def fig_feature_importance():
    """Fit a fresh, uncalibrated logistic regression (same k=40 selection as
    production) purely for interpretability -- not used for scoring."""
    df = pd.read_csv(train_multitask.DATA_PATH)
    y = df["label"].to_numpy()
    X = df.drop(columns=["label", "participant_id"])

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=40)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    pipeline.fit(X, y)
    select = pipeline.named_steps["select"]
    clf = pipeline.named_steps["clf"]
    selected = X.columns[select.get_support()]
    coefs = clf.coef_[0]

    importance = pd.DataFrame({"feature": selected, "coef": coefs})
    importance["abs_coef"] = importance["coef"].abs()
    importance = importance.sort_values("abs_coef", ascending=False).head(15).iloc[::-1]

    colors = ["#dc2626" if c > 0 else "#2563eb" for c in importance["coef"]]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(importance["feature"], importance["coef"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Logistic regression coefficient\n(red = pushes toward ALS-consistent, blue = pushes toward control)")
    ax.set_title("Top 15 features driving the multi-task model")
    fig.text(0.5, -0.03,
             "Caveat: most top features are MFCC statistics (generic vocal timbre), not clinical\n"
             "markers like jitter/shimmer -- see README for the confound-fitting risk this implies.",
             ha="center", fontsize=9, style="italic", color="#555555")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "feature_importance.png"), bbox_inches="tight")
    plt.close()


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    with open(REPORT_PATH) as f:
        report = json.load(f)

    print("Recomputing single-task and tabular-baseline CV AUCs for the comparison chart...")
    aucs = compute_comparison_aucs()
    print(f"  single_task={aucs['single_task']:.3f}  tabular_baseline={aucs['tabular_baseline']:.3f}  "
          f"multitask_nested_cv={aucs['multitask_nested_cv']:.3f}  multitask_heldout={aucs['multitask_heldout']:.3f}")
    fig_model_comparison(aucs)

    print("Reproducing evaluate.py's exact held-out split for the ROC/calibration/confusion figures...")
    df = pd.read_csv(evaluate.DATA_PATH)
    y = df["label"].to_numpy()
    X = df.drop(columns=["label", "participant_id"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=evaluate.RANDOM_STATE
    )
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=evaluate.RANDOM_STATE)
    final_model, final_name = evaluate.select_and_fit(X_train, y_train, inner)
    test_prob = final_model.predict_proba(X_test)[:, 1]

    fig_roc_curve(y_test, test_prob, report["heldout"]["roc_auc"], report["nested_cv"]["roc_auc"])
    fig_calibration(y_test, test_prob, final_name)
    fig_confusion_matrices(report)

    print("Fitting an uncalibrated model for feature-importance interpretability...")
    fig_feature_importance()

    print(f"\nSaved 5 figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
