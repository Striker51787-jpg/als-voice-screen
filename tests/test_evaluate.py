"""Tests for evaluate.py's pure metric helpers on small, hand-verifiable
arrays -- doesn't touch the real dataset or retrain anything."""
import numpy as np

from evaluate import metrics_at, sensitivity_threshold


def test_metrics_at_perfect_separation():
    y_true = np.array([0, 0, 1, 1])
    prob = np.array([0.2, 0.4, 0.6, 0.8])
    result = metrics_at(y_true, prob, threshold=0.5)
    assert result["sensitivity"] == 1.0
    assert result["specificity"] == 1.0
    assert result["balanced_accuracy"] == 1.0
    assert result["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}


def test_metrics_at_all_predicted_positive():
    y_true = np.array([0, 0, 1, 1])
    prob = np.array([0.6, 0.7, 0.8, 0.9])
    result = metrics_at(y_true, prob, threshold=0.5)
    assert result["sensitivity"] == 1.0
    assert result["specificity"] == 0.0
    assert result["confusion_matrix"] == {"tn": 0, "fp": 2, "fn": 0, "tp": 2}


def test_sensitivity_threshold_meets_target_on_separable_data():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    prob = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    threshold = sensitivity_threshold(y_true, prob, target_sensitivity=1.0)
    achieved = metrics_at(y_true, prob, threshold)
    assert achieved["sensitivity"] >= 1.0


def test_sensitivity_threshold_lower_target_allows_higher_threshold():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    prob = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    loose = sensitivity_threshold(y_true, prob, target_sensitivity=1.0)
    strict = sensitivity_threshold(y_true, prob, target_sensitivity=0.5)
    # Asking for less sensitivity should never require a lower (more
    # permissive) threshold than asking for more sensitivity.
    assert strict >= loose
