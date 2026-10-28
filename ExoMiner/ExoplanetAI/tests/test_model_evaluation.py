"""
test_model_evaluation.py - Regression Test Suite for Model Lab Evaluation and Metrics.
Verifies that:
- Confusion matrices are derived from actual labels and predictions (no hardcoded matrices).
- Metric values change when predictions change.
- ROC and PR curve data are based on actual prediction scores (no synthetic polynomials).
- Missing or incompatible datasets do not produce fabricated metrics.
- A dataset containing only one class is handled correctly without crashing.
- Undefined metrics and zero-division cases are reported honestly.
- Static demonstration values cannot be mistaken for measured model performance.
- Model metadata contains no unsupported 100% claims.
"""

import os
import json
import numpy as np
import pandas as pd
import pytest

from backend.metrics import (
    calculate_classification_metrics,
    generate_confusion_matrix_data,
    calculate_roc_curve_data,
    calculate_pr_curve_data,
    load_benchmark_evaluation_artifact,
    DEFAULT_BENCHMARK_ARTIFACT_PATH
)


def test_confusion_matrix_derived_from_actual_labels_and_predictions():
    """Verify confusion matrix counts are strictly derived from input vectors, not hardcoded."""
    y_true = [0, 0, 0, 1, 1]
    y_pred = [0, 0, 1, 0, 1]
    # TN: 2, FP: 1, FN: 1, TP: 1
    cm_data = generate_confusion_matrix_data(y_true, y_pred)
    assert cm_data["true_negatives"] == 2
    assert cm_data["false_positives"] == 1
    assert cm_data["false_negatives"] == 1
    assert cm_data["true_positives"] == 1
    assert cm_data["matrix"] == [[2, 1], [1, 1]]

    # Another vector to verify it is not static
    y_true_2 = [1, 1, 1, 1]
    y_pred_2 = [1, 1, 1, 1]
    cm_data_2 = generate_confusion_matrix_data(y_true_2, y_pred_2)
    assert cm_data_2["true_positives"] == 4
    assert cm_data_2["true_negatives"] == 0
    assert cm_data_2["matrix"] == [[0, 0], [0, 4]]


def test_metric_values_change_when_predictions_change():
    """Verify classification metrics change dynamically with prediction vector modifications."""
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    # Perfect predictions
    preds_perfect = [0, 0, 0, 0, 1, 1, 1, 1]
    m_perfect = calculate_classification_metrics(y_true, preds_perfect)
    assert m_perfect["accuracy"] == 1.0
    assert m_perfect["recall"] == 1.0
    assert m_perfect["precision"] == 1.0
    assert m_perfect["f1_score"] == 1.0

    # Inverted predictions
    preds_inverted = [1, 1, 1, 1, 0, 0, 0, 0]
    m_inverted = calculate_classification_metrics(y_true, preds_inverted)
    assert m_inverted["accuracy"] == 0.0
    assert m_inverted["recall"] == 0.0
    assert m_inverted["precision"] == 0.0
    assert m_inverted["f1_score"] == 0.0

    # Partial predictions
    preds_partial = [0, 0, 0, 0, 0, 0, 1, 1]
    m_partial = calculate_classification_metrics(y_true, preds_partial)
    assert m_partial["accuracy"] == 0.75
    assert m_partial["recall"] == 0.50
    assert m_partial["precision"] == 1.0


def test_roc_data_based_on_actual_prediction_scores():
    """Verify ROC curve coordinates and AUC reflect genuine empirical probabilities, not formula."""
    y_true = [0, 0, 1, 1]
    
    # Perfect separation: class 0 gets low prob, class 1 gets high prob
    probs_perfect = [0.1, 0.2, 0.8, 0.9]
    roc_perfect = calculate_roc_curve_data(y_true, probs_perfect)
    assert roc_perfect["valid"] is True
    assert roc_perfect["roc_auc"] == 1.0
    assert 0.0 in roc_perfect["fpr"]
    assert 1.0 in roc_perfect["tpr"]

    # Inverted scores
    probs_inverted = [0.9, 0.8, 0.2, 0.1]
    roc_inverted = calculate_roc_curve_data(y_true, probs_inverted)
    assert roc_inverted["valid"] is True
    assert roc_inverted["roc_auc"] == 0.0

    # Intermediate scores
    probs_intermediate = [0.2, 0.7, 0.4, 0.9]
    roc_inter = calculate_roc_curve_data(y_true, probs_intermediate)
    assert roc_inter["valid"] is True
    assert 0.0 < roc_inter["roc_auc"] < 1.0
    # FPR/TPR lists must match unique threshold cutoffs, not 100 linspace points
    assert len(roc_inter["fpr"]) == len(roc_inter["tpr"])
    assert len(roc_inter["fpr"]) <= len(y_true) + 2


def test_pr_curve_data_based_on_actual_prediction_scores():
    """Verify Precision-Recall curve derives from empirical thresholds."""
    y_true = [0, 0, 1, 1]
    probs = [0.1, 0.3, 0.7, 0.9]
    pr_data = calculate_pr_curve_data(y_true, probs)
    assert pr_data["valid"] is True
    assert pr_data["average_precision"] == 1.0
    assert pr_data["baseline_prevalence"] == 0.5


def test_single_class_dataset_handled_gracefully():
    """Verify that a dataset containing only one class is handled honestly without crashing."""
    y_true_all_zeros = [0, 0, 0, 0]
    preds = [0, 0, 0, 0]
    probs = [0.1, 0.2, 0.1, 0.3]

    # Metrics computation: ROC-AUC is mathematically undefined for single class
    metrics = calculate_classification_metrics(y_true_all_zeros, preds, probs)
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] is None
    assert "single-class" in metrics.get("roc_auc_status", "").lower()

    # ROC curve generator should report valid=False and give explicit reason
    roc_res = calculate_roc_curve_data(y_true_all_zeros, probs)
    assert roc_res["valid"] is False
    assert roc_res["roc_auc"] is None
    assert "single-class" in roc_res["reason"].lower()

    # PR curve generator should report valid=False when no positive samples exist
    pr_res = calculate_pr_curve_data(y_true_all_zeros, probs)
    assert pr_res["valid"] is False
    assert pr_res["average_precision"] is None


def test_zero_division_handling_when_no_positives_predicted():
    """Verify zero-division handling when model predicts all negative class (as on exoTest.csv)."""
    y_true = [0, 0, 0, 1, 1]
    # Predict all zeros
    preds_all_zeros = [0, 0, 0, 0, 0]
    metrics = calculate_classification_metrics(y_true, preds_all_zeros)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1_score"] == 0.0
    assert metrics["accuracy"] == 0.60


def test_empty_or_mismatched_inputs_rejected():
    """Verify input validation catches empty or length-mismatched arrays."""
    with pytest.raises(ValueError):
        calculate_classification_metrics([], [])

    with pytest.raises(ValueError):
        calculate_classification_metrics([0, 1], [0])

    with pytest.raises(ValueError):
        generate_confusion_matrix_data([], [])

    with pytest.raises(ValueError):
        calculate_roc_curve_data([0, 1], [0.5])


def test_model_metadata_contains_no_unsupported_100_percent_claims():
    """Verify that models/model_metadata.json does not claim fabricated 100% (1.0) metrics."""
    meta_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "models", "model_metadata.json"
    )
    assert os.path.exists(meta_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    metrics = meta.get("metrics", {})
    # Accuracy on test benchmark is 0.9912, not 1.0
    assert metrics.get("accuracy") != 1.0
    assert metrics.get("precision") != 1.0
    assert metrics.get("recall") != 1.0
    assert metrics.get("f1_score") != 1.0
    assert metrics.get("roc_auc") != 1.0

    # Ensure warning is present
    assert "imbalance_warning" in metrics or "verified_on_test_benchmark" in metrics


def test_benchmark_evaluation_artifact_integrity():
    """Verify the reproducible benchmark evaluation artifact reflects true empirical measurements."""
    artifact_path = DEFAULT_BENCHMARK_ARTIFACT_PATH
    assert os.path.exists(artifact_path), f"Benchmark artifact missing at {artifact_path}"

    eval_data = load_benchmark_evaluation_artifact(artifact_path)
    assert eval_data is not None

    meta = eval_data.get("metadata", {})
    assert meta.get("dataset_name") == "exoTest.csv"
    assert meta.get("verified_authentic") is True
    assert meta.get("is_simulated_or_formula") is False

    telemetry = eval_data.get("dataset_telemetry", {})
    assert telemetry.get("total_samples") == 570
    assert telemetry.get("positive_samples") == 5
    assert telemetry.get("negative_samples") == 565

    cm = eval_data.get("confusion_matrix", {})
    # Actual performance on exoTest.csv has 0 true positives, 5 false negatives, 565 true negatives, 0 false positives
    assert cm.get("true_positives") == 0
    assert cm.get("false_negatives") == 5
    assert cm.get("true_negatives") == 565
    assert cm.get("false_positives") == 0

    metrics = eval_data.get("classification_metrics", {})
    assert metrics.get("accuracy") == 0.9912
    assert metrics.get("recall") == 0.0
    assert metrics.get("precision") == 0.0

    # Verify ROC curve has real points, not mathematical formula
    roc = eval_data.get("roc_curve", {})
    assert len(roc.get("fpr", [])) > 0
    assert len(roc.get("fpr", [])) == len(roc.get("tpr", []))
    assert roc.get("roc_auc") == pytest.approx(0.6430, abs=0.01)


def test_static_demonstration_values_removed_from_model_lab():
    """Verify that pages/05_Model_Lab.py contains no hardcoded [[238, 2], [0, 60]] or (1 - fpr)**4.2."""
    model_lab_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "pages", "05_Model_Lab.py"
    )
    with open(model_lab_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Hardcoded confusion matrix count
    assert "[[238, 2], [0, 60]]" not in content
    assert "[[238,2],[0,60]]" not in content

    # Mathematical polynomial ROC curve formula
    assert "(1.0 - fpr)**4.2" not in content
    assert "(1 - fpr)**4.2" not in content

    # Hardcoded 5-fold cross-validation scores
    assert "99.30% ± 0.15%" not in content
    assert "99.30% +- 0.15%" not in content
    assert "Fold 1 (1,131 targets):" not in content
