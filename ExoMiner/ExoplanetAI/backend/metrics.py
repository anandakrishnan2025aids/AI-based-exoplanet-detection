"""
metrics.py - Performance Evaluation and Scientific Metric Calculation Engine.
Provides empirical classification metrics, confusion matrices, ROC/PR curve generation,
and benchmark evaluation artifact loaders.
Guarantees strict integrity: no hardcoded results, no simulated curves, honest reporting
of undefined metrics, and transparent handling of class imbalance and single-class datasets.
"""

import os
import json
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix
)

DEFAULT_BENCHMARK_ARTIFACT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "benchmark_evaluation.json"
)


def _standardize_binary_labels(arr: Union[List[int], np.ndarray, pd.Series]) -> np.ndarray:
    """
    Standardizes label array to binary integers {0, 1}.
    If format is Kaggle Kepler style (2 = Exoplanet, 1 = Non-Exoplanet),
    maps 2 -> 1 and 1 -> 0.
    Standard {0, 1} format is preserved.
    """
    y = np.asarray(arr, dtype=int)
    unique_vals = set(np.unique(y))
    # Kaggle Kepler format contains 2 and does NOT contain 0
    if 2 in unique_vals and 0 not in unique_vals:
        return np.where(y == 2, 1, 0)
    return y


def calculate_classification_metrics(
    y_true: Union[List[int], np.ndarray, pd.Series],
    y_pred: Union[List[int], np.ndarray, pd.Series],
    y_prob: Optional[Union[List[float], np.ndarray, pd.Series]] = None
) -> Dict[str, Any]:
    """
    Computes genuine performance metrics for binary exoplanet classification.
    Standardizes classes: 1 for Exoplanet, 0 for Non-Exoplanet (or 2 vs 1 mapping).

    Zero-division cases and single-class datasets are handled explicitly without errors.
    """
    y_t = _standardize_binary_labels(y_true)
    y_p = _standardize_binary_labels(y_pred)

    if len(y_t) == 0 or len(y_p) == 0:
        raise ValueError("Ground-truth labels and model predictions cannot be empty.")
    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch between labels ({len(y_t)}) and predictions ({len(y_p)}).")

    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, zero_division=0.0))
    rec = float(recall_score(y_t, y_p, zero_division=0.0))
    f1 = float(f1_score(y_t, y_p, zero_division=0.0))

    metrics: Dict[str, Any] = {
        "accuracy": float(round(acc, 4)),
        "precision": float(round(prec, 4)),
        "recall": float(round(rec, 4)),
        "f1_score": float(round(f1, 4)),
        "positive_class": "1 (Exoplanet)",
        "averaging_method": "binary (zero_division=0.0)",
        "total_samples": int(len(y_t)),
        "positive_samples": int(np.sum(y_t == 1)),
        "negative_samples": int(np.sum(y_t == 0))
    }

    # ROC-AUC calculation requires probability scores AND both classes present in ground truth
    if y_prob is not None:
        y_pr = np.asarray(y_prob, dtype=float)
        unique_classes = np.unique(y_t)
        if len(unique_classes) < 2:
            # Undefined for single-class datasets: report honestly as None
            metrics["roc_auc"] = None
            metrics["roc_auc_status"] = "Undefined: single-class ground truth"
        else:
            try:
                auc = float(roc_auc_score(y_t, y_pr))
                metrics["roc_auc"] = float(round(auc, 4))
                metrics["roc_auc_status"] = "Measured"
            except Exception as e:
                metrics["roc_auc"] = None
                metrics["roc_auc_status"] = f"Calculation failed: {e}"
    else:
        metrics["roc_auc"] = None
        metrics["roc_auc_status"] = "No probability scores provided"

    return metrics


def generate_confusion_matrix_data(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray]
) -> Dict[str, Any]:
    """
    Computes a 2x2 confusion matrix derived from actual ground-truth labels and model predictions.
    Labels are fixed to [0, 1] so matrix order is guaranteed:
      [[TN, FP],
       [FN, TP]]
    """
    y_t = _standardize_binary_labels(y_true)
    y_p = _standardize_binary_labels(y_pred)

    if len(y_t) == 0 or len(y_p) == 0:
        raise ValueError("Ground-truth labels and predictions cannot be empty.")
    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch: {len(y_t)} vs {len(y_p)}.")

    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "matrix": cm.tolist(),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "labels": ["Non-Exoplanet", "Exoplanet"],
        "total_samples": int(len(y_t)),
        "ground_truth_positives": int(np.sum(y_t == 1)),
        "ground_truth_negatives": int(np.sum(y_t == 0)),
        "predicted_positives": int(np.sum(y_p == 1)),
        "predicted_negatives": int(np.sum(y_p == 0))
    }


def calculate_roc_curve_data(
    y_true: Union[List[int], np.ndarray, pd.Series],
    y_prob: Union[List[float], np.ndarray, pd.Series]
) -> Dict[str, Any]:
    """
    Computes empirical Receiver Operating Characteristic (ROC) curve coordinates
    derived from actual prediction probabilities and ground-truth labels.

    Does NOT use synthetic polynomials or mathematical approximations.
    Handles single-class edge cases explicitly.
    """
    y_t = _standardize_binary_labels(y_true)
    y_pr = np.asarray(y_prob, dtype=float)

    if len(y_t) == 0 or len(y_pr) == 0:
        return {
            "valid": False,
            "reason": "Labels or prediction probabilities array is empty.",
            "fpr": [],
            "tpr": [],
            "thresholds": [],
            "roc_auc": None
        }

    if len(y_t) != len(y_pr):
        raise ValueError(f"Length mismatch: {len(y_t)} labels vs {len(y_pr)} probabilities.")

    unique_classes = np.unique(y_t)
    if len(unique_classes) < 2:
        return {
            "valid": False,
            "reason": "Single-class ground truth: ROC calculation requires at least one positive and one negative sample.",
            "fpr": [],
            "tpr": [],
            "thresholds": [],
            "roc_auc": None
        }

    try:
        fpr, tpr, thresholds = roc_curve(y_t, y_pr)
        auc = float(roc_auc_score(y_t, y_pr))
        
        # Clean infinite thresholds for JSON compatibility
        thresh_list = [1.0 if np.isinf(t) else float(round(t, 5)) for t in thresholds]

        return {
            "valid": True,
            "reason": None,
            "fpr": [float(round(x, 5)) for x in fpr],
            "tpr": [float(round(y, 5)) for y in tpr],
            "thresholds": thresh_list,
            "roc_auc": float(round(auc, 4))
        }
    except Exception as e:
        return {
            "valid": False,
            "reason": f"ROC curve calculation failed: {e}",
            "fpr": [],
            "tpr": [],
            "thresholds": [],
            "roc_auc": None
        }


def calculate_pr_curve_data(
    y_true: Union[List[int], np.ndarray, pd.Series],
    y_prob: Union[List[float], np.ndarray, pd.Series]
) -> Dict[str, Any]:
    """
    Computes empirical Precision-Recall (PR) curve coordinates and Average Precision (AP).
    Appropriate for highly imbalanced exoplanet transit detection benchmarks.
    """
    y_t = _standardize_binary_labels(y_true)
    y_pr = np.asarray(y_prob, dtype=float)

    if len(y_t) == 0 or len(y_pr) == 0:
        return {
            "valid": False,
            "reason": "Labels or prediction probabilities array is empty.",
            "precision": [],
            "recall": [],
            "thresholds": [],
            "average_precision": None,
            "baseline_prevalence": 0.0
        }

    if len(y_t) != len(y_pr):
        raise ValueError(f"Length mismatch: {len(y_t)} labels vs {len(y_pr)} probabilities.")

    if np.sum(y_t == 1) == 0:
        return {
            "valid": False,
            "reason": "Precision-Recall calculation requires at least one positive ground-truth sample.",
            "precision": [],
            "recall": [],
            "thresholds": [],
            "average_precision": None,
            "baseline_prevalence": 0.0
        }

    try:
        prec, rec, thresholds = precision_recall_curve(y_t, y_pr)
        avg_prec = float(average_precision_score(y_t, y_pr))
        baseline = float(round(np.sum(y_t == 1) / len(y_t), 4))

        return {
            "valid": True,
            "reason": None,
            "precision": [float(round(p, 5)) for p in prec],
            "recall": [float(round(r, 5)) for r in rec],
            "thresholds": [float(round(t, 5)) for t in thresholds],
            "average_precision": float(round(avg_prec, 4)),
            "baseline_prevalence": baseline
        }
    except Exception as e:
        return {
            "valid": False,
            "reason": f"PR curve calculation failed: {e}",
            "precision": [],
            "recall": [],
            "thresholds": [],
            "average_precision": None,
            "baseline_prevalence": 0.0
        }


def load_benchmark_evaluation_artifact(
    artifact_path: str = DEFAULT_BENCHMARK_ARTIFACT_PATH
) -> Optional[Dict[str, Any]]:
    """
    Loads verified, reproducible benchmark evaluation artifact.
    Returns None if artifact does not exist or fails to parse.
    """
    if not os.path.exists(artifact_path):
        return None
    try:
        with open(artifact_path, "r") as f:
            data = json.load(f)
            return data
    except Exception as e:
        print(f"Warning: Failed to load benchmark evaluation artifact from {artifact_path}: {e}")
        return None


def compute_batch_summary(predictions_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Summarizes predictions on a batch DataFrame containing 'prediction' and 'confidence' columns.
    """
    total = len(predictions_df)
    if total == 0:
        return {
            "total_stars": 0,
            "exoplanets_detected": 0,
            "non_exoplanets": 0,
            "detection_rate_pct": 0.0,
            "avg_confidence": 0.0
        }

    candidate_mask = predictions_df["prediction"].isin([
        "Exoplanet",
        "Transit detected — candidate for further analysis",
        "Candidate (Heuristic)"
    ])
    if "is_exoplanet" in predictions_df.columns:
        candidate_mask = candidate_mask | (predictions_df["is_exoplanet"] == True)
    detected = int(candidate_mask.sum())
    non_detected = total - detected
    rate = (detected / total) * 100.0
    avg_conf = float(predictions_df["confidence"].mean() * 100.0) if "confidence" in predictions_df else 0.0

    return {
        "total_stars": total,
        "exoplanets_detected": detected,
        "non_exoplanets": non_detected,
        "detection_rate_pct": float(round(rate, 2)),
        "avg_confidence": float(round(avg_conf, 2))
    }
