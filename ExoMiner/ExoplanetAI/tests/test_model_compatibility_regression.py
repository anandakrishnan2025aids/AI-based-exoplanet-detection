"""
test_model_compatibility_regression.py - Comprehensive Regression Test Suite for Task 6.
Validates:
1. Feature names, ordering, units, and preprocessing consistency.
2. Rejection or clear warning for incompatible feature inputs (e.g. unnormalized difference flux).
3. Correct positive-class probability selection (including non-standard class orderings).
4. No-transit data not treated as proof that no planet exists (honest inconclusive semantics).
5. Insufficient data producing an honest inconclusive/error result.
6. Model Lab and metadata clearly distinguishing valid evaluation metrics from potentially incompatible benchmark results.
7. End-to-end lightcurve prediction pipeline consistency.
8. Report generation with honest non-detection semantics and error-free execution.
"""

import os
import sys
import json
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.prediction import (
    ExoplanetPredictor,
    get_default_predictor,
    FeatureValidationError,
    ModelInferenceError,
    REQUIRED_PHYSICAL_FEATURES
)
from backend.feature_engineering import (
    ExoplanetFeatureExtractor,
    STANDARD_FEATURE_NAMES,
    extract_features
)
from backend.metrics import (
    load_benchmark_evaluation_artifact,
    DEFAULT_BENCHMARK_ARTIFACT_PATH,
    compute_batch_summary
)
from components.report import generate_candidate_report


# =========================================================================
# 1. Feature Names, Ordering, Units, and Preprocessing Consistency
# =========================================================================

def test_feature_names_and_ordering_consistency():
    """Verify that expected features match the trained Random Forest model's schema in exact names and order."""
    predictor = get_default_predictor()
    expected_order = [
        "transit_depth",
        "period_days",
        "duration_hours",
        "snr",
        "flux_std",
        "flux_skew",
        "flux_kurtosis",
        "flux_min",
        "flux_median"
    ]

    assert STANDARD_FEATURE_NAMES == expected_order
    assert predictor.feature_names == expected_order

    if hasattr(predictor.model, "feature_names_in_"):
        assert list(predictor.model.feature_names_in_) == expected_order

    # Validate that _align_features returns DataFrame with exact column order
    input_dict = {
        "transit_depth": 0.015,
        "period_days": 3.52,
        "duration_hours": 2.4,
        "snr": 12.0,
        "flux_std": 0.002,
        "flux_skew": -1.2,
        "flux_kurtosis": 3.5,
        "flux_min": 0.985,
        "flux_median": 1.0
    }
    df_aligned = predictor._align_features(input_dict)
    assert list(df_aligned.columns) == expected_order


# =========================================================================
# 2. Rejection of Incompatible Feature Inputs
# =========================================================================

def test_rejection_of_unnormalized_difference_flux_inputs():
    """Verify that unnormalized difference flux values (e.g. from exoTest.csv) are rejected."""
    predictor = get_default_predictor()

    base_valid = {
        "transit_depth": 0.015,
        "period_days": 3.52,
        "duration_hours": 2.4,
        "snr": 12.0,
        "flux_std": 0.002,
        "flux_skew": -1.2,
        "flux_kurtosis": 3.5,
        "flux_min": 0.985,
        "flux_median": 1.0
    }

    # 1. Unnormalized flux_std (> 1.0, e.g. 394 counts from exoTest.csv Target #3)
    bad_std = dict(base_valid, flux_std=394.45)
    with pytest.raises(FeatureValidationError, match="Incompatible feature input.*flux_std"):
        predictor.predict_single(bad_std)

    # 2. Zero-centered flux_median (<= 0.0, e.g. 0.0 from exoTest.csv difference flux)
    bad_med = dict(base_valid, flux_median=0.0)
    with pytest.raises(FeatureValidationError, match="Incompatible feature input.*flux_median"):
        predictor.predict_single(bad_med)

    # 3. Unnormalized transit depth (> 1.0, e.g. 2044.8 from exoTest.csv)
    bad_depth = dict(base_valid, transit_depth=2044.8)
    with pytest.raises(FeatureValidationError, match="Physical constraint violated.*transit_depth"):
        predictor.predict_single(bad_depth)

    # 4. Out-of-bounds flux_min (e.g. -2289 counts from exoTest.csv)
    bad_min = dict(base_valid, flux_min=-2289.09)
    with pytest.raises(FeatureValidationError, match="Incompatible feature input.*flux_min"):
        predictor.predict_single(bad_min)


# =========================================================================
# 3. Correct Positive-Class Probability Selection
# =========================================================================

def test_positive_class_probability_selection_with_standard_classes():
    """Verify that predictor correctly identifies positive class probability column."""
    predictor = get_default_predictor()
    assert list(predictor.model.classes_) == [0, 1]

    valid_exo = {
        "transit_depth": 0.025,
        "period_days": 3.52,
        "duration_hours": 2.8,
        "snr": 22.0,
        "flux_std": 0.0012,
        "flux_skew": -2.1,
        "flux_kurtosis": 6.5,
        "flux_min": -0.025,
        "flux_median": 1.0
    }
    res = predictor.predict_single(valid_exo)
    # Probability of exoplanet must be non-trivial for strong candidate
    assert res["probability_exoplanet"] >= 0.50
    assert abs((res["probability_exoplanet"] + res["probability_non_exoplanet"]) - 1.0) < 1e-4


def test_positive_class_probability_selection_with_permuted_classes():
    """Verify that predictor dynamically looks up positive class index even if classes_ is reversed [1, 0]."""
    predictor = ExoplanetPredictor()

    class MockReversedModel:
        classes_ = np.array([1, 0])
        feature_names_in_ = np.array(STANDARD_FEATURE_NAMES)

        def predict_proba(self, X):
            # Returns [P(class 1), P(class 0)]
            return np.array([[0.85, 0.15]])

        def predict(self, X):
            return np.array([1])

    predictor.model = MockReversedModel()

    valid_feats = {
        "transit_depth": 0.02,
        "period_days": 3.0,
        "duration_hours": 2.0,
        "snr": 15.0,
        "flux_std": 0.002,
        "flux_skew": -1.5,
        "flux_kurtosis": 4.0
    }
    res = predictor.predict_single(valid_feats)
    # The exoplanet probability must be 0.85 (from index 0 where class==1), NOT 0.15 (from index 1)!
    assert res["probability_exoplanet"] == 0.85
    assert res["probability_non_exoplanet"] == 0.15
    assert res["is_exoplanet"] is True


# =========================================================================
# 4. No-Transit Data Not Treated as Proof of No Planet
# =========================================================================

def test_no_transit_data_treated_as_inconclusive_not_definitive():
    """
    Absence of evidence is not evidence of absence.
    When no transit is detected, the result must be 'No transit detected — inconclusive',
    never 'Non-Exoplanet with 100% confidence'.
    """
    predictor = get_default_predictor()
    extractor = ExoplanetFeatureExtractor()

    # Pure Gaussian noise light curve (no transit)
    np.random.seed(42)
    time = np.linspace(0, 20, 1000)
    flux = 1.0 + np.random.normal(0, 0.003, 1000)

    feats = extractor.extract_from_series(time, flux)
    assert feats["detection_status"] == "no_transit_detected"

    res = predictor.predict_single(feats)

    # Verdict must be scientifically honest
    assert res["prediction"] == "No transit detected — inconclusive"
    assert res["is_exoplanet"] is False

    # Confidence must NOT be manufactured 100% (1.0)
    assert res["confidence"] == 0.0
    assert res["confidence_grade"] == "Inconclusive"

    # Explanatory narrative must acknowledge inclination / detection limits
    summary = res["vetting_summary"]
    assert "inconclusive" in summary.lower()
    assert "does not prove the absence" in summary.lower() or "not rule out" in summary.lower()


# =========================================================================
# 5. Insufficient Data Producing Honest Inconclusive / Error Result
# =========================================================================

def test_insufficient_cadences_returns_unable_to_evaluate():
    """Short light curve (< 10 points) must return 'Unable to evaluate — insufficient or invalid data'."""
    predictor = get_default_predictor()
    extractor = ExoplanetFeatureExtractor()

    time = np.array([1.0, 2.0, 3.0, 4.0])
    flux = np.array([1.0, 0.99, 1.0, 1.01])

    feats = extractor.extract_from_series(time, flux)
    assert feats["detection_status"] == "insufficient_data"

    res = predictor.predict_single(feats)

    assert res["prediction"] == "Unable to evaluate — insufficient or invalid data"
    assert res["is_exoplanet"] is False
    assert res["confidence"] == 0.0
    assert res["confidence_grade"] == "Inconclusive"
    assert res["model_type"] == "Data Quality Vetting"
    assert "insufficient samples" in res["vetting_summary"].lower()


def test_constant_flux_returns_unable_to_evaluate():
    """Constant flat flux (zero variance) must return 'Unable to evaluate — insufficient or invalid data'."""
    predictor = get_default_predictor()
    extractor = ExoplanetFeatureExtractor()

    time = np.linspace(0, 10, 500)
    flux = np.full(500, 1.0)

    feats = extractor.extract_from_series(time, flux)
    assert feats["detection_status"] == "constant_flux"

    res = predictor.predict_single(feats)

    assert res["prediction"] == "Unable to evaluate — insufficient or invalid data"
    assert res["is_exoplanet"] is False
    assert res["confidence"] == 0.0


# =========================================================================
# 6. Model Lab Distinguishes Valid Metrics from Incompatible Benchmark
# =========================================================================

def test_benchmark_artifact_marked_inconclusive_and_diagnostic():
    """Verify that models/benchmark_evaluation.json is marked as inconclusive historical diagnostic."""
    artifact = load_benchmark_evaluation_artifact(DEFAULT_BENCHMARK_ARTIFACT_PATH)
    assert artifact is not None

    meta = artifact.get("metadata", {})
    assert meta.get("evaluation_validity") == "inconclusive_feature_mismatch"
    assert meta.get("compatibility_status") == "incompatible_feature_pipeline"
    assert meta.get("is_fair_assessment") is False
    assert "inconclusive" in meta.get("validity_notice", "").lower()

    # Historical metrics must be preserved intact without fabrication
    metrics = artifact.get("classification_metrics", {})
    assert metrics.get("accuracy") == 0.9912
    assert metrics.get("recall") == 0.0
    assert metrics.get("precision") == 0.0
    assert metrics.get("roc_auc") == pytest.approx(0.643, abs=0.01)


def test_model_metadata_discloses_benchmark_inconclusive_status():
    """Verify that models/model_metadata.json discloses feature distribution mismatch and invalidity."""
    meta_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "models", "model_metadata.json"
    )
    with open(meta_path, "r") as f:
        meta = json.load(f)

    metrics = meta.get("metrics", {})
    assert metrics.get("evaluation_validity") == "inconclusive_feature_mismatch"
    assert metrics.get("is_fair_assessment") is False
    assert "inconclusive" in metrics.get("imbalance_warning", "").lower()


# =========================================================================
# 7. End-to-End Lightcurve Prediction Pipeline Semantics
# =========================================================================

def test_predict_lightcurve_clean_transit_candidate():
    """End-to-end lightcurve inference on a strong transit signal returns Candidate status."""
    predictor = get_default_predictor()
    np.random.seed(42)
    # Use Kepler-10b realistic test signal matching app demo curve
    t = np.linspace(120.0, 145.0, 1000)
    f = 1.0 - 0.0152 * np.exp(-((t % 0.837 - 0.2)**2)/(2*0.04**2)) + np.random.normal(0, 0.002, len(t))

    res = predictor.predict_lightcurve(t, f)
    assert res["prediction"] == "Transit detected \u2014 candidate for further analysis"
    assert res["is_exoplanet"] is True
    assert res["confidence"] >= 0.50
    assert "preprocessed_data" in res


def test_predict_lightcurve_white_noise_inconclusive():
    """End-to-end lightcurve inference on pure noise returns Inconclusive status."""
    predictor = get_default_predictor()
    np.random.seed(123)
    time = np.linspace(0, 20, 1000)
    flux = 1.0 + np.random.normal(0, 0.005, len(time))

    res = predictor.predict_lightcurve(time, flux)
    assert res["prediction"] == "No transit detected — inconclusive"
    assert res["is_exoplanet"] is False
    assert res["confidence"] == 0.0


# =========================================================================
# 8. Report Generation Integrity
# =========================================================================

def test_report_generation_safe_and_scientifically_honest():
    """Verify that generate_candidate_report formats inconclusive and candidate dossiers without exceptions."""
    # Test inconclusive result
    inconclusive_res = {
        "prediction": "No transit detected — inconclusive",
        "is_exoplanet": False,
        "confidence": 0.0,
        "probability_exoplanet": 0.0,
        "probability_non_exoplanet": 0.0,
        "confidence_grade": "Inconclusive",
        "model_type": "Deterministic Vetting (No Transit Detected)",
        "is_heuristic": False,
        "features_used": {
            "transit_depth": 0.0,
            "period_days": None,
            "duration_hours": None,
            "snr": 0.0,
            "flux_std": 0.004,
            "flux_skew": 0.1,
            "flux_kurtosis": 0.05
        },
        "vetting_summary": "No statistically significant transit dips detected."
    }

    report = generate_candidate_report("KIC-TestNoise", inconclusive_res, rows_count=1000)
    assert "NO TRANSIT DETECTED — INCONCLUSIVE" in report
    assert "MODEL SCORE          : N/A (Inconclusive)" in report
    assert "KIC-TestNoise" in report

    # Test candidate result
    candidate_res = {
        "prediction": "Transit detected — candidate for further analysis",
        "is_exoplanet": True,
        "confidence": 0.942,
        "probability_exoplanet": 0.942,
        "probability_non_exoplanet": 0.058,
        "confidence_grade": "High",
        "model_type": "Random Forest",
        "is_heuristic": False,
        "features_used": {
            "transit_depth": 0.015,
            "period_days": 3.52,
            "duration_hours": 2.4,
            "snr": 18.5,
            "flux_std": 0.001,
            "flux_skew": -1.8,
            "flux_kurtosis": 5.2
        },
        "vetting_summary": "Characteristic planetary transit signature detected."
    }

    rep_cand = generate_candidate_report("Kepler-10b", candidate_res, rows_count=2000)
    assert "TRANSIT DETECTED — CANDIDATE FOR FURTHER ANALYSIS" in rep_cand
    assert "94.20%" in rep_cand


# =========================================================================
# 9. Batch Summary Utility Consistency
# =========================================================================

def test_batch_summary_detects_all_candidate_label_variants():
    """Verify compute_batch_summary aggregates both legacy and new candidate labels."""
    df_preds = pd.DataFrame([
        {"prediction": "Transit detected — candidate for further analysis", "confidence": 0.95},
        {"prediction": "Exoplanet", "confidence": 0.90},
        {"prediction": "Candidate (Heuristic)", "confidence": 0.75},
        {"prediction": "No transit detected — inconclusive", "confidence": 0.0},
        {"prediction": "Unable to evaluate — insufficient or invalid data", "confidence": 0.0}
    ])
    summary = compute_batch_summary(df_preds)
    assert summary["total_stars"] == 5
    assert summary["exoplanets_detected"] == 3
    assert summary["non_exoplanets"] == 2
    assert summary["detection_rate_pct"] == 60.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
