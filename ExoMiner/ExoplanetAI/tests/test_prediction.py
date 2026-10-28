"""
test_prediction.py - Unit and regression tests for ExoplanetAI prediction engine.
Validates input validation, failure modes, error handling, and model inference.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.prediction import (
    ExoplanetPredictor,
    get_default_predictor,
    FeatureValidationError,
    ModelInferenceError,
    REQUIRED_PHYSICAL_FEATURES
)


def test_predictor_singleton():
    p1 = get_default_predictor()
    p2 = get_default_predictor()
    assert p1 is p2


def test_predict_single_exoplanet():
    predictor = get_default_predictor()
    exo_features = {
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
    result = predictor.predict_single(exo_features)

    assert "prediction" in result
    assert "confidence" in result
    assert "is_exoplanet" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["is_exoplanet"] is True
    assert result["prediction"] == "Transit detected — candidate for further analysis"
    assert result["model_type"] == "Random Forest"
    assert result["is_heuristic"] is False


def test_predict_single_noise():
    predictor = get_default_predictor()
    noise_features = {
        "transit_depth": 0.0001,
        "period_days": 10.0,
        "duration_hours": 1.0,
        "snr": 0.5,
        "flux_std": 0.008,
        "flux_skew": 0.2,
        "flux_kurtosis": 0.1,
        "flux_min": -0.0001,
        "flux_median": 1.0
    }
    result = predictor.predict_single(noise_features)

    assert result["is_exoplanet"] is False
    assert result["prediction"] == "No transit detected — inconclusive"
    assert result["model_type"] == "Random Forest"
    assert result["is_heuristic"] is False


def test_predict_batch():
    predictor = get_default_predictor()
    df = pd.DataFrame([
        {"transit_depth": 0.02, "period_days": 2.0, "duration_hours": 2.0, "snr": 18.0, "flux_std": 0.001, "flux_skew": -1.5, "flux_kurtosis": 4.0},
        {"transit_depth": 0.0002, "period_days": 5.0, "duration_hours": 1.0, "snr": 0.8, "flux_std": 0.005, "flux_skew": 0.1, "flux_kurtosis": 0.1}
    ])
    res_df = predictor.predict_batch(df)

    assert "prediction" in res_df.columns
    assert "confidence" in res_df.columns
    assert "model_type" in res_df.columns
    assert "is_heuristic" in res_df.columns
    assert len(res_df) == 2
    assert res_df.iloc[0]["prediction"] == "Transit detected — candidate for further analysis"
    assert res_df.iloc[1]["prediction"] == "No transit detected — inconclusive"
    assert bool(res_df.iloc[0]["is_heuristic"]) is False


# ---------------- TASK 3 REGRESSION TESTS ----------------

def test_empty_dict_rejected():
    """Empty feature dict must be rejected with FeatureValidationError, never given default values."""
    predictor = get_default_predictor()
    with pytest.raises(FeatureValidationError, match="cannot be empty"):
        predictor.predict_single({})


def test_missing_required_features_rejected():
    """Missing any mandatory physical feature must raise FeatureValidationError."""
    predictor = get_default_predictor()
    # Missing 'snr' and 'duration_hours'
    incomplete_features = {
        "transit_depth": 0.02,
        "period_days": 3.5,
        "flux_std": 0.001,
        "flux_skew": -1.5,
        "flux_kurtosis": 4.0
    }
    with pytest.raises(FeatureValidationError, match="Missing required astrophysical feature"):
        predictor.predict_single(incomplete_features)


def test_nan_values_rejected():
    """NaN numeric values must be explicitly rejected."""
    predictor = get_default_predictor()
    nan_features = {
        "transit_depth": np.nan,
        "period_days": 3.5,
        "duration_hours": 2.0,
        "snr": 15.0,
        "flux_std": 0.001,
        "flux_skew": -1.5,
        "flux_kurtosis": 4.0
    }
    with pytest.raises(FeatureValidationError, match="non-finite value"):
        predictor.predict_single(nan_features)


def test_infinite_values_rejected():
    """Infinite numeric values must be explicitly rejected."""
    predictor = get_default_predictor()
    inf_features = {
        "transit_depth": 0.02,
        "period_days": float("inf"),
        "duration_hours": 2.0,
        "snr": 15.0,
        "flux_std": 0.001,
        "flux_skew": -1.5,
        "flux_kurtosis": 4.0
    }
    with pytest.raises(FeatureValidationError, match="non-finite value"):
        predictor.predict_single(inf_features)


def test_non_numeric_values_rejected():
    """Non-numeric string values must raise FeatureValidationError."""
    predictor = get_default_predictor()
    bad_type_features = {
        "transit_depth": "deep_transit",
        "period_days": 3.5,
        "duration_hours": 2.0,
        "snr": 15.0,
        "flux_std": 0.001,
        "flux_skew": -1.5,
        "flux_kurtosis": 4.0
    }
    with pytest.raises(FeatureValidationError, match="must be a numeric value"):
        predictor.predict_single(bad_type_features)


def test_physical_boundary_violations_rejected():
    """Negative periods, durations, depths, or depths > 1.0 must be rejected."""
    predictor = get_default_predictor()
    base = {
        "transit_depth": 0.02,
        "period_days": 3.5,
        "duration_hours": 2.0,
        "snr": 15.0,
        "flux_std": 0.001,
        "flux_skew": -1.5,
        "flux_kurtosis": 4.0
    }

    # Negative period
    bad_p = dict(base, period_days=-2.0)
    with pytest.raises(FeatureValidationError, match="must be strictly positive"):
        predictor.predict_single(bad_p)

    # Negative duration
    bad_dur = dict(base, duration_hours=0.0)
    with pytest.raises(FeatureValidationError, match="must be strictly positive"):
        predictor.predict_single(bad_dur)

    # Depth > 1.0
    bad_depth = dict(base, transit_depth=1.5)
    with pytest.raises(FeatureValidationError, match="must be between 0.0 and 1.0"):
        predictor.predict_single(bad_depth)

    # Negative SNR
    bad_snr = dict(base, snr=-5.0)
    with pytest.raises(FeatureValidationError, match="cannot be negative"):
        predictor.predict_single(bad_snr)


def test_model_inference_failure_raises_model_inference_error():
    """When model inference fails, ModelInferenceError must be raised unless explicitly allowed."""
    predictor = ExoplanetPredictor()
    class BrokenModel:
        def predict_proba(self, X):
            raise RuntimeError("CUDA out of memory simulation")

    predictor.model = BrokenModel()
    valid_features = {
        "transit_depth": 0.02,
        "period_days": 3.5,
        "duration_hours": 2.0,
        "snr": 15.0,
        "flux_std": 0.001,
        "flux_skew": -1.5,
        "flux_kurtosis": 4.0
    }

    with pytest.raises(ModelInferenceError, match="Model inference failed"):
        predictor.predict_single(valid_features, allow_heuristic=False)


def test_explicit_opt_in_heuristic_mode():
    """Opting into heuristic mode when model fails returns clearly marked heuristic results."""
    predictor = ExoplanetPredictor()
    predictor.model = None

    valid_features = {
        "transit_depth": 0.02,
        "period_days": 3.5,
        "duration_hours": 2.0,
        "snr": 15.0,
        "flux_std": 0.001,
        "flux_skew": -1.5,
        "flux_kurtosis": 4.0
    }

    res = predictor.predict_single(valid_features, allow_heuristic=True)
    assert res["is_heuristic"] is True
    assert res["model_type"] == "Astrophysical Heuristic Fallback (Uncalibrated)"
    assert res["confidence_grade"] == "Heuristic (Uncalibrated)"
    assert "Candidate (Heuristic)" in res["prediction"] or "Non-Candidate (Heuristic)" in res["prediction"]
    assert "[HEURISTIC ESTIMATION" in res["vetting_summary"]


def test_algebraic_auxiliary_derivation():
    """When flux_median and flux_min are omitted, they are scientifically derived, not artificially defaulted."""
    predictor = get_default_predictor()
    # 7 required features provided, omitting flux_median and flux_min
    features = {
        "transit_depth": 0.03,
        "period_days": 4.0,
        "duration_hours": 3.0,
        "snr": 20.0,
        "flux_std": 0.0015,
        "flux_skew": -2.0,
        "flux_kurtosis": 5.0
    }
    df_aligned = predictor._align_features(features)
    assert df_aligned["flux_median"].iloc[0] == 1.0  # normalized baseline
    assert abs(df_aligned["flux_min"].iloc[0] - (1.0 - 0.03)) < 1e-6  # baseline - depth


def test_predict_lightcurve_empty_arrays_rejected():
    """Passing empty time or flux arrays to predict_lightcurve must raise FeatureValidationError."""
    predictor = get_default_predictor()
    with pytest.raises(FeatureValidationError, match="arrays cannot be empty"):
        predictor.predict_lightcurve(np.array([]), np.array([]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
