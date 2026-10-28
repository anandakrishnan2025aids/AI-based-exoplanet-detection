"""
test_audit_scientific_correctness.py - Regression tests for Task 3 & 4 fixes
Validating input validation rejection, explicit failure signaling, robust depth, and single-transit duration.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.prediction import ExoplanetPredictor, FeatureValidationError, ModelInferenceError
from backend.feature_engineering import ExoplanetFeatureExtractor
from backend.preprocessing import LightCurvePreprocessor


def test_empty_input_rejected_with_validation_error():
    """Task 3 Fix: Verifies that passing an empty dictionary raises FeatureValidationError instead of synthesizing transit features."""
    predictor = ExoplanetPredictor()
    with pytest.raises(FeatureValidationError, match="Input feature dictionary cannot be empty"):
        predictor._align_features({})

    with pytest.raises(FeatureValidationError):
        predictor.predict_single({})


def test_multi_transit_duration_measures_single_transit():
    """Task 4 Fix: Verifies that multiple transits measure the duration of an individual transit (~2.4h), not their sum."""
    extractor = ExoplanetFeatureExtractor()
    time = np.linspace(0, 30, 1500)  # 30 days
    # 6 transits with duration ~ 2.4 hours each
    flux = 1.0 - 0.02 * np.exp(-(((time % 5.0) - 2.5) ** 2) / (2 * 0.05 ** 2))

    feats = extractor.extract_from_series(time, flux)
    # The true single-transit duration is ~2.4h. The corrected method must NOT sum across transits to >5.0h.
    assert feats["duration_hours"] is not None
    assert 1.0 <= feats["duration_hours"] <= 4.0, f"Duration {feats['duration_hours']}h should be ~2.4h, not summed"


def test_white_noise_produces_zero_transit_depth():
    """Task 4 Fix: Verifies that robust feature extraction does not fabricate a transit signal on pure Gaussian noise."""
    np.random.seed(42)
    extractor = ExoplanetFeatureExtractor()
    time = np.linspace(0, 10, 500)
    flux = 1.0 + np.random.normal(0, 0.005, 500)

    feats = extractor.extract_from_series(time, flux)
    assert feats["transit_depth"] == 0.0, f"Pure white noise should produce 0.0 transit depth, got {feats['transit_depth']}"
    assert feats["snr"] == 0.0, f"Pure white noise should produce 0.0 SNR, got {feats['snr']}"
    assert feats["period_days"] is None, "Pure white noise should not produce an orbital period"


def test_model_failure_raises_model_inference_error_by_default():
    """Task 3 Fix: Verifies that when model is unavailable, predict_single raises ModelInferenceError instead of silent fallback."""
    predictor = ExoplanetPredictor()
    # Force model to None to simulate failure
    predictor.model = None

    valid_feats = {
        "transit_depth": 0.01,
        "period_days": 3.52,
        "duration_hours": 2.4,
        "snr": 10.0,
        "flux_std": 0.001,
        "flux_skew": -1.0,
        "flux_kurtosis": 3.0
    }

    # Must raise ModelInferenceError by default
    with pytest.raises(ModelInferenceError, match="model is not loaded"):
        predictor.predict_single(valid_feats, allow_heuristic=False)

    # Opt-in heuristic mode explicitly labels output as heuristic
    res = predictor.predict_single(valid_feats, allow_heuristic=True)
    assert res["is_heuristic"] is True
    assert "Heuristic" in res["confidence_grade"]
    assert "Heuristic" in res["model_type"]
    assert "[HEURISTIC ESTIMATION" in res["vetting_summary"]
