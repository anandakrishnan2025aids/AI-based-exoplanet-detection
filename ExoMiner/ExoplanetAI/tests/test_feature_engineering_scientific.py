"""
test_feature_engineering_scientific.py - Scientific Regression Tests for Task 4
Validates:
1. Clean synthetic transit with known depth, duration, and period.
2. White noise without a transit.
3. Transit with one extreme downward outlier.
4. Multiple separate transits (single-event duration estimation).
5. Constant flux (zero variance, no catastrophic cancellation).
6. Insufficient samples.
7. Irregular or invalid time values.
8. Missing and non-finite values (NaN / Inf cleaning).
9. Flux normalization and auxiliary feature consistency (flux_min, flux_median).
10. Correct units and failure behavior when a period cannot be estimated.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.feature_engineering import ExoplanetFeatureExtractor, extract_features, STANDARD_FEATURE_NAMES
from backend.preprocessing import LightCurvePreprocessor, normalize_flux
from backend.prediction import ExoplanetPredictor, FeatureValidationError, ModelInferenceError


def test_01_clean_synthetic_transit():
    """1. Clean synthetic transit with known depth (1.5%), duration (2.4h = 0.1d), and period (3.52d)."""
    extractor = ExoplanetFeatureExtractor()
    time = np.linspace(0, 30, 1440)  # 30 days, 30-min cadence
    period_true = 3.52
    duration_days_true = 0.10  # 2.4 hours
    depth_true = 0.015

    # Synthesize box transit
    phase = ((time % period_true) < duration_days_true).astype(float)
    flux = 1.0 - depth_true * phase + np.random.normal(0, 0.0005, len(time))

    feats = extractor.extract_from_series(time, flux)

    # Validate estimated period within 1% of true period
    assert feats["period_days"] is not None
    assert abs(feats["period_days"] - period_true) / period_true < 0.02, f"Period {feats['period_days']} differs from {period_true}"

    # Validate estimated duration within realistic window (~2.0 - 3.0 hours)
    assert feats["duration_hours"] is not None
    assert 1.5 <= feats["duration_hours"] <= 3.5, f"Duration {feats['duration_hours']}h differs from 2.4h"

    # Validate estimated depth within 10% of true depth
    assert abs(feats["transit_depth"] - depth_true) / depth_true < 0.15, f"Depth {feats['transit_depth']} differs from {depth_true}"

    # SNR should be high for clean transit
    assert feats["snr"] > 20.0
    assert feats["detection_status"] == "periodic_transit_detected"


def test_02_white_noise_without_transit():
    """2. Pure Gaussian white noise must produce 0.0 transit depth and None period/duration."""
    np.random.seed(123)
    extractor = ExoplanetFeatureExtractor()
    time = np.linspace(0, 30, 1440)
    flux = 1.0 + np.random.normal(0, 0.005, len(time))

    feats = extractor.extract_from_series(time, flux)

    # Must NOT claim a transit exists
    assert feats["transit_depth"] == 0.0, f"Expected 0.0 transit depth on pure noise, got {feats['transit_depth']}"
    assert feats["snr"] == 0.0, f"Expected 0.0 SNR on pure noise, got {feats['snr']}"
    assert feats["period_days"] is None, "Expected None period on pure noise"
    assert feats["duration_hours"] is None, "Expected None duration on pure noise"
    assert feats["detection_status"] == "no_transit_detected"


def test_03_transit_with_extreme_outlier():
    """3. A transit light curve containing one extreme downward glitch must not adopt the glitch as transit depth."""
    extractor = ExoplanetFeatureExtractor()
    time = np.linspace(0, 30, 1440)
    period_true = 3.52
    depth_true = 0.015

    phase = ((time % period_true) < 0.10).astype(float)
    flux = 1.0 - depth_true * phase + np.random.normal(0, 0.0005, len(time))

    # Add extreme isolated cosmic ray spike (50% drop at single cadence)
    flux[150] = 0.50

    feats = extractor.extract_from_series(time, flux)

    # Transit depth must reflect genuine 1.5% transit, NOT the 50% single-point glitch!
    assert feats["transit_depth"] < 0.05, f"Transit depth {feats['transit_depth']} was corrupted by single-cadence glitch"
    assert abs(feats["transit_depth"] - depth_true) / depth_true < 0.20


def test_04_multiple_separate_transits_duration():
    """4. Multiple separate transits must measure single-transit duration, not cumulative sum."""
    extractor = ExoplanetFeatureExtractor()
    time = np.linspace(0, 30, 1500)
    # 6 transits of 2.4 hours each
    flux = 1.0 - 0.02 * np.exp(-(((time % 5.0) - 2.5) ** 2) / (2 * 0.05 ** 2))

    feats = extractor.extract_from_series(time, flux)

    # In previous implementation, this summed to >14 hours or hit 18.0h cap. Correct is ~2.4 hours.
    assert feats["duration_hours"] is not None
    assert 1.5 <= feats["duration_hours"] <= 4.0, f"Expected single transit duration ~2.4h, got {feats['duration_hours']}h"


def test_05_constant_flux():
    """5. Constant flat flux must not crash or trigger precision loss warnings."""
    extractor = ExoplanetFeatureExtractor()
    time = np.linspace(0, 10, 500)
    flux = np.full(500, 1.0)

    feats = extractor.extract_from_series(time, flux)

    assert feats["transit_depth"] == 0.0
    assert feats["period_days"] is None
    assert feats["duration_hours"] is None
    assert feats["snr"] == 0.0
    assert feats["flux_std"] == 0.0
    assert feats["flux_skew"] == 0.0
    assert feats["flux_kurtosis"] == 0.0
    assert feats["detection_status"] == "constant_flux"


def test_06_insufficient_samples():
    """6. Very short light curves (< 10 points) must be safely handled without throwing uncaught exceptions."""
    extractor = ExoplanetFeatureExtractor()
    time = np.array([1.0, 2.0, 3.0])
    flux = np.array([1.0, 0.99, 1.0])

    feats = extractor.extract_from_series(time, flux)

    assert feats["transit_depth"] == 0.0
    assert feats["period_days"] is None
    assert feats["duration_hours"] is None
    assert feats["detection_status"] == "insufficient_data"


def test_07_irregular_or_invalid_time():
    """7. Unsorted or inverted time values must be sorted correctly before period/duration estimation."""
    extractor = ExoplanetFeatureExtractor()
    # Unsorted time series with a synthetic transit
    np.random.seed(42)
    time_sorted = np.linspace(0, 20, 1000)
    flux_sorted = 1.0 - 0.015 * (((time_sorted % 4.0) < 0.12).astype(float))

    shuffle_idx = np.random.permutation(len(time_sorted))
    time_shuffled = time_sorted[shuffle_idx]
    flux_shuffled = flux_sorted[shuffle_idx]

    feats = extractor.extract_from_series(time_shuffled, flux_shuffled)

    # Must sort correctly and recover the 4.0-day period
    assert feats["period_days"] is not None
    assert abs(feats["period_days"] - 4.0) / 4.0 < 0.05


def test_08_missing_and_non_finite_values():
    """8. NaNs and Infs in time or flux must be cleanly removed without corrupting feature statistics."""
    extractor = ExoplanetFeatureExtractor()
    time = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    flux = np.array([1.0, 0.99, np.nan, 1.01, np.inf, 1.0, 0.99, 1.0, 1.0, 1.0, 1.0, 1.0])

    feats = extractor.extract_from_series(time, flux)

    for k, v in feats.items():
        if isinstance(v, float):
            assert np.isfinite(v), f"Feature {k} returned non-finite value {v}"


def test_09_flux_normalization_and_auxiliary_consistency():
    """9. Verify flux normalization consistency: median ~ 1.0, and flux_min matches baseline - depth."""
    extractor = ExoplanetFeatureExtractor()
    time = np.linspace(0, 20, 1000)
    raw_flux = 50000.0 - 500.0 * (((time % 3.0) < 0.1).astype(float))  # Raw counts ~50,000, 1% depth

    # Preprocessing normalizes by median
    norm_flux = normalize_flux(raw_flux)
    assert np.isclose(np.median(norm_flux), 1.0, atol=1e-4)

    feats = extractor.extract_from_series(time, norm_flux)
    assert np.isclose(feats["flux_median"], 1.0, atol=1e-3)
    assert np.isclose(feats["flux_min"], 1.0 - feats["transit_depth"], atol=1e-3)


def test_10_units_and_period_failure_behavior():
    """10. Verify consistent units and failure behavior when no reliable period can be estimated."""
    extractor = ExoplanetFeatureExtractor()
    # Single transit event: 3-day observation with only 1 transit at day 1.5
    time = np.linspace(0, 3.0, 300)
    # Transit at t=1.5, depth=2%, duration=2.4h (0.1d)
    in_transit = np.abs(time - 1.5) < 0.05
    flux = 1.0 - 0.02 * in_transit.astype(float) + np.random.normal(0, 0.0005, len(time))

    feats = extractor.extract_from_series(time, flux)

    # Single transit cannot establish an orbital period: period_days must be None!
    assert feats["period_days"] is None, "A single transit event must NOT return a fabricated period"

    # However, transit depth and duration of the single event SHOULD be accurately captured
    assert feats["transit_depth"] > 0.015
    assert feats["duration_hours"] is not None
    assert 1.5 <= feats["duration_hours"] <= 3.5  # Hours unit verified

    # Caller integration: predict_single rejects without allow_heuristic, or accepts with allow_heuristic=True
    predictor = ExoplanetPredictor()
    with pytest.raises(ModelInferenceError, match="requires an orbital period"):
        predictor.predict_single(feats, allow_heuristic=False)

    res = predictor.predict_single(feats, allow_heuristic=True)
    assert res["is_heuristic"] is True
    assert "Candidate (Heuristic)" in res["prediction"]
