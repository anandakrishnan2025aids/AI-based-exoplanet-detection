"""
test_backend.py - Unit tests for ExoplanetAI backend modules.
"""

import sys
import os
import math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.preprocessing import LightCurvePreprocessor, clean_lightcurve, normalize_flux
from backend.feature_engineering import ExoplanetFeatureExtractor, extract_features, STANDARD_FEATURE_NAMES
from backend.validation import validate_manual_inputs, validate_batch_csv, validate_lightcurve_df
from backend.metrics import calculate_classification_metrics, generate_confusion_matrix_data


def test_preprocessing_nan_removal():
    time = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    flux = np.array([1.0, 0.99, 1.01, np.nan, 0.98])
    t_clean, f_clean, _ = LightCurvePreprocessor().remove_nans(time, flux)

    assert len(t_clean) == 3
    assert len(f_clean) == 3
    assert not np.any(np.isnan(t_clean))
    assert not np.any(np.isnan(f_clean))


def test_preprocessing_normalization():
    flux = np.array([100.0, 200.0, 300.0])
    f_norm = normalize_flux(flux)
    assert math.isclose(np.median(f_norm), 1.0, abs_tol=1e-5)


def test_feature_extraction_keys():
    time = np.linspace(0, 10, 200)
    flux = 1.0 - 0.02 * np.exp(-((time - 5.0) ** 2) / 0.1)  # Transit dip at t=5
    features = extract_features(time, flux)

    for col in STANDARD_FEATURE_NAMES:
        assert col in features, f"Missing feature {col}"

    assert features["transit_depth"] > 0.01
    assert features["snr"] > 0.0


def test_validation_manual_inputs():
    valid_inputs = {
        "transit_depth": 0.015,
        "period_days": 3.52,
        "duration_hours": 2.4,
        "snr": 12.0,
        "flux_std": 0.002,
        "flux_skew": -1.2,
        "flux_kurtosis": 3.5
    }
    is_valid, errors = validate_manual_inputs(valid_inputs)
    assert is_valid
    assert len(errors) == 0

    invalid_inputs = {
        "transit_depth": -0.5,  # Invalid negative
        "period_days": 5000.0   # Out of bounds
    }
    is_valid, errors = validate_manual_inputs(invalid_inputs)
    assert not is_valid
    assert len(errors) == 2


def test_classification_metrics():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 1, 0, 0]
    metrics = calculate_classification_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0


def test_confusion_matrix():
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 0, 1]
    cm = generate_confusion_matrix_data(y_true, y_pred)

    assert cm["true_positives"] == 1
    assert cm["true_negatives"] == 1
    assert cm["false_positives"] == 1
    assert cm["false_negatives"] == 1


if __name__ == "__main__":
    test_preprocessing_nan_removal()
    test_preprocessing_normalization()
    test_feature_extraction_keys()
    test_validation_manual_inputs()
    test_classification_metrics()
    test_confusion_matrix()
    print("All backend tests passed!")
