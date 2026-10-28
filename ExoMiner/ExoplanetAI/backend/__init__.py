"""
ExoplanetAI Backend Package.
Unified scientific and machine learning pipeline for exoplanet transit detection.
"""

from .preprocessing import LightCurvePreprocessor, clean_lightcurve, normalize_flux, download_target_lightcurve
from .feature_engineering import ExoplanetFeatureExtractor, extract_features, batch_extract_features, STANDARD_FEATURE_NAMES
from .prediction import ExoplanetPredictor, get_default_predictor, FeatureValidationError, ModelInferenceError
from .validation import validate_manual_inputs, validate_batch_csv, validate_lightcurve_df
from .metrics import calculate_classification_metrics, generate_confusion_matrix_data, compute_batch_summary

__all__ = [
    "LightCurvePreprocessor",
    "clean_lightcurve",
    "normalize_flux",
    "download_target_lightcurve",
    "ExoplanetFeatureExtractor",
    "extract_features",
    "batch_extract_features",
    "STANDARD_FEATURE_NAMES",
    "ExoplanetPredictor",
    "get_default_predictor",
    "FeatureValidationError",
    "ModelInferenceError",
    "validate_manual_inputs",
    "validate_batch_csv",
    "validate_lightcurve_df",
    "calculate_classification_metrics",
    "generate_confusion_matrix_data",
    "compute_batch_summary",
]
