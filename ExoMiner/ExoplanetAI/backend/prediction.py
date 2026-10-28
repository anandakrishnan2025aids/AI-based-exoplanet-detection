"""
prediction.py - Model Inference and Candidate Vetting Engine.
Central source of truth for exoplanet predictions, confidence grading,
input validation, and batch classification.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import os
import json
import joblib
import numpy as np
import pandas as pd

from .feature_engineering import ExoplanetFeatureExtractor, STANDARD_FEATURE_NAMES, extract_features
from .preprocessing import LightCurvePreprocessor


DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "exoplanet_model.pkl")
DEFAULT_METADATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "model_metadata.json")

# Mandatory physical features required for transit classification
REQUIRED_PHYSICAL_FEATURES: List[str] = [
    "transit_depth",
    "period_days",
    "duration_hours",
    "snr",
    "flux_std",
    "flux_skew",
    "flux_kurtosis"
]


class FeatureValidationError(ValueError):
    """Raised when input feature dictionary fails physical or structural validation."""
    pass


class ModelInferenceError(RuntimeError):
    """Raised when model loading or inference fails and heuristic fallback is not permitted."""
    pass


class ExoplanetPredictor:
    """
    Production-grade predictor for exoplanet candidates.
    Supports single feature dicts, raw lightcurves, and batch DataFrames.
    Guarantees strict input validation and explicit failure signaling.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, metadata_path: str = DEFAULT_METADATA_PATH):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = None
        self.metadata: Dict[str, Any] = {}
        self.feature_names: List[str] = STANDARD_FEATURE_NAMES
        self.preprocessor = LightCurvePreprocessor()
        self.extractor = ExoplanetFeatureExtractor()
        self._load_model()

    def _load_model(self):
        """Loads trained model and metadata."""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                # Check model's expected features if stored in model
                if hasattr(self.model, "feature_names_in_"):
                    self.feature_names = list(self.model.feature_names_in_)
            except Exception as e:
                print(f"Warning: Failed to load model from {self.model_path}: {e}")

        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r") as f:
                    self.metadata = json.load(f)
                    if "features" in self.metadata:
                        self.feature_names = self.metadata["features"]
            except Exception as e:
                print(f"Warning: Failed to load metadata from {self.metadata_path}: {e}")

    def _align_features(self, feature_dict: Dict[str, Any]) -> pd.DataFrame:
        """
        Validates and aligns incoming feature dictionary with expected model input schema.

        Validation rules:
        - Rejects empty, non-dict, or None inputs.
        - Rejects missing required physical features (no silent imputation of fake astronomical signals).
        - Rejects NaN, infinite, or non-numeric values.
        - Enforces physical boundaries (e.g., non-negative depth, strictly positive period/duration).
        - Standardizes auxiliary features (flux_median and flux_min derived algebraically if omitted).
        """
        if not isinstance(feature_dict, dict) or len(feature_dict) == 0:
            raise FeatureValidationError("Input feature dictionary cannot be empty or non-dict.")

        # Check for missing required physical features
        missing_required = [f for f in REQUIRED_PHYSICAL_FEATURES if f not in feature_dict]
        if missing_required:
            raise FeatureValidationError(
                f"Missing required astrophysical feature(s): {missing_required}. "
                f"Input must contain all primary features: {REQUIRED_PHYSICAL_FEATURES}."
            )

        row: Dict[str, float] = {}

        # Validate required physical features
        for key in REQUIRED_PHYSICAL_FEATURES:
            raw_val = feature_dict[key]
            try:
                val = float(raw_val)
            except (ValueError, TypeError):
                raise FeatureValidationError(
                    f"Feature '{key}' must be a numeric value, got {type(raw_val).__name__} ({raw_val!r})."
                )

            if not np.isfinite(val):
                raise FeatureValidationError(
                    f"Feature '{key}' has invalid non-finite value ({val}). NaN and Inf are rejected."
                )

            # Astrophysical sanity bounds
            if key == "transit_depth":
                if val < 0.0 or val > 1.0:
                    raise FeatureValidationError(
                        f"Physical constraint violated: 'transit_depth' must be between 0.0 and 1.0, got {val}."
                    )
            elif key in ("period_days", "duration_hours"):
                if val <= 0.0:
                    raise FeatureValidationError(
                        f"Physical constraint violated: '{key}' must be strictly positive, got {val}."
                    )
            elif key == "snr":
                if val < 0.0:
                    raise FeatureValidationError(
                        f"Physical constraint violated: 'snr' cannot be negative, got {val}."
                    )
            elif key == "flux_std":
                if val < 0.0:
                    raise FeatureValidationError(
                        f"Physical constraint violated: 'flux_std' cannot be negative, got {val}."
                    )
                if val > 1.0:
                    raise FeatureValidationError(
                        f"Incompatible feature input: 'flux_std' ({val}) exceeds physical limits for normalized light curves (expected 0.0 <= flux_std <= 1.0). "
                        "Raw or zero-centered counts must be normalized before model inference."
                    )

            row[key] = val

        # Handle auxiliary features expected by the trained Random Forest model
        # 1. flux_median: For normalized Kepler/TESS curves, baseline median is ~1.0
        if "flux_median" in feature_dict:
            raw_val = feature_dict["flux_median"]
            try:
                val = float(raw_val)
                if not np.isfinite(val):
                    raise FeatureValidationError("Feature 'flux_median' must be finite.")
                if val <= 0.0 or abs(val - 1.0) > 0.5:
                    raise FeatureValidationError(
                        f"Incompatible feature input: 'flux_median' ({val}) is not normalized around 1.0 (expected 0.5 <= flux_median <= 1.5). "
                        "Zero-centered difference flux datasets like exoTest.csv must be normalized before model inference."
                    )
                row["flux_median"] = val
            except (ValueError, TypeError) as e:
                if isinstance(e, FeatureValidationError):
                    raise
                raise FeatureValidationError(f"Invalid non-numeric 'flux_median': {raw_val!r}")
        else:
            row["flux_median"] = 1.0

        # 2. flux_min: In a normalized transit, flux_min = baseline_median - transit_depth
        if "flux_min" in feature_dict:
            raw_val = feature_dict["flux_min"]
            try:
                val = float(raw_val)
                if not np.isfinite(val):
                    raise FeatureValidationError("Feature 'flux_min' must be finite.")
                if val < -1.0 or val > 1.5:
                    raise FeatureValidationError(
                        f"Incompatible feature input: 'flux_min' ({val}) is outside physical normalized range (expected -1.0 <= flux_min <= 1.5)."
                    )
                row["flux_min"] = val
            except (ValueError, TypeError) as e:
                if isinstance(e, FeatureValidationError):
                    raise
                raise FeatureValidationError(f"Invalid non-numeric 'flux_min': {raw_val!r}")
        else:
            row["flux_min"] = float(row["flux_median"] - row["transit_depth"])

        # Ensure exact column schema matching trained model
        expected_cols = self.feature_names if self.feature_names else STANDARD_FEATURE_NAMES
        for col in expected_cols:
            if col not in row:
                raise FeatureValidationError(f"Missing schema column '{col}' expected by model.")

        return pd.DataFrame([row], columns=expected_cols)

    def predict_single(self, features: Dict[str, Any], allow_heuristic: bool = False) -> Dict[str, Any]:
        """
        Runs inference on a single star's feature dictionary.

        Args:
            features: Dictionary containing required astronomical features.
            allow_heuristic: If True, permits astrophysical heuristic calculation
                             when the machine learning model is unavailable or inference fails.
                             Defaults to False to prevent silent uncalibrated predictions.

        Returns:
            dict containing:
                prediction: "Exoplanet" | "Non-Exoplanet" (or "Candidate (Heuristic)" if fallback)
                is_exoplanet: bool
                confidence: float (0.0 - 1.0)
                probability_exoplanet: float
                probability_non_exoplanet: float
                confidence_grade: "High" | "Moderate" | "Low" | "Heuristic (Uncalibrated)"
                model_type: "Random Forest" | "Astrophysical Heuristic Fallback (Uncalibrated)"
                is_heuristic: bool
                features_used: dict
                vetting_summary: str

        Raises:
            FeatureValidationError: If input features fail schema or physical constraints.
            ModelInferenceError: If model is not loaded or inference raises an exception,
                                 and allow_heuristic is False.
        """
        if not isinstance(features, dict) or len(features) == 0:
            raise FeatureValidationError("Input feature dictionary cannot be empty or non-dict.")

        # Case A: Insufficient data or invalid / constant flux
        if features.get("detection_status") in ("insufficient_data", "constant_flux"):
            return {
                "prediction": "Unable to evaluate — insufficient or invalid data",
                "is_exoplanet": False,
                "confidence": 0.0,
                "probability_exoplanet": 0.0,
                "probability_non_exoplanet": 0.0,
                "confidence_grade": "Inconclusive",
                "model_type": "Data Quality Vetting",
                "is_heuristic": False,
                "features_used": {k: float(v) for k, v in features.items() if isinstance(v, (int, float)) and v is not None and np.isfinite(v)},
                "vetting_summary": "Photometric observation contains insufficient samples (<10 cadences) or constant zero-variance flux. Signal cannot be evaluated."
            }

        # Case B: No transit detected in light curve
        if features.get("detection_status") == "no_transit_detected" or (
            features.get("transit_depth") == 0.0 and features.get("period_days") is None and "detection_status" in features
        ):
            return {
                "prediction": "No transit detected — inconclusive",
                "is_exoplanet": False,
                "confidence": 0.0,
                "probability_exoplanet": 0.0,
                "probability_non_exoplanet": 0.0,
                "confidence_grade": "Inconclusive",
                "model_type": "Deterministic Vetting (No Transit Detected)",
                "is_heuristic": False,
                "features_used": {k: float(v) for k, v in features.items() if isinstance(v, (int, float)) and v is not None and np.isfinite(v)},
                "vetting_summary": "Photometric light curve does not exhibit statistically significant transit dips or periodic occultation within the observation baseline. The absence of a detected transit does not prove the absence of a planet (non-transiting orbital inclination, long period, or signal below noise threshold). Classified as inconclusive."
            }

        # Safe type check for transit_depth
        raw_depth = features.get("transit_depth")
        is_positive_depth = isinstance(raw_depth, (int, float)) and np.isfinite(raw_depth) and raw_depth > 0.0

        # Single transit event where periodicity could not be determined
        if is_positive_depth and features.get("period_days") is None:
            if not allow_heuristic:
                raise ModelInferenceError(
                    "Random Forest model requires an orbital period ('period_days'), but no recurring periodicity "
                    "could be estimated from the single observation baseline. Multi-sector observation or "
                    "radial velocity follow-up is required. To evaluate candidate depth with astrophysical heuristics, "
                    "specify allow_heuristic=True."
                )
            prob_exo = self._astrophysical_heuristic(features)
            prob_non = 1.0 - prob_exo
            is_exo = bool(prob_exo >= 0.5)
            conf = prob_exo if is_exo else prob_non
            summary = self._generate_summary(features, is_exo, conf, is_heuristic=True)
            return {
                "prediction": "Candidate (Heuristic)" if is_exo else "Non-Candidate (Heuristic)",
                "is_exoplanet": is_exo,
                "confidence": float(round(conf, 4)),
                "probability_exoplanet": float(round(prob_exo, 4)),
                "probability_non_exoplanet": float(round(prob_non, 4)),
                "confidence_grade": "Heuristic (Uncalibrated)",
                "model_type": "Astrophysical Heuristic Fallback (Uncalibrated)",
                "is_heuristic": True,
                "features_used": {k: float(v) for k, v in features.items() if isinstance(v, (int, float)) and v is not None and np.isfinite(v)},
                "vetting_summary": summary
            }

        df_input = self._align_features(features)

        if self.model is None:
            if not allow_heuristic:
                raise ModelInferenceError(
                    "Exoplanet prediction model is not loaded and allow_heuristic is False. "
                    "Cannot execute machine learning inference."
                )
            # Explicitly opted-in heuristic execution
            prob_exo = self._astrophysical_heuristic(features)
            prob_non = 1.0 - prob_exo
            is_exo = bool(prob_exo >= 0.5)
            conf = prob_exo if is_exo else prob_non
            summary = self._generate_summary(features, is_exo, conf, is_heuristic=True)
            return {
                "prediction": "Candidate (Heuristic)" if is_exo else "Non-Candidate (Heuristic)",
                "is_exoplanet": is_exo,
                "confidence": float(round(conf, 4)),
                "probability_exoplanet": float(round(prob_exo, 4)),
                "probability_non_exoplanet": float(round(prob_non, 4)),
                "confidence_grade": "Heuristic (Uncalibrated)",
                "model_type": "Astrophysical Heuristic Fallback (Uncalibrated)",
                "is_heuristic": True,
                "features_used": {k: float(v) for k, v in features.items() if isinstance(v, (int, float))},
                "vetting_summary": summary
            }

        try:
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(df_input)[0]
                # Dynamically locate positive-class (1) index to avoid assumption about class ordering
                if hasattr(self.model, "classes_"):
                    classes = list(self.model.classes_)
                    pos_idx = classes.index(1) if 1 in classes else (classes.index("Exoplanet") if "Exoplanet" in classes else 1)
                    neg_idx = 1 - pos_idx if len(classes) == 2 else (classes.index(0) if 0 in classes else 0)
                else:
                    pos_idx = 1
                    neg_idx = 0
                prob_exo = float(probs[pos_idx]) if len(probs) > pos_idx else float(probs[0])
                prob_non = float(probs[neg_idx]) if len(probs) > neg_idx else 1.0 - prob_exo
                pred_class = int(prob_exo >= 0.5)
            else:
                pred_class = int(self.model.predict(df_input)[0])
                prob_exo = 0.92 if pred_class == 1 else 0.08
                prob_non = 1.0 - prob_exo
        except Exception as e:
            if not allow_heuristic:
                raise ModelInferenceError(f"Model inference failed: {e}") from e
            # Caller explicitly allowed heuristic on model failure
            prob_exo = self._astrophysical_heuristic(features)
            prob_non = 1.0 - prob_exo
            is_exo = bool(prob_exo >= 0.5)
            conf = prob_exo if is_exo else prob_non
            summary = self._generate_summary(features, is_exo, conf, is_heuristic=True)
            return {
                "prediction": "Candidate (Heuristic)" if is_exo else "Non-Candidate (Heuristic)",
                "is_exoplanet": is_exo,
                "confidence": float(round(conf, 4)),
                "probability_exoplanet": float(round(prob_exo, 4)),
                "probability_non_exoplanet": float(round(prob_non, 4)),
                "confidence_grade": "Heuristic (Uncalibrated)",
                "model_type": "Astrophysical Heuristic Fallback (Uncalibrated)",
                "is_heuristic": True,
                "features_used": {k: float(v) for k, v in features.items() if isinstance(v, (int, float))},
                "vetting_summary": summary
            }

        is_exo = bool(pred_class == 1)
        confidence = prob_exo if is_exo else prob_non

        if confidence >= 0.85:
            grade = "High"
        elif confidence >= 0.65:
            grade = "Moderate"
        else:
            grade = "Low"

        summary = self._generate_summary(features, is_exo, confidence, is_heuristic=False)
        pred_label = "Transit detected — candidate for further analysis" if is_exo else "No transit detected — inconclusive"

        return {
            "prediction": pred_label,
            "is_exoplanet": is_exo,
            "confidence": float(round(confidence, 4)),
            "probability_exoplanet": float(round(prob_exo, 4)),
            "probability_non_exoplanet": float(round(prob_non, 4)),
            "confidence_grade": grade,
            "model_type": "Random Forest",
            "is_heuristic": False,
            "features_used": {k: float(v) for k, v in features.items() if isinstance(v, (int, float))},
            "vetting_summary": summary
        }

    def predict_lightcurve(self, time: np.ndarray, flux: np.ndarray, allow_heuristic: bool = False) -> Dict[str, Any]:
        """
        Full end-to-end inference from raw light curve arrays.
        Cleans, detrends, extracts features, and runs model prediction.
        """
        if len(time) == 0 or len(flux) == 0:
            raise FeatureValidationError("Light curve time and flux arrays cannot be empty.")
        if len(time) != len(flux):
            raise FeatureValidationError(f"Time and flux length mismatch: {len(time)} vs {len(flux)}.")

        df_processed = self.preprocessor.process(time, flux)
        t_clean = df_processed["time"].values
        f_clean = df_processed["flux_detrended"].values

        features = self.extractor.extract_from_series(t_clean, f_clean)
        res = self.predict_single(features, allow_heuristic=allow_heuristic)
        res["preprocessed_data"] = df_processed
        return res

    def predict_batch(self, df: pd.DataFrame, allow_heuristic: bool = False) -> pd.DataFrame:
        """
        Performs batch inference on an uploaded DataFrame.
        Supports both pre-computed feature tables and raw flux sequences.
        """
        if df.empty:
            raise FeatureValidationError("Batch DataFrame is empty.")

        out_df = df.copy()

        # Case 1: Flux series columns ('FLUX.1', 'FLUX.2', ...)
        flux_cols = [c for c in df.columns if str(c).startswith("FLUX.")]
        if flux_cols:
            predictions = []
            confidences = []
            prob_exos = []
            model_types = []
            is_heuristics = []
            for _, row in df[flux_cols].iterrows():
                f_vals = row.values.astype(float)
                feats = self.extractor.extract_from_flux_sequence(f_vals)
                pred_res = self.predict_single(feats, allow_heuristic=allow_heuristic)
                predictions.append(pred_res["prediction"])
                confidences.append(pred_res["confidence"])
                prob_exos.append(pred_res["probability_exoplanet"])
                model_types.append(pred_res.get("model_type", "Random Forest"))
                is_heuristics.append(pred_res.get("is_heuristic", False))

            out_df["prediction"] = predictions
            out_df["confidence"] = confidences
            out_df["prob_exoplanet"] = prob_exos
            out_df["model_type"] = model_types
            out_df["is_heuristic"] = is_heuristics
            return out_df

        # Case 2: Tabular feature columns
        predictions = []
        confidences = []
        prob_exos = []
        model_types = []
        is_heuristics = []
        for idx, row in df.iterrows():
            feats = row.to_dict()
            pred_res = self.predict_single(feats, allow_heuristic=allow_heuristic)
            predictions.append(pred_res["prediction"])
            confidences.append(pred_res["confidence"])
            prob_exos.append(pred_res["probability_exoplanet"])
            model_types.append(pred_res.get("model_type", "Random Forest"))
            is_heuristics.append(pred_res.get("is_heuristic", False))

        out_df["prediction"] = predictions
        out_df["confidence"] = confidences
        out_df["prob_exoplanet"] = prob_exos
        out_df["model_type"] = model_types
        out_df["is_heuristic"] = is_heuristics
        return out_df

    def _astrophysical_heuristic(self, features: Dict[str, Any]) -> float:
        """
        Astrophysical heuristic fallback based on Kepler transit physics:
        - Significant depth (> 0.002) with high SNR (> 5.0) and negative skewness indicates transit.
        """
        depth = float(features.get("transit_depth") or 0.0)
        snr = float(features.get("snr") or 0.0)
        skew = float(features.get("flux_skew") or 0.0)

        score = 0.1
        if depth > 0.001:
            score += 0.2
        if depth > 0.005:
            score += 0.25
        if snr > 4.0:
            score += 0.2
        if snr > 8.0:
            score += 0.15
        if skew < -0.3:
            score += 0.15

        return min(0.98, max(0.02, score))

    def _generate_summary(self, features: Dict[str, Any], is_exo: bool, conf: float, is_heuristic: bool = False) -> str:
        """Generates scientific text summary of findings."""
        depth_val = float(features.get("transit_depth") or 0.0)
        depth_pct = depth_val * 100.0
        snr_val = float(features.get("snr") or 0.0)
        period_val = features.get("period_days")

        period_desc = f"Estimated orbital period is {period_val:.2f} days." if period_val is not None else "Orbital period undetermined from single observation baseline."

        prefix = "[HEURISTIC ESTIMATION - UNCALIBRATED] " if is_heuristic else ""
        if is_exo:
            label = "Candidate (Heuristic)" if is_heuristic else "Transit detected — candidate for further analysis"
            return (
                f"{prefix}Candidate exhibits characteristic transit signature with a depth of {depth_pct:.3f}% "
                f"and SNR of {snr_val:.1f}. {period_desc} "
                f"Classified as {label} with {conf * 100:.1f}% confidence."
            )
        else:
            label = "Non-Candidate (Heuristic)" if is_heuristic else "No transit detected — inconclusive"
            return (
                f"{prefix}Photometric light curve does not exhibit statistically significant transit dips. "
                f"Observed depth ({depth_pct:.3f}%) or SNR ({snr_val:.1f}) is consistent with stellar noise or inconclusive. "
                f"Classified as {label} with {conf * 100:.1f}% confidence."
            )


# Global singleton cache
_PREDICTOR_INSTANCE: Optional[ExoplanetPredictor] = None

def get_default_predictor() -> ExoplanetPredictor:
    """Returns singleton instance of ExoplanetPredictor."""
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None:
        _PREDICTOR_INSTANCE = ExoplanetPredictor()
    return _PREDICTOR_INSTANCE
