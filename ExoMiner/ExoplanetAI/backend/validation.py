"""
validation.py - Input and Schema Validation Engine.
Ensures data integrity, type safety, physical boundary checks, and batch schema compliance.
"""

from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np


PHYSICAL_BOUNDS = {
    "transit_depth": (0.0, 1.0, "Transit depth must be between 0.0 and 1.0 (relative flux drop)."),
    "period_days": (0.1, 1000.0, "Orbital period must be between 0.1 and 1000 days."),
    "duration_hours": (0.1, 48.0, "Transit duration must be between 0.1 and 48 hours."),
    "snr": (0.0, 500.0, "Signal-to-Noise Ratio (SNR) must be non-negative."),
    "flux_std": (0.0, 1.0, "Flux standard deviation must be non-negative."),
    "flux_skew": (-20.0, 20.0, "Flux skewness must be in realistic statistical range."),
    "flux_kurtosis": (-10.0, 100.0, "Flux kurtosis must be in realistic statistical range.")
}


def validate_manual_inputs(inputs: Dict[str, float]) -> Tuple[bool, List[str]]:
    """
    Validates manual single-star input parameters against physical astrophysical constraints.
    Returns: (is_valid, list_of_error_messages)
    """
    errors = []
    for key, (min_val, max_val, msg) in PHYSICAL_BOUNDS.items():
        if key in inputs:
            val = inputs[key]
            if val is None or not np.isfinite(val):
                errors.append(f"Field '{key}' is invalid or empty.")
            elif not (min_val <= val <= max_val):
                errors.append(f"Field '{key}': {msg} (Got: {val})")

    return len(errors) == 0, errors


def validate_batch_csv(df: pd.DataFrame, max_rows: int = 10000) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validates an uploaded batch DataFrame.
    Checks:
    - Non-empty dataframe
    - Row count within limit
    - Presence of either flux time-series or standard feature set
    - Numeric data integrity
    """
    summary: Dict[str, Any] = {
        "rows": len(df),
        "columns": len(df.columns),
        "mode": "unknown",
        "missing_values": int(df.isnull().sum().sum()),
        "invalid_columns": []
    }

    if df.empty:
        return False, "Uploaded CSV file contains no data rows.", summary

    if len(df) > max_rows:
        return False, f"File exceeds maximum allowed rows ({max_rows}). Got {len(df)} rows.", summary

    # Detect mode
    flux_cols = [c for c in df.columns if str(c).startswith("FLUX.")]
    std_feature_keys = set(PHYSICAL_BOUNDS.keys())
    present_std_keys = std_feature_keys.intersection(set(df.columns))

    if len(flux_cols) >= 50:
        summary["mode"] = "flux_series"
        summary["flux_count"] = len(flux_cols)
    elif len(present_std_keys) >= 3:
        summary["mode"] = "standard_features"
        summary["matched_features"] = list(present_std_keys)
    else:
        # Check if generic numeric columns can be used
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 3:
            summary["mode"] = "generic_tabular"
            summary["numeric_columns"] = numeric_cols
        else:
            return False, (
                "CSV does not contain recognized columns. Please provide either flux columns "
                "('FLUX.1', 'FLUX.2', ...) or feature columns ('transit_depth', 'period_days', 'snr', etc.)."
            ), summary

    # Check for NaN / infinite values
    if summary["missing_values"] > 0:
        pct = (summary["missing_values"] / (len(df) * len(df.columns))) * 100
        if pct > 40.0:
            return False, f"CSV has excessive missing values ({pct:.1f}%). Please clean data before uploading.", summary

    return True, "Validation successful", summary


def validate_lightcurve_df(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validates lightcurve DataFrame containing time and flux columns."""
    cols_lower = [str(c).lower().strip() for c in df.columns]
    has_time = any("time" in c for c in cols_lower)
    has_flux = any("flux" in c for c in cols_lower)

    if not (has_time and has_flux):
        return False, "Light curve DataFrame must contain at least 'time' and 'flux' columns."

    if len(df) < 20:
        return False, f"Light curve is too short ({len(df)} points). Minimum 20 points required."

    return True, "Valid light curve"
