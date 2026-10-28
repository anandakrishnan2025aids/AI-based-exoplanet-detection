"""
feature_engineering.py - Astrophysical and Statistical Feature Engineering Engine.
Calculates high-fidelity exoplanet transit indicators from photometric observations:
Transit depth, orbital period, single-transit duration, SNR, skewness, kurtosis,
and robust photometric metrics.

Uses Box Least Squares (BLS) periodogram search and robust median/MAD statistics
to prevent outlier corruption and multi-transit summation errors.
"""

from typing import Dict, Any, Union, Optional, Tuple, List
import numpy as np
import pandas as pd
from scipy import stats
from astropy.timeseries import BoxLeastSquares


STANDARD_FEATURE_NAMES = [
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


class ExoplanetFeatureExtractor:
    """
    Extracts physically motivated transit features and higher-order moments
    from processed astronomical light curve series.
    
    Guarantees:
    - Robust transit depth estimation using median in-transit vs out-of-transit statistics.
    - Single-transit duration estimation (no multi-transit summation).
    - Box Least Squares (BLS) period search without fabricated defaults.
    - Outlier rejection (single bad cadences do not generate false transit signals).
    - Safe handling of constant flux, NaNs, infinities, and insufficient samples.
    """

    def __init__(self, baseline_flux: float = 1.0, sde_threshold: float = 6.0):
        self.baseline_flux = baseline_flux
        self.sde_threshold = sde_threshold

    def clean_series(self, time: np.ndarray, flux: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Filters non-finite points and sorts chronologically."""
        t_arr = np.asarray(time, dtype=float)
        f_arr = np.asarray(flux, dtype=float)
        mask = np.isfinite(t_arr) & np.isfinite(f_arr)
        t_clean = t_arr[mask]
        f_clean = f_arr[mask]

        if len(t_clean) > 1 and not np.all(np.diff(t_clean) >= 0):
            sort_idx = np.argsort(t_clean)
            t_clean = t_clean[sort_idx]
            f_clean = f_clean[sort_idx]

        return t_clean, f_clean

    def estimate_period_bls(
        self,
        time: np.ndarray,
        flux: np.ndarray
    ) -> Tuple[Optional[float], Optional[float], Optional[float], float, Optional[Dict[str, Any]]]:
        """
        Box Least Squares (BLS) periodogram search.
        
        Returns:
            (best_period, best_duration_days, best_t0, sde, stats_dict)
            If no reliable periodic transit is detected, returns (None, None, None, 0.0, None).
        """
        t, f = self.clean_series(time, flux)
        if len(t) < 30:
            return None, None, None, 0.0, None

        t_span = float(t[-1] - t[0])
        if t_span <= 0:
            return None, None, None, 0.0, None

        dt = float(np.median(np.diff(t)))
        if dt <= 0:
            return None, None, None, 0.0, None

        p_min = max(0.4, 4.0 * dt)
        p_max = min(50.0, t_span / 2.0)
        if p_max <= p_min:
            return None, None, None, 0.0, None

        min_dur = max(0.015, 2.0 * dt)
        max_dur = min(0.35, 0.20 * p_min)
        if min_dur >= max_dur:
            return None, None, None, 0.0, None

        durations = np.linspace(min_dur, max_dur, 10)
        n_periods = min(1000, max(200, int(t_span / dt / 4)))
        periods = np.exp(np.linspace(np.log(p_min), np.log(p_max), n_periods))

        try:
            bls = BoxLeastSquares(t, f)
            res = bls.power(periods, durations)
            best_idx = np.argmax(res.power)
            max_power = float(res.power[best_idx])
            mean_power = float(np.mean(res.power))
            std_power = float(np.std(res.power))
            sde = (max_power - mean_power) / (std_power + 1e-12)

            best_p = float(res.period[best_idx])
            best_dur = float(res.duration[best_idx])
            best_t0 = float(res.transit_time[best_idx])

            if sde >= self.sde_threshold:
                bls_stats = bls.compute_stats(best_p, best_dur, best_t0)
                # Verify that transit depth is positive (dip, not emission/flare)
                if bls_stats.get("depth", (0.0,))[0] > 0.0:
                    return best_p, best_dur, best_t0, sde, bls_stats
        except Exception:
            pass

        return None, None, None, 0.0, None

    def estimate_period(self, time: np.ndarray, flux: np.ndarray) -> Optional[float]:
        """
        Estimate dominant orbital period in days using Box Least Squares (BLS).
        Returns None if no statistically significant periodic transit is found.
        Never fabricates an unjustified default period.
        """
        best_p, _, _, sde, _ = self.estimate_period_bls(time, flux)
        return float(round(best_p, 4)) if best_p is not None else None

    def estimate_duration(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        period: Optional[float] = None
    ) -> Optional[float]:
        """
        Estimates the duration of an individual candidate transit in hours.
        Avoids summing cadences across multiple transits.
        Returns None if no candidate transit is detected.
        """
        t, f = self.clean_series(time, flux)
        if len(t) < 10:
            return None

        # Check BLS first
        if period is not None:
            best_p, best_dur, _, _, _ = self.estimate_period_bls(t, f)
            if best_dur is not None:
                return float(round(best_dur * 24.0, 3))

        # Check for contiguous dip cluster
        med_val = float(np.median(f))
        mad = 1.4826 * float(np.median(np.abs(f - med_val)))
        sigma = mad if mad > 1e-9 else float(np.std(f))
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0204

        dip_mask = f < (med_val - 2.5 * sigma)
        diffs = np.diff(np.concatenate(([0], dip_mask.astype(int), [0])))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        durations = []
        for s_idx, e_idx in zip(starts, ends):
            if (e_idx - s_idx) >= 2:
                seg_t = t[s_idx:e_idx]
                dur_h = float((seg_t[-1] - seg_t[0] + dt) * 24.0)
                durations.append(dur_h)

        if durations:
            return float(round(float(np.median(durations)), 3))

        return None

    def estimate_transit_depth(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        period: Optional[float] = None,
        duration: Optional[float] = None
    ) -> float:
        """
        Estimates transit depth using robust statistics (median in-transit vs out-of-transit).
        Rejects isolated single-cadence spikes. Returns 0.0 if no candidate transit is detected.
        """
        t, f = self.clean_series(time, flux)
        if len(t) < 10 or float(np.std(f)) < 1e-12:
            return 0.0

        best_p, best_dur, best_t0, sde, _ = self.estimate_period_bls(t, f)
        if best_p is not None and best_dur is not None:
            phase = ((t - best_t0 + best_p / 2.0) % best_p) - best_p / 2.0
            in_tr = np.abs(phase) <= (best_dur / 2.0)
            out_tr = ~in_tr
            if np.sum(in_tr) >= 2 and np.sum(out_tr) >= 5:
                f_out = float(np.median(f[out_tr]))
                f_in = float(np.median(f[in_tr]))
                return max(0.0, float(round((f_out - f_in) / (f_out if f_out != 0 else 1.0), 6)))

        # Fallback to contiguous dip cluster
        med_val = float(np.median(f))
        mad = 1.4826 * float(np.median(np.abs(f - med_val)))
        sigma = mad if mad > 1e-9 else float(np.std(f))

        dip_mask = f < (med_val - 2.5 * sigma)
        diffs = np.diff(np.concatenate(([0], dip_mask.astype(int), [0])))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        depths = []
        for s_idx, e_idx in zip(starts, ends):
            if (e_idx - s_idx) >= 2:
                seg_f = f[s_idx:e_idx]
                d = float((med_val - np.median(seg_f)) / (med_val if med_val != 0 else 1.0))
                if d > 0:
                    depths.append(d)

        if depths:
            return float(round(float(np.max(depths)), 6))

        return 0.0

    def extract_from_series(self, time: np.ndarray, flux: np.ndarray) -> Dict[str, Any]:
        """
        Extract complete exoplanet transit feature vector from time & flux arrays.
        Safely handles noise, outliers, constant flux, NaNs, and short light curves.
        """
        t, f = self.clean_series(time, flux)

        # Edge case: Insufficient data or invalid time span
        if len(t) < 10 or (t[-1] - t[0]) <= 0:
            return {
                "transit_depth": 0.0,
                "period_days": None,
                "duration_hours": None,
                "snr": 0.0,
                "flux_std": 0.0,
                "flux_skew": 0.0,
                "flux_kurtosis": 0.0,
                "flux_min": 1.0,
                "flux_median": 1.0,
                "detection_status": "insufficient_data"
            }

        std_val = float(np.std(f))
        med_val = float(np.median(f))

        # Edge case: Constant flux (zero variance)
        if std_val < 1e-12:
            return {
                "transit_depth": 0.0,
                "period_days": None,
                "duration_hours": None,
                "snr": 0.0,
                "flux_std": 0.0,
                "flux_skew": 0.0,
                "flux_kurtosis": 0.0,
                "flux_min": med_val,
                "flux_median": med_val,
                "detection_status": "constant_flux"
            }

        # Robust noise dispersion via Median Absolute Deviation (MAD)
        mad = 1.4826 * float(np.median(np.abs(f - med_val)))
        sigma = mad if mad > 1e-9 else std_val
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0204

        # Higher-order statistical moments protected against catastrophic cancellation
        skewness = float(stats.skew(f)) if len(f) > 2 else 0.0
        kurt = float(stats.kurtosis(f)) if len(f) > 3 else 0.0
        if not np.isfinite(skewness):
            skewness = 0.0
        if not np.isfinite(kurt):
            kurt = 0.0

        # Run Box Least Squares (BLS) periodogram search
        best_p, best_dur, best_t0, sde, bls_stats = self.estimate_period_bls(t, f)

        if best_p is not None and best_dur is not None:
            # Periodic transit confirmed
            phase = ((t - best_t0 + best_p / 2.0) % best_p) - best_p / 2.0
            in_tr = np.abs(phase) <= (best_dur / 2.0)
            out_tr = ~in_tr

            if np.sum(in_tr) >= 2 and np.sum(out_tr) >= 5:
                f_out = float(np.median(f[out_tr]))
                f_in = float(np.median(f[in_tr]))
                depth = max(0.0, (f_out - f_in) / (f_out if f_out != 0 else 1.0))
                dur_hours = float(best_dur * 24.0)
                snr = float((depth / sigma) * np.sqrt(np.sum(in_tr)))

                return {
                    "transit_depth": float(round(depth, 6)),
                    "period_days": float(round(best_p, 4)),
                    "duration_hours": float(round(dur_hours, 3)),
                    "snr": float(round(snr, 3)),
                    "flux_std": float(round(std_val, 6)),
                    "flux_skew": float(round(skewness, 4)),
                    "flux_kurtosis": float(round(kurt, 4)),
                    "flux_min": float(round(f_in, 6)),
                    "flux_median": float(round(med_val, 6)),
                    "detection_status": "periodic_transit_detected"
                }

        # If no periodic signal found, search for candidate contiguous dip clusters (e.g. single-transit event)
        dip_mask = f < (med_val - 2.5 * sigma)
        diffs = np.diff(np.concatenate(([0], dip_mask.astype(int), [0])))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        valid_clusters = []
        for s_idx, e_idx in zip(starts, ends):
            # Require at least 2 consecutive cadences to reject isolated single-cadence spikes
            if (e_idx - s_idx) >= 2:
                seg_f = f[s_idx:e_idx]
                seg_t = t[s_idx:e_idx]
                cluster_dur = float((seg_t[-1] - seg_t[0] + dt) * 24.0)
                cluster_depth = float((med_val - np.median(seg_f)) / (med_val if med_val != 0 else 1.0))
                if cluster_depth > 0:
                    valid_clusters.append((cluster_depth, cluster_dur, e_idx - s_idx, float(np.median(seg_f))))

        if valid_clusters:
            valid_clusters.sort(key=lambda x: x[0], reverse=True)
            rep_depth, rep_dur, n_pts, rep_min = valid_clusters[0]
            snr = float((rep_depth / sigma) * np.sqrt(n_pts))
            return {
                "transit_depth": float(round(rep_depth, 6)),
                "period_days": None,
                "duration_hours": float(round(rep_dur, 3)),
                "snr": float(round(snr, 3)),
                "flux_std": float(round(std_val, 6)),
                "flux_skew": float(round(skewness, 4)),
                "flux_kurtosis": float(round(kurt, 4)),
                "flux_min": float(round(rep_min, 6)),
                "flux_median": float(round(med_val, 6)),
                "detection_status": "single_transit_detected"
            }

        # Pure white noise or non-transit light curve: Zero transit depth
        return {
            "transit_depth": 0.0,
            "period_days": None,
            "duration_hours": None,
            "snr": 0.0,
            "flux_std": float(round(std_val, 6)),
            "flux_skew": float(round(skewness, 4)),
            "flux_kurtosis": float(round(kurt, 4)),
            "flux_min": float(round(med_val, 6)),
            "flux_median": float(round(med_val, 6)),
            "detection_status": "no_transit_detected"
        }

    def extract_from_flux_sequence(self, fluxes: np.ndarray) -> Dict[str, Any]:
        """
        Extract features when only flux sequence is provided (e.g. Kaggle FLUX.1 ... FLUX.N).
        Synthesizes uniform time cadences (30-min Kepler long cadence = 0.02043365 days).
        """
        n = len(fluxes)
        dt = 0.02043365  # Kepler long-cadence in days
        time = np.arange(n) * dt
        return self.extract_from_series(time, fluxes)


def extract_features(
    time_or_flux: Union[np.ndarray, pd.Series],
    flux: Optional[Union[np.ndarray, pd.Series]] = None
) -> Dict[str, Any]:
    """Convenience functional wrapper for feature extraction."""
    extractor = ExoplanetFeatureExtractor()
    if flux is None:
        arr = np.asarray(time_or_flux, dtype=float)
        return extractor.extract_from_flux_sequence(arr)
    else:
        t_arr = np.asarray(time_or_flux, dtype=float)
        f_arr = np.asarray(flux, dtype=float)
        return extractor.extract_from_series(t_arr, f_arr)


def batch_extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features across a batch DataFrame.
    Supports either:
    1. Flux columns (e.g., 'FLUX.1' to 'FLUX.N')
    2. Already computed features dataframe
    """
    flux_cols = [c for c in df.columns if str(c).startswith("FLUX.")]
    if flux_cols:
        extractor = ExoplanetFeatureExtractor()
        rows = []
        for _, row in df[flux_cols].iterrows():
            f_vals = row.values.astype(float)
            feats = extractor.extract_from_flux_sequence(f_vals)
            rows.append(feats)
        res_df = pd.DataFrame(rows)
        if "LABEL" in df.columns:
            res_df["LABEL"] = df["LABEL"].values
        elif "label" in df.columns:
            res_df["label"] = df["label"].values
        return res_df

    return df
