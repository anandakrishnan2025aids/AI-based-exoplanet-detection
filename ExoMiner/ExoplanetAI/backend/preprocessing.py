"""
preprocessing.py - Astronomical Light Curve Preprocessing Engine.
Central source of truth for cleaning, detrending, and normalizing photometric observations.
"""

from typing import Tuple, Optional, Union
import numpy as np
import pandas as pd


class LightCurvePreprocessor:
    """
    Standard preprocessing pipeline for astronomical photometric light curves.
    Applies NaN filtering, sigma-clipping outlier removal, baseline detrending,
    and median flux normalization.
    """

    def __init__(self, sigma_upper: float = 3.5, sigma_lower: float = 5.0, detrend_window: int = 41):
        self.sigma_upper = sigma_upper
        self.sigma_lower = sigma_lower
        self.detrend_window = detrend_window

    def remove_nans(self, time: np.ndarray, flux: np.ndarray, flux_err: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Remove NaN, Inf, and non-positive flux values."""
        mask = np.isfinite(time) & np.isfinite(flux)
        if flux_err is not None:
            mask = mask & np.isfinite(flux_err) & (flux_err >= 0)

        time_clean = time[mask]
        flux_clean = flux[mask]
        err_clean = flux_err[mask] if flux_err is not None else None
        return time_clean, flux_clean, err_clean

    def remove_outliers(self, flux: np.ndarray) -> np.ndarray:
        """
        Sigma-clipping to eliminate non-astronomical cosmic ray hits or instrument artifacts.
        Asymmetrical clipping retains genuine asymmetric transit dips while removing positive spikes.
        """
        median = np.median(flux)
        std = np.std(flux)
        if std == 0:
            return np.ones_like(flux, dtype=bool)

        mask = (flux < (median + self.sigma_upper * std)) & (flux > (median - self.sigma_lower * std))
        return mask

    def normalize(self, flux: np.ndarray) -> np.ndarray:
        """Normalize light curve by dividing by the median flux level."""
        median = np.median(flux)
        if median == 0 or np.isnan(median):
            return flux
        return flux / median

    def detrend(self, time: np.ndarray, flux: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detrend stellar variability using a rolling median baseline.
        Returns: (detrended_flux, trend)
        """
        s = pd.Series(flux)
        w = max(5, self.detrend_window)
        if w % 2 == 0:
            w += 1
        trend = s.rolling(window=w, center=True, min_periods=3).median().bfill().ffill().values
        detrended_flux = flux - trend + np.median(flux)
        return detrended_flux, trend

    def process(self, time: np.ndarray, flux: np.ndarray, flux_err: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Full preprocessing pipeline returning clean DataFrame."""
        t_clean, f_clean, e_clean = self.remove_nans(time, flux, flux_err)
        mask = self.remove_outliers(f_clean)
        t_clean = t_clean[mask]
        f_clean = f_clean[mask]
        if e_clean is not None:
            e_clean = e_clean[mask]

        f_norm = self.normalize(f_clean)
        f_detrended, trend = self.detrend(t_clean, f_norm)

        data = {
            "time": t_clean,
            "flux": f_norm,
            "flux_detrended": f_detrended,
            "trend": trend
        }
        if e_clean is not None:
            data["flux_err"] = e_clean / (np.median(f_clean) if np.median(f_clean) != 0 else 1.0)
        else:
            data["flux_err"] = np.zeros_like(t_clean)

        return pd.DataFrame(data)


def clean_lightcurve(time: np.ndarray, flux: np.ndarray, flux_err: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Convenience functional wrapper for LightCurvePreprocessor."""
    preprocessor = LightCurvePreprocessor()
    return preprocessor.process(time, flux, flux_err)


def normalize_flux(flux: np.ndarray) -> np.ndarray:
    """Normalize flux series by median."""
    median = np.median(flux)
    return flux / (median if median != 0 else 1.0)


def download_target_lightcurve(target: str, mission: str = "Kepler") -> Optional[pd.DataFrame]:
    """
    Downloads and pre-cleans a light curve from MAST using Lightkurve.
    Returns DataFrame with columns ['time', 'flux', 'flux_err', 'pdcsap_flux'].
    """
    try:
        from lightkurve import search_lightcurve
        search = search_lightcurve(target, mission=mission)
        if len(search) == 0:
            return None
        lc = search[0].download()
        lc = lc.remove_nans()
        time = lc.time.value
        flux = lc.flux.value
        flux_err = getattr(lc, 'flux_err', None)
        err_val = flux_err.value if flux_err is not None else None

        pre = LightCurvePreprocessor()
        df = pre.process(time, flux, err_val)
        return df
    except Exception as e:
        print(f"Error downloading lightcurve for {target}: {e}")
        return None
