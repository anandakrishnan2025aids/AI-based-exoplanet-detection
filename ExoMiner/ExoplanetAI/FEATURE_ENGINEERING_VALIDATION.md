# ExoplanetAI: Feature Engineering Scientific Validation Report

**Document ID:** `FEATURE_ENGINEERING_VALIDATION.md`  
**Target Repository:** `ExoMiner/ExoplanetAI`  
**Engineer:** Senior Astronomical Data Scientist & ML Engineer  
**Date:** October 27, 2026  
**Status:** Audit Completed & Corrected (34/34 Tests Passed)

---

## 1. Executive Summary

During the initial audit documented in [`MODEL_VALIDATION_AUDIT.md`](./MODEL_VALIDATION_AUDIT.md), critical mathematical and astrophysical flaws were identified in the feature extraction pipeline [`backend/feature_engineering.py`](./backend/feature_engineering.py):
1. **Transit Depth Outlier Vulnerability:** Calculated via $\delta = \text{median}(F) - \min(F)$, meaning a single negative cosmic ray spike or readout artifact caused false detections with 10%–50% depth. On pure Gaussian noise ($\sigma=0.005$), the extreme value statistic $\min(F)$ fabricated a false transit depth of $\sim 1.65\%$ and an SNR of $\sim 3.32$.
2. **Transit Duration Overcounting:** Summed every cadence below a $2\sigma$ threshold across the *entire* observation span ($N \times \Delta t_{\text{cadence}}$). A 30-day observation with 6 distinct transits of 2.4 hours each was reported as 14.4 hours (or clamped at the arbitrary 18.0h cap), confusing cumulative time-in-transit with individual transit duration.
3. **Orbital Period Guesswork:** Defaulted to $3.524$ days whenever peak dip spacing or FFT was ambiguous, inventing a synthetic exoplanet orbital period for pure noise and short light curves.
4. **Catastrophic Cancellation on Constant Flux:** Unprotected calls to `scipy.stats.skew` and `scipy.stats.kurtosis` caused precision loss warnings and runtime failures on flat data.

This engineering task repaired these vulnerabilities, replacing heuristic approximations with **Astropy Box Least Squares (BLS)** period search, **robust phase-folded and contiguous-cluster transit depth estimation**, **single-transit duration measurement**, and **deterministic non-detection vetting**.

---

## 2. Comparison of Calculation Methods

| Feature | Legacy Implementation | Corrected Scientific Implementation | Astronomical Justification |
| :--- | :--- | :--- | :--- |
| **Transit Depth** (`transit_depth`) | `(median(flux) - min(flux)) / median(flux)` | Robust phase-folded median difference: $\delta = (F_{\text{out}} - F_{\text{in}}) / F_{\text{out}}$, where $F_{\text{in}} = \text{median}(F[\|\phi\| \le T_{\text{dur}}/2])$ and $F_{\text{out}} = \text{median}(F[\|\phi\| > T_{\text{dur}}/2])$. For single transit events, uses the median of contiguous cadences ($N \ge 2$) below $2.5\sigma_{\text{MAD}}$. Returns $0.0$ if no transit is detected. | Eliminates sensitivity to single-point cosmic ray outliers. Prevents Gaussian noise from generating false positive transits. |
| **Transit Duration** (`duration_hours`) | $\sum (F < \mu - 2\sigma) \cdot dt \cdot 24$, clamped to $[0.5, 18.0]$ hours, fallback $2.4$h. | Fits the box width $T_{\text{dur}}$ of an **individual transit event** via BLS or contiguous dip cluster ($N \ge 2$). In hours: $T_{\text{dur}} \times 24.0$. Returns `None` if no transit event exists. | Accurately measures single occultation ingress-to-egress duration without accumulating across multiple orbits. |
| **Orbital Period** (`period_days`) | Dip spacing median with fallback to $3.524$ days. | **Box Least Squares (BLS)** periodogram search across $[P_{\text{min}}, P_{\text{max}}]$. Enforces Signal Detection Efficiency $\text{SDE} \ge 6.0$. Returns `None` if data is insufficient or non-periodic. | Never fabricates an orbital period when the light curve does not mathematically support a detection. |
| **Signal-to-Noise Ratio** (`snr`) | `transit_depth / flux_std` | $\text{SNR} = \frac{\delta}{\sigma_{\text{MAD}}} \times \sqrt{N_{\text{tr}}}$, where $\sigma_{\text{MAD}} = 1.4826 \times \text{median}(\|F - \text{median}(F)\|)$ and $N_{\text{tr}}$ is in-transit cadence count. | Accurately accounts for in-transit point stacking and resists outlier corruption. On pure noise ($\delta=0.0$), $\text{SNR} = 0.0$. |
| **Higher-Order Moments** (`flux_skew`, `flux_kurtosis`) | Raw `scipy.stats.skew` and `kurtosis` without variance guards. | Guarded: returns $0.0$ if $\sigma_{\text{phot}} < 10^{-12}$ or $N < 4$; sanitizes non-finite outputs. | Prevents division-by-zero crashes and catastrophic cancellation warnings on flat flux. |
| **Auxiliary Min** (`flux_min`) | Raw `np.min(flux)` | Robust in-transit flux level $F_{\text{in}} = F_{\text{med}} - \delta$ (or median of transit points). | Immune to single cosmic ray drops. |
| **Auxiliary Median** (`flux_median`) | `np.median(flux)` | `np.median(flux)` (normalized to $\sim 1.0$). | Preserved. |

---

## 3. Assumptions and Units

- **Time Standard:** Barycentric Kepler Julian Day (BKJD) or Barycentric Corrected Julian Day (BJD). Time values must be monotonically increasing floats.
- **Cadence Spacing:** Median interval $\Delta t = \text{median}(\Delta t_i)$. For raw flux sequences without time arrays (e.g. Kaggle `FLUX.1`...`FLUX.N`), standard Kepler Long Cadence is assumed: $\Delta t = 0.02043365\text{ days} \approx 29.42\text{ minutes}$.
- **Units:**
  - `transit_depth`: Relative flux drop $\Delta F / F$ (dimensionless, range $[0.0, 1.0]$).
  - `period_days`: Orbital period in Earth days ($d$).
  - `duration_hours`: Individual transit duration from first contact to fourth contact in hours ($h$).
  - `snr`: Statistical transit significance (dimensionless).
  - `flux_std`: Photometric standard deviation of normalized flux (dimensionless).
  - `flux_skew`: Fisher-Pearson coefficient of skewness (dimensionless).
  - `flux_kurtosis`: Excess kurtosis (Fisher definition where normal distribution $= 0.0$).
  - `flux_min`: Normalized in-transit minimum flux level (typically $1.0 - \delta$).
  - `flux_median`: Baseline normalized median flux ($\approx 1.0$).

---

## 4. Files Changed

| File | Changes Made |
| :--- | :--- |
| [`backend/feature_engineering.py`](./backend/feature_engineering.py) | Full overhaul of `ExoplanetFeatureExtractor`. Replaced heuristic period estimation with Astropy `BoxLeastSquares`. Implemented robust phase-folded and cluster transit depth estimation. Implemented single-event duration estimation. Added variance checks to statistical moments. Documented failure behavior when no transit or period exists. |
| [`backend/prediction.py`](./backend/prediction.py) | Added deterministic non-detection path: returns clean `Non-Exoplanet` verdict ($P=0.0$, confidence $1.0$) when feature extraction reports `no_transit_detected`. Handled single-transit events missing orbital periods gracefully (raises `ModelInferenceError` or executes explicit heuristic fallback when `allow_heuristic=True`). Sanitized `None` handling in vetting summaries. |
| [`pages/02_Detect.py`](./pages/02_Detect.py) | Safely handles `None` in `period_days` for KPI displays and phase-folded plots, preventing `TypeError` formatting exceptions. |
| [`components/report.py`](./components/report.py) | Safely formats `None` periods and durations in downloadable laboratory vetting dossiers. |
| [`tests/test_audit_scientific_correctness.py`](./tests/test_audit_scientific_correctness.py) | Updated regression assertions to verify that multiple transits measure individual duration ($\le 4.0$h, not summed $>5.0$h) and pure white noise yields zero transit depth. |
| [`tests/test_feature_engineering_scientific.py`](./tests/test_feature_engineering_scientific.py) | **New dedicated test suite** covering all 10 mandated scientific edge-case regression scenarios. |

---

## 5. Verification & Test Results

The full test suite was executed in the project's active `exoplnt_dl` environment:
`C:\Users\anand\anaconda3\envs\exoplnt_dl\python.exe -m pytest c:\Users\anand\OneDrive\ExoMiner-main\ExoMiner\ExoplanetAI\tests -v`

### Actual Execution Output:
```text
============================= test session starts =============================
platform win32 -- Python 3.11.15, pytest-9.1.1, pluggy-1.7.0 -- C:\Users\anand\anaconda3\envs\exoplnt_dl\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\anand\OneDrive\ExoMiner-main\ExoMiner\ExoplanetAI
plugins: dash-3.0.4
collecting ... collected 34 items

tests/test_audit_scientific_correctness.py::test_empty_input_rejected_with_validation_error PASSED [  2%]
tests/test_audit_scientific_correctness.py::test_multi_transit_duration_measures_single_transit PASSED [  5%]
tests/test_audit_scientific_correctness.py::test_white_noise_produces_zero_transit_depth PASSED [  8%]
tests/test_audit_scientific_correctness.py::test_model_failure_raises_model_inference_error_by_default PASSED [ 11%]
tests/test_backend.py::test_preprocessing_nan_removal PASSED             [ 14%]
tests/test_backend.py::test_preprocessing_normalization PASSED           [ 17%]
tests/test_backend.py::test_feature_extraction_keys PASSED               [ 20%]
tests/test_backend.py::test_validation_manual_inputs PASSED              [ 23%]
tests/test_backend.py::test_classification_metrics PASSED                [ 26%]
tests/test_backend.py::test_confusion_matrix PASSED                      [ 29%]
tests/test_feature_engineering_scientific.py::test_01_clean_synthetic_transit PASSED [ 32%]
tests/test_feature_engineering_scientific.py::test_02_white_noise_without_transit PASSED [ 35%]
tests/test_feature_engineering_scientific.py::test_03_transit_with_extreme_outlier PASSED [ 38%]
tests/test_feature_engineering_scientific.py::test_04_multiple_separate_transits_duration PASSED [ 41%]
tests/test_feature_engineering_scientific.py::test_05_constant_flux PASSED [ 44%]
tests/test_feature_engineering_scientific.py::test_06_insufficient_samples PASSED [ 47%]
tests/test_feature_engineering_scientific.py::test_07_irregular_or_invalid_time PASSED [ 50%]
tests/test_feature_engineering_scientific.py::test_08_missing_and_non_finite_values PASSED [ 52%]
tests/test_feature_engineering_scientific.py::test_09_flux_normalization_and_auxiliary_consistency PASSED [ 55%]
tests/test_feature_engineering_scientific.py::test_10_units_and_period_failure_behavior PASSED [ 58%]
tests/test_prediction.py::test_predictor_singleton PASSED                [ 61%]
tests/test_prediction.py::test_predict_single_exoplanet PASSED           [ 64%]
tests/test_prediction.py::test_predict_single_noise PASSED               [ 67%]
tests/test_prediction.py::test_predict_batch PASSED                      [ 70%]
tests/test_prediction.py::test_empty_dict_rejected PASSED                [ 73%]
tests/test_prediction.py::test_missing_required_features_rejected PASSED [ 76%]
tests/test_prediction.py::test_nan_values_rejected PASSED                [ 79%]
tests/test_prediction.py::test_infinite_values_rejected PASSED           [ 82%]
tests/test_prediction.py::test_non_numeric_values_rejected PASSED        [ 85%]
tests/test_prediction.py::test_physical_boundary_violations_rejected PASSED [ 88%]
tests/test_prediction.py::test_model_inference_failure_raises_model_inference_error PASSED [ 91%]
tests/test_prediction.py::test_explicit_opt_in_heuristic_mode PASSED     [ 94%]
tests/test_prediction.py::test_algebraic_auxiliary_derivation PASSED     [ 97%]
tests/test_prediction.py::test_predict_lightcurve_empty_arrays_rejected PASSED [100%]

======================= 34 passed, 12 warnings in 3.93s =======================
```
**Result:** **34 of 34 tests passed** (100% pass rate).

---

## 6. Remaining Scientific Limitations

1. **Multi-Planet Systems:**
   BLS identifies the single most prominent periodic signal in the light curve. If a system contains multiple transiting exoplanets (e.g. TRAPPIST-1 with 7 planets), iterative transit subtraction (pre-whitening) would be required to detect the secondary planets.
2. **Highly Eccentric Orbits:**
   BLS assumes a circular orbit box approximation. Planetary candidates on highly eccentric orbits ($e > 0.5$) with anomalously short or long transit durations can have reduced BLS power.
3. **Single-Transit Candidates:**
   Observations shorter than twice the orbital period cannot establish periodicity. While the pipeline now accurately measures the single transit depth and duration without inventing a period, machine learning inference cannot proceed without an orbital period feature. Multi-sector observations or radial velocity follow-up are scientifically required.

---

## 7. Model Compatibility & Retraining Assessment

### Analysis:
- The saved model `exoplanet_model.pkl` was trained with a legacy feature pipeline where non-transit noise was systematically assigned **fabricated non-zero depths** ($\sim 1.5\%$), **fabricated SNR** ($\sim 3.0$), and **fabricated default periods** ($3.524$ days).
- In our corrected pipeline, pure stellar noise now honestly yields:
  - `transit_depth = 0.0`
  - `snr = 0.0`
  - `period_days = None`
- To preserve backward compatibility without crashing or modifying weights, `ExoplanetPredictor` now includes a **deterministic vetting branch**: when feature extraction finds `no_transit_detected` (zero depth and no periodicity), it directly classifies the target as a **Non-Exoplanet with 100% confidence**, bypassing model inference.
- However, for real planetary signals with established BLS periods, the corrected transit depths and durations are mathematically shifted relative to the uncorrected training distribution (e.g., duration is now $\sim 2.4$h rather than summed $\sim 14$h).

### Final Recommendation:
> **The current Random Forest model MUST be retrained.**
> While the code is now mathematically, astrophysically, and architecturally robust, the existing `exoplanet_model.pkl` was fitted on distorted feature distributions. For production research or scientific discovery, the classifier should be retrained on genuine Kepler/TESS TCE catalogs using the corrected feature extractor.
