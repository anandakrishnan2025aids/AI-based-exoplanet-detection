# MODEL COMPATIBILITY AND SCIENTIFIC PREDICTION SEMANTICS AUDIT
**ExoMiner / ExoplanetAI System**  
**Role:** Senior Machine Learning Engineer & Exoplanet Data Scientist  
**Date:** October 2026  
**Status:** Audit Complete — Rigorously Verified & Documented  
**Scope:** `ExoMiner/ExoplanetAI` (Random Forest Classifier, Feature Engineering Pipeline, Evaluation Benchmark, Prediction Semantics)

---

## 1. Executive Summary

This audit investigates the root causes behind the benchmark evaluation results reported for the saved Random Forest model (`models/exoplanet_model.pkl`), where the model achieved **99.12% accuracy but 0.0% recall** across the 570 light curves in `data/training/exoTest.csv` (failing to identify all 5 confirmed exoplanets).

### Core Findings
1. **Feature Distribution & Scale Mismatch (Out-of-Distribution Inputs):**  
   The benchmark evaluation on `exoTest.csv` was **not a valid test of model classification capacity**. `exoTest.csv` consists of Kaggle pre-whitened difference flux centered at $0.0$ with raw cadence values between $-16,000$ and $+10,000$. Conversely, `exoplanet_model.pkl` was trained on normalized relative flux features where baseline flux $F_0 \equiv 1.0$, `flux_std` $\sim 0.0035$, and `transit_depth` $\sim 0.004$. Passing difference flux into feature extractors generated input values $10,000\times$ to $500,000\times$ larger than the model's physical domain, causing decision trees to partition all test samples down non-exoplanet branches.
2. **Missing Training Script & Data Provenance:**  
   Inspection of the decision tree estimators inside `exoplanet_model.pkl` reveals bootstrap sample sizes of $\sim 758$ unique records drawn from an effective dataset of $N \approx 1,200$ samples with $\approx 41\%$ positive class (exoplanets) and $\approx 59\%$ negative class. The repository contains **no training script** for this model, and the model was **never trained on `exoTrain.csv`** (which has 5,087 samples and only 0.73% exoplanets).
3. **Flawed Prior Non-Detection Semantics:**  
   The previous pipeline automatically assigned light curves with no detectable transit dip to `"Non-Exoplanet"` with **100% confidence**. In observational astrophysics, absence of a detected transit is never evidence that a star hosts no exoplanet (due to geometric inclination, baseline limitations, and instrument noise floors). This has been corrected to an honest, three-tiered scientific classification.
4. **Resolution & Preservation:**  
   The historical evaluation metrics (99.12% accuracy, 0% recall, 0.643 ROC-AUC) have been preserved as historical diagnostic measurements of an incompatible feature pipeline, accompanied by explicit transparency disclosures. The codebase has been updated with rigorous physical boundary validation, dynamic class indexing, and 13 new regression tests (57/57 tests passing).

---

## 2. Feature Pipeline Audit

### 2.1 Target Prediction Semantics
The Random Forest classifier (`exoplanet_model.pkl`) is a binary estimator:
- **Class 0:** Non-Exoplanet / False Positive / Astrophysical Noise.
- **Class 1:** Confirmed Exoplanet Candidate.
- **Model Architecture:** `sklearn.ensemble.RandomForestClassifier` with 100 decision trees, max depth = 12, max features = $\sqrt{9} = 3$.

### 2.2 Expected Nine Features (Exact Order, Units, and Scales)
The model strictly expects a 9-dimensional vector in the following order:

| Index | Feature Name | Physical Meaning & Units | Expected Scale (Normalized Flux $F_0 = 1.0$) | Decision Tree Split Thresholds |
|---|---|---|---|---|
| 0 | `transit_depth` | Relative flux decrement during transit ($\Delta F / F_0$, dimensionless) | $0.0001 \le \delta \le 0.05$ | Primary split at $0.00408$; splits up to $0.045$ |
| 1 | `transit_duration` | Fractional transit duration ($T_{\text{dur}} / P$, dimensionless phase units) | $0.005 \le w \le 0.25$ | Splits span $0.015 - 0.125$ |
| 2 | `transit_period` | Orbital period extracted via BLS / periodogram (days) | $0.2 \le P \le 500.0\text{ days}$ | Major splits at $8.33\text{ d}$, $14.2\text{ d}$, $32.1\text{ d}$ |
| 3 | `snr` | Transit signal-to-noise ratio: $\delta \cdot \sqrt{N_{\text{in}}} / \sigma$ (dimensionless) | $3.0 \le \text{SNR} \le 150.0+$ | Splits span $5.2 - 28.5$ |
| 4 | `odd_even_mismatch` | Difference between odd and even transit depths ($\Delta F_{\text{odd}} - \Delta F_{\text{even}}$) | $0.0 \le \Delta_{\text{oe}} \le 0.10$ | Splits around $0.001 - 0.015$ |
| 5 | `secondary_depth` | Flux drop at phase $\phi \approx 0.5$ (secondary eclipse / occultation depth) | $0.0 \le \delta_{\text{sec}} \le 0.02$ | Splits span $0.0005 - 0.008$ |
| 6 | `flux_std` | Standard deviation of baseline-normalized flux ($\sigma_{\text{flux}}$, dimensionless) | $0.0005 \le \sigma \le 0.010$ | **Top root split at $0.00352$** (29.87% feature importance) |
| 7 | `flux_skew` | Fisher-Pearson skewness of normalized flux distribution (dimensionless) | $-3.0 \le \gamma_1 \le +3.0$ | Splits span $-1.5$ to $+1.2$ |
| 8 | `flux_min` | Minimum normalized flux relative to baseline ($F_{\min} - 1.0$ or raw minimum) | $-0.08 \le F_{\min} \le -0.0005$ | Splits span $-0.0125$ to $-0.0030$ |

### 2.3 How Features Were Calculated During Training
Analysis of `models/exoplanet_model.pkl` tree node thresholds confirms that:
- Training instances were derived from **normalized Kepler light curves** where stellar baseline flux was normalized to $F_0 \equiv 1.0$.
- In the training set, `flux_std` had a mean of $\sim 0.003$ ($0.3\%$ photometric dispersion), and `transit_depth` had a mean of $\sim 0.005$ ($5,000\text{ ppm}$).
- The top split in over 40% of the trees is `flux_std <= 0.00352`. If `flux_std > 0.00352`, the decision path immediately branches toward majority Class 0 leaves (non-exoplanet).

### 2.4 How Inference Currently Calculates Features
In `backend/feature_engineering.py`:
- `ScientificFeatureExtractor` expects normalized time-series data $(t, F)$.
- It runs Box Least Squares (BLS) periodogram analysis across a grid of candidate orbital periods ($0.5$ to $30$ days), finding the peak power to estimate period, duration, depth, and SNR.
- Statistical moments (`flux_std`, `flux_skew`, `flux_min`) are computed using standard numpy/scipy estimators.
- `_align_features` in `backend/prediction.py` validates column names, aligns order, and now strictly enforces boundary checks:
  ```python
  if flux_std > 1.0 or abs(flux_median - 1.0) > 0.5 or flux_min < -1.0:
      raise FeatureValidationError("Incompatible feature scale: expected normalized flux...")
  ```

### 2.5 How `exoTest.csv` Was Converted Into Model Inputs
In the historical evaluation script:
- `exoTest.csv` contains 570 rows where each row has a `LABEL` column (1 = non-planet, 2 = confirmed planet) and 3,197 flux cadences (`FLUX.1` to `FLUX.3197`).
- These flux cadences are **pre-whitened difference fluxes from the Kepler instrument**, not normalized light curves. The values are centered at $0.0$ and reach amplitudes of $\pm 16,000$.
- The historical script ran `ScientificFeatureExtractor` directly on these raw unnormalized rows without baseline normalization.
- Extracted features on `exoTest.csv`:
  * `flux_std`: **$100.0$ to $1,644.0$** (expected $\sim 0.0035$) $\rightarrow 30,000\times$ out of bounds.
  * `transit_depth`: **$50.0$ to $2,044.8$** (expected $\sim 0.004$) $\rightarrow 500,000\times$ out of bounds.
  * `flux_min`: **$-2,624.7$** (expected $\sim -0.01$) $\rightarrow 260,000\times$ out of bounds.
  * `flux_median`: **$\approx 0.0$** (expected $1.0$).
- When these numbers were fed into `exoplanet_model.pkl`, every single sample had `flux_std > 0.00352` and `transit_depth > 0.045`, triggering non-exoplanet splits across all 100 trees.

### 2.6 Compatibility Matrix

| Property | Original Training | Live Inference (`02_Detect.py`) | Benchmark (`exoTest.csv`) | Compatibility Status |
|---|---|---|---|---|
| **Flux Baseline** | Normalized ($F_0 \approx 1.0$) | Normalized ($F_0 \approx 1.0$) | Zero-centered difference ($\mu \approx 0.0$) | **INCOMPATIBLE** |
| **Flux Std Dev** | $\sim 0.001 - 0.009$ | $\sim 0.001 - 0.009$ | $10.0 - 1,644.0$ | **INCOMPATIBLE** |
| **Transit Depth** | $0.0001 - 0.05$ | $0.0001 - 0.05$ | $1.0 - 2,044.8$ | **INCOMPATIBLE** |
| **Feature Ordering** | 9 features (fixed) | 9 features (fixed) | 9 features (fixed) | COMPATIBLE |
| **Class Indexing** | `0=Negative, 1=Positive` | `0=Negative, 1=Positive` | `1=Negative, 2=Positive` | Mapped (`LABEL-1`) |

**Verdict:** The feature pipelines for training and live inference are mutually consistent, but `exoTest.csv` is completely incompatible with both.

---

## 3. Verification of Evaluation Validity (`benchmark_evaluation.json`)

### 3.1 Prior Evaluation Code Inspection
The historical evaluation in `models/benchmark_evaluation.json` recorded:
- Accuracy: **99.12%**
- Precision: **0.0%**
- Recall: **0.0%**
- F1 Score: **0.0%**
- ROC-AUC: **0.6430**
- Confusion Matrix: `[[565, 0], [5, 0]]` (565 True Negatives, 5 False Negatives, 0 True Positives, 0 False Positives).

### 3.2 Probability Column Selection Check
- The model's `classes_` array is `array([0, 1])`.
- In binary classification, probability column `1` corresponds to Class 1 (Exoplanet Candidate).
- The prior script selected column `1` (`proba[:, 1]`). Thus, the 0% recall was **not caused by indexing the wrong probability column**.
- It was caused exclusively by the out-of-distribution feature values forcing all trees down negative leaves.
- The 99.12% accuracy is simply the proportion of negative cases in `exoTest.csv` ($565 / 570 = 99.1228\%$), representing a trivial majority-class baseline.
- The 0.643 ROC-AUC reflects marginal ranking separation among noise features, not a calibrated detector.

### 3.3 Validity Classification & Preservation
In accordance with scientific integrity guidelines:
- **Status:** Marked as `inconclusive_feature_mismatch`.
- **Fair Assessment Flag:** `is_fair_assessment = false`.
- **Action:** The existing metrics are preserved in `models/benchmark_evaluation.json` as historical diagnostic artifacts with explicit limitations metadata. No replacement metrics have been fabricated.
- **UI Transparency:** `pages/05_Model_Lab.py` and `pages/01_Observatory.py` prominently display `"Historical Benchmark Diagnostic (Inconclusive)"` and warn users of the feature distribution mismatch.

---

## 4. Scientific Non-Detection Semantics

### 4.1 Astronomical Principles
In observational astronomy:
1. **Geometric Transit Probability:** The probability that a planetary orbit is aligned such that the planet transits the stellar disc is $P_{\text{transit}} \approx R_* / a$. For Earth-Sun analogues, $P_{\text{transit}} \approx 0.47\%$; for hot Jupiters, $P_{\text{transit}} \approx 10\%$. More than 90% of planet-hosting stars will never show a transit from Earth's vantage point.
2. **Temporal Coverage:** A planet with an orbital period longer than the photometric observing baseline will not exhibit periodic transits during the observation window.
3. **Signal-to-Noise & Noise Floor:** A sub-Earth planet orbiting a noisy or distant star may produce a dip below the instrumental detection limit ($\sim 10 - 20\text{ ppm}$).

Therefore, the previous behavior—labeling any light curve with no detected transit dip as **"Non-Exoplanet" with 100% confidence**—was scientifically false.

### 4.2 Corrected Honest Semantics
`backend/prediction.py` now maps all prediction scenarios to three honest scientific states:

```
                                  [Light Curve Input]
                                           |
                           ---------------------------------
                          |                                 |
               [Insufficient / Flat Data]             [Valid Data]
                          |                                 |
              --------------------------             [Transit Search (BLS)]
             |                          |                   |
    [Points < 50]              [Zero Variance]       -------------------------
             |                          |           |                         |
             v                          v     [Transit Found]          [No Transit Found]
  "Unable to evaluate — insufficient            |                              |
            or invalid data"                    v                              v
        (confidence = 0.0)            [Model Inference]            "No transit detected —
                                                |                       inconclusive"
                                    ------------------------         (confidence = 0.0)
                                   |                        |
                              [Positive]               [Negative]
                                   |                        |
                                   v                        v
                        "Transit detected — candidate     "No transit detected —
                            for further analysis"             inconclusive"
                             (model confidence)            (confidence = 0.0)
```

1. **`Transit detected — candidate for further analysis`**:
   - Condition: Transit detected by BLS and classified as positive candidate by model.
   - Confidence: Calibrated model probability $P(\text{Exoplanet})$.
   - Grade: `"Candidate"` / `"Tier 1 Priority"`.
2. **`No transit detected — inconclusive`**:
   - Condition: No periodic transit detected or model score below threshold.
   - Confidence: `0.0` (uncalibrated / not applicable).
   - Grade: `"Inconclusive"`.
   - Narrative: *"No periodic transit signal was detected in this photometric light curve. Note that non-detection does not rule out the presence of non-transiting exoplanets or planets below the detection threshold."*
3. **`Unable to evaluate — insufficient or invalid data`**:
   - Condition: Fewer than 50 photometric points, constant/flat flux, or unnormalized difference flux.
   - Confidence: `0.0`.
   - Grade: `"Inconclusive"`.
   - Narrative: *"Insufficient or invalid photometric data points to extract a meaningful signal."*

### 4.3 Dynamic Probability Mapping
To prevent bugs if scikit-learn classes are ordered `[1, 0]` or strings `["Non-Exoplanet", "Exoplanet"]`:
```python
if 1 in model.classes_:
    pos_idx = list(model.classes_).index(1)
elif "Exoplanet" in model.classes_:
    pos_idx = list(model.classes_).index("Exoplanet")
else:
    pos_idx = 1
confidence = float(proba[pos_idx])
```

---

## 5. Regression Testing & Verification

A dedicated regression suite (`tests/test_model_compatibility_regression.py`) was created with 13 comprehensive test cases. The complete test suite was executed:

```
tests/test_audit_fixes.py::test_no_st_sidebar_inside_column PASSED
tests/test_audit_fixes.py::test_css_no_unscoped_element_selectors PASSED
tests/test_audit_fixes.py::test_pages_exist_and_readable PASSED
tests/test_audit_fixes.py::test_model_lab_structure PASSED
tests/test_backend.py::test_lightcurve_preprocessor_clean PASSED
tests/test_backend.py::test_lightcurve_preprocessor_outliers PASSED
tests/test_backend.py::test_feature_extractor_transit PASSED
tests/test_backend.py::test_feature_extractor_noise PASSED
tests/test_backend.py::test_fold_light_curve PASSED
tests/test_backend.py::test_batch_summary_math PASSED
tests/test_features.py::test_feature_extractor_contract PASSED
tests/test_features.py::test_feature_extractor_noise_flat PASSED
tests/test_features.py::test_feature_extractor_reproducible PASSED
tests/test_features.py::test_feature_extractor_insufficient_data PASSED
tests/test_features.py::test_feature_extractor_constant_flux PASSED
tests/test_features.py::test_feature_extractor_synthetic_kepler10b PASSED
tests/test_features.py::test_feature_vector_keys_and_types PASSED
tests/test_features.py::test_feature_extractor_empty_series PASSED
tests/test_features.py::test_feature_extractor_nan_handling PASSED
tests/test_features.py::test_feature_extractor_negative_flux PASSED
tests/test_model_compatibility_regression.py::test_feature_names_ordering_and_units PASSED
tests/test_model_compatibility_regression.py::test_rejection_of_unnormalized_difference_flux PASSED
tests/test_model_compatibility_regression.py::test_dynamic_positive_class_probability_mapping PASSED
tests/test_model_compatibility_regression.py::test_reversed_class_order_probability_mapping PASSED
tests/test_model_compatibility_regression.py::test_string_class_order_probability_mapping PASSED
tests/test_model_compatibility_regression.py::test_no_transit_data_treated_as_inconclusive PASSED
tests/test_model_compatibility_regression.py::test_insufficient_data_produces_honest_result PASSED
tests/test_model_compatibility_regression.py::test_constant_flux_produces_honest_result PASSED
tests/test_model_compatibility_regression.py::test_benchmark_evaluation_marked_inconclusive PASSED
tests/test_model_compatibility_regression.py::test_model_metadata_marked_inconclusive PASSED
tests/test_model_compatibility_regression.py::test_predict_lightcurve_clean_transit_candidate PASSED
tests/test_model_compatibility_regression.py::test_report_formatting_honest_semantics PASSED
tests/test_model_compatibility_regression.py::test_batch_summary_supports_honest_labels PASSED
tests/test_model_evaluation.py::test_benchmark_json_exists_and_valid PASSED
tests/test_model_evaluation.py::test_benchmark_metrics_types PASSED
tests/test_model_evaluation.py::test_confusion_matrix_dimensions PASSED
tests/test_model_evaluation.py::test_roc_pr_curves_structure PASSED
tests/test_model_evaluation.py::test_per_class_metrics PASSED
tests/test_model_evaluation.py::test_metadata_exists_and_valid PASSED
tests/test_model_evaluation.py::test_dataset_info PASSED
tests/test_model_evaluation.py::test_model_parameters PASSED
tests/test_model_evaluation.py::test_feature_importance_consistency PASSED
tests/test_model_evaluation.py::test_evaluation_metrics_consistency PASSED
tests/test_prediction.py::test_predict_single_exoplanet PASSED
tests/test_prediction.py::test_predict_single_noise PASSED
tests/test_prediction.py::test_predict_batch PASSED
tests/test_prediction.py::test_model_loading PASSED
tests/test_prediction.py::test_heuristic_fallback_when_model_missing PASSED
tests/test_prediction.py::test_generate_summary_exoplanet PASSED
tests/test_prediction.py::test_generate_summary_candidate_heuristic PASSED
tests/test_prediction.py::test_generate_summary_non_planet PASSED
tests/test_prediction.py::test_generate_summary_indeterminate PASSED
tests/test_prediction.py::test_prediction_reproducibility PASSED
tests/test_prediction.py::test_align_features_valid PASSED
tests/test_prediction.py::test_align_features_missing_column PASSED
tests/test_prediction.py::test_align_features_order PASSED
tests/test_prediction.py::test_predict_single_with_raw_features PASSED

============================= 57 passed in 3.97s =============================
```
**Test Pass Rate:** **100% (57 passed, 0 failed, 0 skipped)**.

---

## 6. Confirmed Findings, Unresolved Questions, & Recommended Next Steps

### 6.1 Confirmed Findings
1. `models/exoplanet_model.pkl` is internally consistent with normalized relative flux inputs ($F_0 \equiv 1.0$), expecting `flux_std` $\sim 0.0035$ and `transit_depth` $\le 0.05$.
2. The benchmark dataset `exoTest.csv` (and `exoTrain.csv`) consists of raw pre-whitened difference fluxes that are 4 to 5 orders of magnitude larger than the model's feature space.
3. The historical benchmark evaluation (99.12% accuracy, 0% recall) was an invalid measurement resulting from this out-of-distribution feature scale mismatch.
4. The previous non-detection logic (claiming 100% confidence "Non-Exoplanet") was scientifically flawed and has been replaced with honest, 3-state astronomical semantics.
5. In accordance with safety rules, `models/exoplanet_model.pkl` was **not modified or retrained**.

### 6.2 Missing Components & Unresolved Questions
1. **Missing Training Script:** There is no script in the repository detailing how `models/exoplanet_model.pkl` was originally trained, how hyperparameters were tuned, or what cross-validation scheme was employed.
2. **Missing Original Training Dataset:** The model was trained on $\approx 1,200$ samples with $\approx 41\%$ positive class (likely curated Kepler threshold-crossing events or KOI table extracts), but that specific dataset is not present in the workspace.
3. **Threshold Calibration:** The model's decision threshold is currently default $P \ge 0.5$. With an imbalanced domain like exoplanet candidate vetting, threshold tuning on a precision-recall curve is necessary to balance candidate discovery rate against false alarm vetting costs.

### 6.3 Recommended Next Steps (Future Work)
1. **Ingest Real Kepler Cumulative TCE Table:** Download the official NASA Exoplanet Archive Kepler DR25 Threshold Crossing Event (TCE) or Kepler Objects of Interest (KOI) catalog with pre-computed astrophysical features (`transit_depth`, `transit_period`, `transit_duration`, `snr`).
2. **Construct an End-to-End Retraining Pipeline:** Write a version-controlled script (`backend/train_model.py`) that trains, validates, and calibrates a classifier directly on standardized Kepler TCE catalogs.
3. **Calibrate Decision Thresholds:** Perform threshold optimization targeting high recall ($\ge 95\%$) on confirmed exoplanets while maintaining acceptable false positive rejection.
4. **Benchmark on Calibrated Data:** Re-run benchmark evaluations on a verified, normalized holdout set and update `benchmark_evaluation.json` with valid empirical metrics.

---

## 7. Modified Files and Safety Compliance Log

| File | Status | Backup Path | Modification Summary |
|---|---|---|---|
| `backend/prediction.py` | Modified | `.backup_task6/prediction.py` | Physical boundary validation in `_align_features`, honest 3-state prediction routing, dynamic positive class index resolution. |
| `backend/metrics.py` | Modified | `.backup_task6/metrics.py` | Updated candidate detection mask for batch summaries to support new honest verdict strings. |
| `components/__init__.py` | Modified | N/A | Wrapped Streamlit imports in `try/except` to allow headless test runners to import report generation. |
| `components/cards.py` | Modified | `.backup_task6/cards.py` | Render honest verdict badges (Amber for invalid, Slate for inconclusive, Emerald for transit candidate). |
| `components/report.py` | Modified | `.backup_task6/report.py` | Fixed bug where `features` was referenced before definition; formatted honest verdict labels and N/A scores. |
| `pages/02_Detect.py` | Modified | `.backup_task6/02_Detect.py` | Signal status KPI updated to "INVALID DATA", "NO TRANSIT", or "TRANSIT"; score card guarded against misleading 100% confidence. |
| `pages/05_Model_Lab.py` | Modified | `.backup_task6/05_Model_Lab.py` | Renamed metrics to "DIAGNOSTIC", added transparency disclosure banner, added live CSV validation in Evaluator tab. |
| `pages/01_Observatory.py` | Modified | `.backup_task6/01_Observatory.py` | Labeled KPI card as "Kepler Test Diagnostic (ROC: 0.643)" with context. |
| `models/benchmark_evaluation.json` | Modified | `.backup_task6/benchmark_evaluation.json` | Added validity metadata (`"inconclusive_feature_mismatch"`), preserved historical metrics with clear limitations notice. |
| `models/model_metadata.json` | Modified | `.backup_task6/model_metadata.json` | Added evaluation validity flags and updated class imbalance warning. |
| `tests/test_prediction.py` | Modified | `.backup_task6/test_prediction.py` | Updated assertions to match honest verdict semantics. |
| `tests/test_model_compatibility_regression.py` | **New** | N/A | Authored 13 regression tests covering all Task 6 objectives. |
| `MODEL_COMPATIBILITY_AUDIT.md` | **New** | N/A | Comprehensive audit report (this document). |

### Strict Safety Constraint Verification
- `models/exoplanet_model.pkl` was **NOT** overwritten, modified, or retrained.
- NASA ExoMiner++ model files (`ExoMiner-main/`) were **NOT** modified.
- No artificial training data or synthetic performance metrics were invented.
- All modifications are accompanied by automated regression tests passing at 100%.

