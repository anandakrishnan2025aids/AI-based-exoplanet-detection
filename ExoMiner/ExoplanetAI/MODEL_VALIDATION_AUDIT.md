# Scientific & Machine Learning Model Validation Audit Report
**Target Application:** ExoplanetAI Primary Vetting Pipeline  
**Model Under Review:** `ExoMiner/ExoplanetAI/models/exoplanet_model.pkl`  
**Metadata Specification:** `ExoMiner/ExoplanetAI/models/model_metadata.json`  
**Evaluation Date:** October 2026  
**Auditor:** Senior Machine Learning Engineer & Astronomical Data Specialist  

---

## 1. Executive Summary

An in-depth, evidence-based scientific audit of the ExoplanetAI prediction pipeline was conducted. The application deploys a 100-tree Scikit-Learn `RandomForestClassifier` operating on 9 hand-engineered statistical and photometric features. 

### Key Findings
1. **Severe Evaluation Collapse on Independent Test Data:**
   When evaluated on the independent, officially provided Kepler benchmark test set ([`exoTest.csv`](file:///c:/Users/anand/OneDrive/ExoMiner-main/ExoMiner/ExoplanetAI/data/training/exoTest.csv)), the saved model **missed all 5 confirmed exoplanets** (Recall = **0.00%**, Precision = **0.00%**, F1 = **0.00%**). The reported 99.12% accuracy is a statistical artifact of extreme class imbalance (565 non-exoplanets vs 5 exoplanets).
2. **Fabricated / Hardcoded Benchmark Claims in UI:**
   The performance claims presented in [`pages/05_Model_Lab.py`](file:///c:/Users/anand/OneDrive/ExoMiner-main/ExoMiner/ExoplanetAI/pages/05_Model_Lab.py) (e.g. 99.3% accuracy, ROC-AUC = 0.925, 5-fold cross-validation scores) are **hardcoded strings and synthetic polynomials** (`1 - (1 - fpr)**4.2`), not empirical measurements computed from astronomical test data.
3. **Flawed Astronomical Feature Extraction:**
   - **Transit Depth:** Defined as `median - min(flux)`, making the metric hostage to single-pixel downward outliers, noise spikes, or cosmic rays.
   - **Transit Duration:** Computed by summing all cadences below $2\sigma$ across the entire light curve. On multi-transit curves, it sums all transits together and hits an artificial 18-hour clamp.
   - **Orbital Period:** Relies on median spacing of threshold dips (`np.diff(dip_times) > 0.2`) with an arbitrary fallback of `3.524` days rather than Box Least Squares (BLS) or Lomb-Scargle periodograms.
4. **Silent Error Suppression & Signal Synthesis:**
   If model inference throws an exception, the system silently catches it and substitutes a heuristic fallback (`_astrophysical_heuristic`), returning a plausible-looking prediction without alerting the user. Additionally, missing inputs are backfilled with an invented 1.0% transit depth and an SNR of 5.0.

---

## 2. Complete Prediction Pipeline Trace

The end-to-end inference flow spans six modules and UI components:

```
[UI Ingestion] ────────► [Preprocessing] ────────► [Feature Extraction] 
  02_Detect.py             preprocessing.py          feature_engineering.py
                                                              │
[Final Display] ◄─────── [Post-Processing] ◄────── [Model Inference]
  02_Detect.py             prediction.py             prediction.py
```

| Pipeline Stage | File Location | Class / Function | Operation Performed |
| :--- | :--- | :--- | :--- |
| **1. UI Ingestion** | [`pages/02_Detect.py`](file:///c:/Users/anand/OneDrive/ExoMiner-main/ExoMiner/ExoplanetAI/pages/02_Detect.py#L61-L125) | `st.radio`, `st.file_uploader` | Ingests CSV or selects demo benchmark light curves (`time`, `flux`). |
| **2. Validation** | [`backend/validation.py`](file:///c:/Users/anand/OneDrive/ExoMiner-main/ExoMiner/ExoplanetAI/backend/validation.py#L12) | `validate_lightcurve_df()` | Confirms DataFrame columns `time` and `flux`, non-empty series, and finite floats. |
| **3. Preprocessing** | [`backend/preprocessing.py`](file:///c:/Users/anand/OneDrive/ExoMiner-main/ExoMiner/ExoplanetAI/backend/preprocessing.py#L67-L90) | `LightCurvePreprocessor.process()` | Filters NaNs/Infs, applies asymmetric sigma clipping, normalizes by median, and detrends using a 41-cadence rolling median. |
| **4. Feature Extraction** | [`backend/feature_engineering.py`](file:///c:/Users/anand/OneDrive/ExoMiner-main/ExoMiner/ExoplanetAI/backend/feature_engineering.py#L78-L125) | `ExoplanetFeatureExtractor.extract_from_series()` | Computes 9 features: `transit_depth`, `period_days`, `duration_hours`, `snr`, `flux_std`, `flux_skew`, `flux_kurtosis`, `flux_min`, `flux_median`. |
| **5. Feature Alignment** | [`backend/prediction.py`](file:///c:/Users/anand/OneDrive/ExoMiner-main/ExoMiner/ExoplanetAI/backend/prediction.py#L58-L79) | `ExoplanetPredictor._align_features()` | Aligns dictionary keys to `model.feature_names_in_`. Injects default values if columns are missing. |
| **6. Model Loading** | [`backend/prediction.py`](file:///c:/Users/anand/OneDrive/ExoMiner-main/ExoMiner/ExoplanetAI/backend/prediction.py#L38-L57) | `ExoplanetPredictor._load_model()` | Deserializes `models/exoplanet_model.pkl` via `joblib.load()`. |
| **7. Inference** | [`backend/prediction.py`](file:///c:/Users/anand/OneDrive/ExoMiner-main/ExoMiner/ExoplanetAI/backend/prediction.py#L80-L128) | `ExoplanetPredictor.predict_single()` | Calls `model.predict_proba()` to compute $P(\text{Exo})$; falls back to `_astrophysical_heuristic()` upon failure. |
| **8. Parameter Derivation** | [`pages/02_Detect.py`](file:///c:/Users/anand/OneDrive/ExoMiner-main/ExoMiner/ExoplanetAI/pages/02_Detect.py#L366-L420) | Inline UI calculations | Derives physical attributes: $R_p = \sqrt{\delta} \cdot R_\odot$, $a = (P / 365.25)^{2/3}$, $T_{\text{eq}} = 5778 \cdot \sqrt{\frac{1}{2 a \cdot 215.03}}$. |
| **9. Visualization & Report** | [`components/cards.py`](file:///c:/Users/anand/OneDrive/ExoMiner-main/ExoMiner/ExoplanetAI/components/cards.py), [`components/charts.py`](file:///c:/Users/anand/OneDrive/ExoMiner-main/ExoMiner/ExoplanetAI/components/charts.py) | `render_verdict_badge()`, `plot_phase_folded_transit()` | Renders status badge, phase-folded transit profile, explainability bars, and export dossier. |

---

## 3. Feature Correctness & Astronomical Soundness

| Feature | Extraction Logic | Expected Units | Astronomical / Mathematical Validity | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| `transit_depth` | `(median - min(flux)) / median` | Relative flux ($\Delta F / F$) | **Critically Flawed.** Using a single point `min(flux)` makes depth vulnerable to single noise excursions. In astronomy, depth must be estimated via box fitting or Mandel-Agol transit modeling. | ❌ High Risk |
| `period_days` | `median(diff(dip_times > 0.2))` | Days | **Unreliable.** Taking consecutive differences of points below $2\sigma$ fails on transits lasting $>0.2\text{d}$. Falls back to arbitrary constant `3.524` days if FFT fails. | ❌ High Risk |
| `duration_hours`| `sum(flux < 2σ) * dt * 24` | Hours | **Flawed.** Sums cadences across all transits over the entire observation. A star with 10 transits reports 10x the actual duration, bounded only by a clamp to `18.0` hours. | ❌ High Risk |
| `snr` | `transit_depth / flux_std` | Dimensionless | **Uncalibrated.** Because `transit_depth` is the maximum downward deviation from median, on pure Gaussian noise `snr` is mathematically $\sim 3.0$ even with zero planetary signal. | ⚠️ Moderate Risk |
| `flux_skew` | `scipy.stats.skew(flux)` | Dimensionless | **Mathematically sound**, but prone to catastrophic cancellation / precision loss warnings on flat/identical flux series. | ⚠️ Minor Warning |
| `flux_kurtosis`| `scipy.stats.kurtosis(flux)`| Dimensionless | **Mathematically sound**, captures heavy-tailed transit dips. | ✅ Valid |
| `flux_std` | `np.std(flux)` | Relative flux | **Mathematically sound**, measures photometric dispersion. | ✅ Valid |
| `flux_min` | `np.min(flux)` | Relative flux | Redundant with `transit_depth` when normalized by median. | ⚠️ Redundant |
| `flux_median` | `np.median(flux)` | Relative flux | Near 1.0 for normalized light curves; constant for all preprocessed data. | ⚠️ Zero Variance |

---

## 4. Empirical Edge-Case & Failure-Mode Testing

We executed [`scratch/test_model_edge_cases.py`](file:///C:/Users/anand/.gemini/antigravity-ide/brain/e5ff134d-913d-4469-98e2-9ab380f9d4e4/scratch/test_model_edge_cases.py) with safe test inputs:

```
================ TEST RESULTS ON EDGE CASES ================
Test Input Description                        Predicted Label    Confidence   P(Exoplanet)   Status
---------------------------------------------------------------------------------------------------
Valid Transit Features                        Exoplanet          0.7200       0.7200         Expected Behavior
Empty Dictionary {}                           Non-Exoplanet      0.7000       0.3000         Fabricates Features
All Zeros                                     Non-Exoplanet      1.0000       0.0000         Handled
Only transit_depth=0.02 (rest missing)        Non-Exoplanet      0.7000       0.3000         Injected Defaults
Constant Flat Light Curve (Flux=1.0)          Non-Exoplanet      0.8800       0.1200         Precision Warning
Pure Gaussian White Noise (σ=0.005)           Non-Exoplanet      0.8100       0.1900         Generated False Depth
Single Downward Glitch (1 bad cadence)        Non-Exoplanet      0.8800       0.1200         Handled
Light Curve with NaNs and Infs                Non-Exoplanet      0.8800       0.1200         NaN-filtered
```

### Critical Observations:
- **Pure Gaussian White Noise:** Produced a fabricated `transit_depth` of **1.65%**, a `period_days` of **0.74d**, an SNR of **3.32**, and a `duration_hours` of **14.89h** despite having zero real astronomical signal.
- **Constant Flux Curves:** Triggered `RuntimeWarning: Precision loss occurred in moment calculation due to catastrophic cancellation` in `scipy.stats.skew` and `kurtosis`.

---

## 5. Audit of Confidence Scores & Fallback Behavior

### A. Raw Model Output vs. Calibrated Confidence
- The application describes the classifier as a **"Calibrated Random Forest"** ([`pages/02_Detect.py:4`](file:///c:/Users/anand/OneDrive/ExoMiner-main/ExoMiner/ExoplanetAI/pages/02_Detect.py#L4), [`pages/05_Model_Lab.py:79`](file:///c:/Users/anand/OneDrive/ExoMiner-main/ExoMiner/ExoplanetAI/pages/05_Model_Lab.py#L79)).
- **Audit Reality:** The model output is merely raw tree vote fraction (`model.predict_proba()` without any Platt scaling, Isotonic regression, or `CalibratedClassifierCV`). Uncalibrated tree votes are not true posterior probabilities.

### B. Silent Heuristic Fallback
In [`backend/prediction.py`](file:///c:/Users/anand/OneDrive/ExoMiner-main/ExoMiner/ExoplanetAI/backend/prediction.py#L109-L117):
```python
except Exception as e:
    # Rule-based fallback if model fails
    prob_exo = self._astrophysical_heuristic(features)
    prob_non = 1.0 - prob_exo
    pred_class = int(prob_exo >= 0.5)
```
- **Vulnerability:** If the model throws an unpickling error, shape mismatch, or internal exception, the failure is caught silently. The user is presented with a heuristic score disguised as an AI prediction with "High Confidence", with no warning or error logged.

### C. Artificial Feature Injection
In [`backend/prediction.py`](file:///c:/Users/anand/OneDrive/ExoMiner-main/ExoMiner/ExoplanetAI/backend/prediction.py#L70-L76):
```python
if "depth" in col:
    row[col] = 0.01   # Injects 1% depth
elif "period" in col:
    row[col] = 3.52   # Injects 3.52-day period
elif "snr" in col:
    row[col] = 5.0    # Injects SNR=5.0
```
- Missing features automatically synthesize strong planetary transit attributes.

---

## 6. Empirical Evaluation on Kepler Test Dataset (`exoTest.csv`)

To assess true generalization, we evaluated the unmodified saved model [`exoplanet_model.pkl`](file:///c:/Users/anand/OneDrive/ExoMiner-main/ExoMiner/ExoplanetAI/models/exoplanet_model.pkl) against the standard benchmark test dataset [`data/training/exoTest.csv`](file:///c:/Users/anand/OneDrive/ExoMiner-main/ExoMiner/ExoplanetAI/data/training/exoTest.csv) (570 stars: 565 non-exoplanets, 5 confirmed exoplanets).

```
================ INDEPENDENT BENCHMARK EVALUATION ================
Dataset:                    exoTest.csv (Kepler Labeled Time Series)
Sample Count:               570 stars (3,197 cadences each)
Class Balance:              565 Non-Exoplanet Stars (99.12%)
                            5 Confirmed Exoplanet Stars (0.88%)
------------------------------------------------------------------
Overall Accuracy:           99.12% (0.9912)
Precision:                  0.0000 (0.0%)
Recall:                     0.0000 (0.0%)  <-- FAILED TO DETECT ANY EXOPLANET
F1-Score:                   0.0000 (0.0%)
ROC-AUC:                    0.7039
------------------------------------------------------------------
Confusion Matrix:
                    Predicted Non-Exo    Predicted Exo
Actual Non-Exo:            565                 0
Actual Exoplanet:            5                 0
```

### Detailed Breakdown of the 5 Ground-Truth Exoplanets:
1. **Target #0:** Predicted = Non-Exoplanet ($P = 0.0100$)
2. **Target #1:** Predicted = Non-Exoplanet ($P = 0.0300$)
3. **Target #2:** Predicted = Non-Exoplanet ($P = 0.0400$)
4. **Target #3:** Predicted = Non-Exoplanet ($P = 0.3300$)
5. **Target #4:** Predicted = Non-Exoplanet ($P = 0.3300$)

### Why Did the Model Fail Completely?
1. **Raw Flux vs Normalized Flux Mismatch:** `exoTest.csv` contains raw Kepler flux counts (unnormalized numbers in the tens of thousands), whereas `exoplanet_model.pkl` was fitted with features derived from normalized flux ($\sim 1.0$).
2. **Extreme Class Imbalance Collapse:** The model defaulted to the majority class. Because 99.12% of the samples are negative, a dummy classifier predicting all zeros achieves 99.12% accuracy while having **zero scientific utility**.

---

## 7. Advertised Performance Claims vs. Audit Findings

| Metric / Claim | Claimed in Metadata / UI | Actual Audit Finding | Discrepancy Analysis |
| :--- | :--- | :--- | :--- |
| **Model Metadata Metrics** | `accuracy: 1.0, precision: 1.0, recall: 1.0, f1: 1.0, roc_auc: 1.0` | **Precision = 0.0, Recall = 0.0, F1 = 0.0** on `exoTest.csv` | Metadata values of `1.0` are unverified placeholders or overfitted training artifacts. |
| **UI Test Accuracy** | 99.3% ([`pages/05_Model_Lab.py:68`](file:///c:/Users/anand/OneDrive/ExoMiner-main/ExoMiner/ExoplanetAI/pages/05_Model_Lab.py#L68)) | **99.12%** solely due to 99.12% negative class prevalence | Deceptive metric; masks 0% recall on actual exoplanet targets. |
| **Confusion Matrix** | `[[238, 2], [0, 60]]` ([`pages/05_Model_Lab.py:117`](file:///c:/Users/anand/OneDrive/ExoMiner-main/ExoMiner/ExoplanetAI/pages/05_Model_Lab.py#L117)) | `[[565, 0], [5, 0]]` | UI confusion matrix is hardcoded static HTML, not derived from validation data. |
| **ROC Curve** | Area = 0.925 ([`pages/05_Model_Lab.py:154`](file:///c:/Users/anand/OneDrive/ExoMiner-main/ExoMiner/ExoplanetAI/pages/05_Model_Lab.py#L154)) | **AUC = 0.7039** on `exoTest.csv` | UI ROC curve is plotted from a hardcoded mathematical formula `tpr = 1 - (1-fpr)**4.2`. |
| **5-Fold Cross Validation** | Mean: 99.30% ± 0.15% ([`pages/05_Model_Lab.py:218`](file:///c:/Users/anand/OneDrive/ExoMiner-main/ExoMiner/ExoplanetAI/pages/05_Model_Lab.py#L218)) | No cross-validation code exists | Numbers are hardcoded text strings in an HTML container. |

---

## 8. Severity-Ranked Vulnerabilities Matrix

| Rank | Severity | Issue Description | Impact |
| :---: | :--- | :--- | :--- |
| **1** | 🔴 **CRITICAL** | **0% Recall on Independent Benchmark:** Model failed to detect all 5 ground-truth exoplanets in `exoTest.csv`. | System cannot reliably detect exoplanets in unseen time-series observations. |
| **2** | 🔴 **CRITICAL** | **Fabricated Scientific Visualizations in UI:** Hardcoded confusion matrix, synthetic polynomial ROC curve, and hardcoded CV telemetry. | Misleads users and researchers regarding the model's actual performance. |
| **3** | 🟠 **HIGH** | **Silent Error Suppression in Inference:** Model failures silently redirect to rule-based heuristics without user notification. | Prevents debugging and gives false confidence in broken models. |
| **4** | 🟠 **HIGH** | **Flawed Astronomical Feature Extraction:** `min(flux)` for depth and cadence summation across multi-transits for duration. | Distorts physical feature representations; noisy stars mimic deep transits. |
| **5** | 🟡 **MEDIUM** | **Uncalibrated Probabilities:** Raw tree vote proportions presented as "Calibrated Machine Learning". | Overconfident probability estimates on borderline candidates. |
| **6** | 🟡 **MEDIUM** | **Arbitrary Default Injections:** Missing feature alignment injects synthetic 1% depth, 3.52d period, and SNR=5.0. | Incomplete feature inputs can produce false positive candidate flags. |
| **7** | 🔵 **LOW** | **Scikit-Learn Version Serialization Warning:** Pickled with version 1.9.0, executed in 1.7.2. | Minor deprecation warning; risk of future deserialization breakage. |

---

## 9. Recommended Fixes in Priority Order

1. **Phase 1: Replace Hardcoded UI Artifacts with Real Data**
   - Replace static arrays in [`05_Model_Lab.py`](file:///c:/Users/anand/OneDrive/ExoMiner-main/ExoMiner/ExoplanetAI/pages/05_Model_Lab.py) with dynamic metrics computed on a dedicated, persistent evaluation split.
   - Update [`models/model_metadata.json`](file:///c:/Users/anand/OneDrive/ExoMiner-main/ExoMiner/ExoplanetAI/models/model_metadata.json) with honest, verified metrics.
2. **Phase 2: Harden Preprocessing & Feature Extraction**
   - Normalize all input cadences prior to feature extraction in `extract_from_flux_sequence()`.
   - Implement Box Least Squares (BLS) period searching using `astropy.timeseries.BoxLeastSquares` instead of peak-difference heuristics.
   - Estimate transit depth using median in-transit flux rather than raw `np.min(flux)`.
3. **Phase 3: Implement Proper Model Training & Class Balancing**
   - Train on balanced/augmented data using SMOTE, class-weighted Gini loss, or calibrated thresholds.
   - Employ `CalibratedClassifierCV` (isotonic or sigmoid) to yield genuine posterior probabilities.
4. **Phase 4: Eliminate Silent Heuristic Fallbacks**
   - Raise explicit exceptions or set a clear UI flag (`is_fallback: True`) when heuristic rules are used.
   - Prevent empty inputs from auto-generating planetary transit attributes.

---

## 10. Test Execution & Verification Summary

The test suite was executed to ensure complete repeatability:
- **Command:** `pytest "c:\Users\anand\OneDrive\ExoMiner-main\ExoMiner\ExoplanetAI\tests"`
- **Results:** **14 passed** in 4.12s
  - 10 pre-existing unit tests ([`test_backend.py`](file:///c:/Users/anand/OneDrive/ExoMiner-main/ExoMiner/ExoplanetAI/tests/test_backend.py), [`test_prediction.py`](file:///c:/Users/anand/OneDrive/ExoMiner-main/ExoMiner/ExoplanetAI/tests/test_prediction.py))
  - 4 new non-destructive audit regression tests ([`test_audit_scientific_correctness.py`](file:///c:/Users/anand/OneDrive/ExoMiner-main/ExoMiner/ExoplanetAI/tests/test_audit_scientific_correctness.py)) confirming:
    1. Empty input synthesis of 1% depth / SNR=5.0
    2. Multi-transit duration overcounting
    3. Gaussian white noise false-transit generation
    4. Silent heuristic fallback upon model corruption

*(Per task instructions, no model weights, architectures, or production application behaviors were altered during this audit.)*
