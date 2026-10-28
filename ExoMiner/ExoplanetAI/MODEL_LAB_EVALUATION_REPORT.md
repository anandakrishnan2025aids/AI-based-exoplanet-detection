# Model Lab Evaluation & Integrity Audit Report

**Project:** `ExoMiner/ExoplanetAI`  
**Model Under Review:** `models/exoplanet_model.pkl` (`RandomForestClassifier`, 100 estimators, 9 physical features)  
**Evaluation Artifact:** [`models/benchmark_evaluation.json`](file:///c:/Users/anand/OneDrive/ExoMiner-main/ExoMiner/ExoplanetAI/models/benchmark_evaluation.json)  
**Metadata Configuration:** [`models/model_metadata.json`](file:///c:/Users/anand/OneDrive/ExoMiner-main/ExoMiner/ExoplanetAI/models/model_metadata.json)  
**Evaluation Lab UI:** [`pages/05_Model_Lab.py`](file:///c:/Users/anand/OneDrive/ExoMiner-main/ExoMiner/ExoplanetAI/pages/05_Model_Lab.py)  
**Author:** Senior Machine Learning Engineer  
**Date:** October 28, 2026  

---

## 1. Executive Summary

In accordance with Task 5 and the findings documented in `MODEL_VALIDATION_AUDIT.md`, a comprehensive audit and remediation of the Model Lab evaluation engine was executed. All fabricated, hardcoded, and mathematically synthesized evaluation metrics were permanently removed from the application. 

The saved `exoplanet_model.pkl` Random Forest model was evaluated on the independent Kepler primary mission benchmark dataset (`exoTest.csv`, 570 targets) without altering weights or retraining. Genuine empirical classification metrics, confusion matrices, ROC-AUC rankings, and Precision-Recall curves were calculated and surfaced transparently in the Model Lab UI with explicit scientific disclosures regarding class imbalance and feature distribution mismatch.

All 44 automated regression tests (including 10 newly authored evaluation integrity tests) pass with zero errors.

---

## 2. Fabricated Metrics Removed

The following simulated and hardcoded elements were identified and eliminated:

| Location | Prior Fabricated Implementation | Scientific Defect | Corrected Implementation |
| :--- | :--- | :--- | :--- |
| **Confusion Matrix** (`05_Model_Lab.py:L117`) | `cm_matrix = [[238, 2], [0, 60]]` | Hardcoded static counts claiming 60 true positives and 0 false negatives on 300 targets. | Derived dynamically from actual predictions on test benchmark (`[[565, 0], [5, 0]]`) or live uploaded datasets. |
| **ROC Curve** (`05_Model_Lab.py:L153-154`) | `fpr = np.linspace(0, 1, 100); tpr = 1.0 - (1.0 - fpr)**4.2` | Synthetic polynomial curve formula masquerading as empirical classification performance to achieve AUC ~0.925. | Computed from actual prediction probabilities using `sklearn.metrics.roc_curve` and `roc_auc_score`. |
| **Cross-Validation Telemetry** (`05_Model_Lab.py:L191-223`) | Hardcoded HTML string claiming 5 folds: `Fold 1: 99.2%, Fold 2: 99.5%, Fold 3: 99.1%, Fold 4: 99.4%, Fold 5: 99.3%, Mean: 99.30% ± 0.15%`. | Fabricated numbers designed to sum to 5,657 targets without any executed cross-validation script. | Replaced with an honest protocol panel explaining that pre-trained weights cannot have CV scores back-calculated without raw training folds. |
| **Model Metadata** (`model_metadata.json:L16-22`) | `"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1_score": 1.0, "roc_auc": 1.0` | 100% metrics claim with zero empirical foundation. | Replaced with actual verified benchmark metrics (`accuracy: 0.9912, recall: 0.0, precision: 0.0, roc_auc: 0.643`) and an explicit class imbalance notice. |
| **KPI Fallbacks** (`05_Model_Lab.py:L68-72`, `01_Observatory.py:L48-65`) | Fallbacks to `99.3%` and `0.925` AUC. | Unverifiable promotional numbers. | Bound strictly to verified benchmark artifact and empirical evaluation outputs. |

---

## 3. Evaluation Dataset & Provenance

### Dataset Identification
- **File:** `data/training/exoTest.csv`
- **Origin & Provenance:** NASA Kepler Space Telescope Primary Mission Archive (Data Release 25), prepared in the standard Kaggle Exoplanet Hunting in Deep Space benchmark format.
- **Sample Count:** 570 stars
- **Cadences per Target:** 3,197 photometric observations per light curve.

### Class Mapping & Ground-Truth Labeling
- **Kaggle Label 2:** Confirmed Exoplanet candidate ($N = 5$ targets, $0.88\%$).
- **Kaggle Label 1:** Non-Exoplanet star ($N = 565$ targets, $99.12\%$).
- **Internal Model Schema:** Class 1 = Exoplanet, Class 0 = Non-Exoplanet. Standardized via `_standardize_binary_labels()`.

---

## 4. Actual Measured Metrics on Saved Model

The unmodified Random Forest model (`exoplanet_model.pkl`) was evaluated against `exoTest.csv` using the 9 physical features matching its training schema:

### Summary Performance Table
| Metric | Measured Value | Baseline / Interpretation |
| :--- | :--- | :--- |
| **Total Test Samples** | 570 stars | 3,197 flux cadences per star |
| **Class Distribution** | 565 Non-Exoplanets (99.12%)<br>5 Exoplanets (0.88%) | Extreme astronomical class imbalance |
| **Accuracy** | **99.12%** (0.9912) | Trivial majority-class baseline (predicting all zeros yields 99.12%) |
| **Precision** | **0.00%** (0.0000) | Zero positive predictions made at threshold 0.50 |
| **Recall (Sensitivity)** | **0.00%** (0.0000) | **Detected 0 of 5 confirmed exoplanets** |
| **F1-Score** | **0.00%** (0.0000) | Harmonic mean of precision and recall |
| **ROC-AUC** | **0.6430** | Weak empirical probability ranking capability across thresholds |
| **Average Precision (PR-AUC)** | **0.0174** | Positive prevalence baseline = 0.0088 (5/570) |

### Empirical Confusion Matrix
```
                         Predicted Non-Exoplanet    Predicted Exoplanet
Ground Truth Non-Exo:              565                       0
Ground Truth Exoplanet:              5                       0
```
- **True Negatives (TN):** 565
- **False Positives (FP):** 0
- **False Negatives (FN):** 5
- **True Positives (TP):** 0

### Ground Truth Exoplanet Breakdown
All 5 confirmed exoplanet stars in `exoTest.csv` were assigned low predicted exoplanet probabilities ($\tau < 0.50$) by `exoplanet_model.pkl`:
- **Target #0:** Predicted Non-Exoplanet ($P = 0.0100$)
- **Target #1:** Predicted Non-Exoplanet ($P = 0.0300$)
- **Target #2:** Predicted Non-Exoplanet ($P = 0.0400$)
- **Target #3:** Predicted Non-Exoplanet ($P = 0.3300$)
- **Target #4:** Predicted Non-Exoplanet ($P = 0.3300$)

---

## 5. Feature Compatibility & Data Distribution Concerns

Our investigation revealed three critical reasons why `exoplanet_model.pkl` failed to detect the benchmark exoplanets:

1. **Stellar Flux Normalization Disconnect:**
   - `exoTest.csv` contains raw / pre-whitened difference flux cadences with values ranging between $-16,000$ and $+10,000$, centered around $0.0$.
   - `exoplanet_model.pkl` was fitted with features derived from relative flux normalized around $1.0$ (where `flux_median` $= 1.0$ and transit depth is $\Delta F / F \sim 0.001 - 0.05$).
2. **Missing Multi-Sector Period Information:**
   - The corrected Astropy Box Least Squares (BLS) period finder requires at least two observed transits across an observation baseline to declare a periodic signal. On single-sector snippets where no recurring dip is statistically distinct from noise ($SDE < 6.0$), `period_days` is reported as `None`.
   - The Random Forest model cannot ingest `None` values and was originally trained with artificial static defaults ($3.524$ days).
3. **Severe Class Imbalance Collapse:**
   - Because $99.12\%$ of the training samples were non-exoplanets, the decision tree ensemble optimized Gini impurity by splitting almost exclusively on majority-class non-detection branches. Without cost-sensitive class re-weighting or SMOTE resampling, the model defaults to the negative class.

---

## 6. Files Changed & Implementation Details

1. **[`backend/metrics.py`](file:///c:/Users/anand/OneDrive/ExoMiner-main/ExoMiner/ExoplanetAI/backend/metrics.py)**:
   - Added `_standardize_binary_labels()` to reliably handle standard $\{0, 1\}$ and Kaggle $\{1, 2\}$ labels.
   - Enhanced `calculate_classification_metrics()` with explicit zero-division handling (`zero_division=0.0`), positive class identification, and honest `roc_auc: None` reporting for single-class datasets.
   - Enhanced `generate_confusion_matrix_data()` with dynamic count extraction and total sample telemetry.
   - Added `calculate_roc_curve_data()`: Generates genuine empirical ROC curve coordinates using `sklearn.metrics.roc_curve` and `roc_auc_score`. Rejects single-class datasets with explicit explanations.
   - Added `calculate_pr_curve_data()`: Generates Precision-Recall curve coordinates and Average Precision score for imbalanced evaluations.
   - Added `load_benchmark_evaluation_artifact()`: Loads reproducible benchmark artifact from `models/benchmark_evaluation.json`.

2. **[`components/charts.py`](file:///c:/Users/anand/OneDrive/ExoMiner-main/ExoMiner/ExoplanetAI/components/charts.py)**:
   - Added `plot_roc_curve()`: Plotly interactive visualization for empirical ROC curves with random guess diagonal line and true threshold points.
   - Added `plot_precision_recall_curve()`: Plotly interactive visualization for empirical Precision-Recall curves with class prevalence baseline overlay.

3. **[`models/benchmark_evaluation.json`](file:///c:/Users/anand/OneDrive/ExoMiner-main/ExoMiner/ExoplanetAI/models/benchmark_evaluation.json)**:
   - Created reproducible evaluation artifact storing exact empirical results, confusion matrix, ROC coordinates, and PR coordinates on `exoTest.csv`.

4. **[`models/model_metadata.json`](file:///c:/Users/anand/OneDrive/ExoMiner-main/ExoMiner/ExoplanetAI/models/model_metadata.json)**:
   - Removed unsupported 100% (1.0) claims.
   - Inserted verified benchmark metrics and explicit class imbalance warning.

5. **[`pages/05_Model_Lab.py`](file:///c:/Users/anand/OneDrive/ExoMiner-main/ExoMiner/ExoplanetAI/pages/05_Model_Lab.py)**:
   - Completely redesigned with Deep Space Observatory theme.
   - Added prominent **Scientific Evaluation Transparency Disclosure** banner.
   - Replaced static confusion matrix with real measured matrix (`[[565, 0], [5, 0]]`).
   - Replaced mathematical polynomial ROC curve with real empirical Plotly curve.
   - Replaced fabricated 5-fold CV table with honest Cross-Validation Audit panel.
   - Added interactive **Live Model Evaluator** allowing users to upload custom labeled CSV datasets and inspect live confusion matrices and ROC curves.

6. **[`pages/01_Observatory.py`](file:///c:/Users/anand/OneDrive/ExoMiner-main/ExoMiner/ExoplanetAI/pages/01_Observatory.py)**:
   - Updated compact KPI card to show verified benchmark accuracy and ROC score rather than hardcoded 0.925.

7. **[`tests/test_model_evaluation.py`](file:///c:/Users/anand/OneDrive/ExoMiner-main/ExoMiner/ExoplanetAI/tests/test_model_evaluation.py)**:
   - Created comprehensive test suite with 10 test functions verifying dynamic derivation of metrics, ROC/PR calculation, single-class handling, zero-division handling, and absence of static demo strings.

---

## 7. Test Results & Verification

All tests were executed using the active Conda Python 3.11 environment:

```powershell
C:\Users\anand\anaconda3\envs\exoplnt_dl\python.exe -m pytest tests/ -v
```

### Execution Output:
```
============================= test session starts =============================
platform win32 -- Python 3.11.15, pytest-9.1.1, pluggy-1.7.0
rootdir: C:\Users\anand\OneDrive\ExoMiner-main\ExoMiner\ExoplanetAI
plugins: dash-3.0.4
collected 44 items

tests/test_audit_scientific_correctness.py::test_empty_input_rejected_with_validation_error PASSED [  2%]
tests/test_audit_scientific_correctness.py::test_multi_transit_duration_measures_single_transit PASSED [  4%]
tests/test_audit_scientific_correctness.py::test_white_noise_produces_zero_transit_depth PASSED [  6%]
tests/test_audit_scientific_correctness.py::test_model_failure_raises_model_inference_error_by_default PASSED [  9%]
tests/test_backend.py::test_preprocessing_nan_removal PASSED             [ 11%]
tests/test_backend.py::test_preprocessing_normalization PASSED           [ 13%]
tests/test_backend.py::test_feature_extraction_keys PASSED               [ 15%]
tests/test_backend.py::test_validation_manual_inputs PASSED              [ 18%]
tests/test_backend.py::test_classification_metrics PASSED                [ 20%]
tests/test_backend.py::test_confusion_matrix PASSED                      [ 22%]
tests/test_feature_engineering_scientific.py::test_01_clean_synthetic_transit PASSED [ 25%]
tests/test_feature_engineering_scientific.py::test_02_white_noise_without_transit PASSED [ 27%]
tests/test_feature_engineering_scientific.py::test_03_transit_with_extreme_outlier PASSED [ 29%]
tests/test_feature_engineering_scientific.py::test_04_multiple_separate_transits_duration PASSED [ 31%]
tests/test_feature_engineering_scientific.py::test_05_constant_flux PASSED [ 34%]
tests/test_feature_engineering_scientific.py::test_06_insufficient_samples PASSED [ 36%]
tests/test_feature_engineering_scientific.py::test_07_irregular_or_invalid_time PASSED [ 38%]
tests/test_feature_engineering_scientific.py::test_08_missing_and_non_finite_values PASSED [ 40%]
tests/test_feature_engineering_scientific.py::test_09_flux_normalization_and_auxiliary_consistency PASSED [ 43%]
tests/test_feature_engineering_scientific.py::test_10_units_and_period_failure_behavior PASSED [ 45%]
tests/test_model_evaluation.py::test_confusion_matrix_derived_from_actual_labels_and_predictions PASSED [ 47%]
tests/test_model_evaluation.py::test_metric_values_change_when_predictions_change PASSED [ 50%]
tests/test_model_evaluation.py::test_roc_data_based_on_actual_prediction_scores PASSED [ 52%]
tests/test_model_evaluation.py::test_pr_curve_data_based_on_actual_prediction_scores PASSED [ 54%]
tests/test_model_evaluation.py::test_single_class_dataset_handled_gracefully PASSED [ 56%]
tests/test_model_evaluation.py::test_zero_division_handling_when_no_positives_predicted PASSED [ 59%]
tests/test_model_evaluation.py::test_empty_or_mismatched_inputs_rejected PASSED [ 61%]
tests/test_model_evaluation.py::test_model_metadata_contains_no_unsupported_100_percent_claims PASSED [ 63%]
tests/test_model_evaluation.py::test_benchmark_evaluation_artifact_integrity PASSED [ 65%]
tests/test_model_evaluation.py::test_static_demonstration_values_removed_from_model_lab PASSED [ 68%]
tests/test_prediction.py::test_predictor_singleton PASSED                [ 70%]
tests/test_prediction.py::test_predict_single_exoplanet PASSED           [ 72%]
tests/test_prediction.py::test_predict_single_noise PASSED               [ 75%]
tests/test_prediction.py::test_predict_batch PASSED                      [ 77%]
tests/test_prediction.py::test_empty_dict_rejected PASSED                [ 79%]
tests/test_prediction.py::test_missing_required_features_rejected PASSED [ 81%]
tests/test_prediction.py::test_nan_values_rejected PASSED                [ 84%]
tests/test_prediction.py::test_infinite_values_rejected PASSED           [ 86%]
tests/test_prediction.py::test_non_numeric_values_rejected PASSED        [ 88%]
tests/test_prediction.py::test_physical_boundary_violations_rejected PASSED [ 90%]
tests/test_prediction.py::test_model_inference_failure_raises_model_inference_error PASSED [ 93%]
tests/test_prediction.py::test_explicit_opt_in_heuristic_mode PASSED     [ 95%]
tests/test_prediction.py::test_algebraic_auxiliary_derivation PASSED     [ 97%]
tests/test_prediction.py::test_predict_lightcurve_empty_arrays_rejected PASSED [100%]

======================= 44 passed, 12 warnings in 3.93s =======================
```

---

## 8. Remaining Limitations & Recommendations

1. **Model Weights Require Retraining (Deferred per Safety Constraints):**
   - Per safety guidelines for this task, `exoplanet_model.pkl` was **not** modified or retrained.
   - The classifier's current weights are ineffective on genuine Kepler light curves (0% Recall).
   - In a future retraining phase, the classifier should be refitted on verified Kepler DR25 and TESS TCE catalogs using the corrected Astropy BLS transit features, balanced class weights (`class_weight='balanced_subsample'`), and threshold calibration.
2. **Separation from NASA ExoMiner++:**
   - The lightweight Random Forest classifier remains completely separate from the deep learning NASA ExoMiner++ convolutional pipeline. The UI clearly communicates that ExoMiner++ is an independent NASA research architecture.
3. **Single-Sector Cadence Baseline:**
   - Multi-transit period detection requires observations spanning multiple orbital cycles. When analyzing single-sector snippets where BLS period recovery is unconfirmed, heuristic vetting should be used with explicit disclosure.
