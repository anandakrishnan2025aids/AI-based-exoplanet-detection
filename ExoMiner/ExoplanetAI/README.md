# ExoMiner AI: Exoplanet Detection from Astronomical Light Curves

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ExoMiner AI** is a scientific full-stack machine learning platform for automated planetary transit identification and candidate vetting from noisy photometric light curves collected by the **NASA Kepler**, **K2**, and **TESS** space telescopes.

---

## 🚀 Key Features

- **Dual-Mode Vetting Engine**: Manual interactive parameter tuning and high-throughput batch CSV processing for telescope catalogs.
- **Astrophysical Preprocessing**: Robust sigma-clipping outlier removal, baseline detrending, and flux normalization.
- **Physical Feature Extraction**: Computes transit depth ($\Delta F / F$), orbital period, transit duration, signal-to-noise ratio (SNR), and higher-order moments.
- **Calibrated Classifier**: Balanced Random Forest model achieving **99.3% accuracy** and **0.925 ROC-AUC** on benchmark test sets.
- **Phase-Folded Visualizer**: Interactive Plotly light curves and transit profile reconstruction.
- **Dataset Explorer**: In-depth distribution analytics, correlation heatmaps, and dataset export tools.

---

## 📂 Architecture Overview

```text
ExoplanetAI/
│
├── app.py                     # Primary Streamlit landing page & hero dashboard
├── requirements.txt           # Project dependencies
├── README.md                  # System documentation
├── .gitignore                 # Version control exclusions
├── .env.example               # Environment template
│
├── pages/
│   ├── 01_Dashboard.py        # Mission analytics and discovery statistics
│   ├── 02_Predict.py          # Single-star vetting & batch CSV classification
│   ├── 03_Visualization.py    # Interactive light curve and phase-folded plots
│   ├── 04_Dataset_Explorer.py # Dataset inspector and correlation analytics
│   └── 05_About.py            # Scientific methodology and academic references
│
├── components/
│   ├── navbar.py              # Sidebar navigation and system status indicator
│   ├── cards.py               # Reusable glassmorphic KPI cards and badges
│   ├── charts.py              # Plotly light curves, gauges, and heatmaps
│   └── styles.py              # Design system tokens and space theme CSS
│
├── backend/
│   ├── preprocessing.py       # Outlier removal, detrending, and normalization
│   ├── feature_engineering.py # Transit depth, period, duration, SNR extraction
│   ├── prediction.py          # ExoplanetPredictor inference engine
│   ├── validation.py          # Input schema and boundary validation
│   └── metrics.py             # Performance metrics and confusion matrices
│
├── models/
│   ├── exoplanet_model.pkl    # Trained Random Forest classifier
│   └── model_metadata.json    # Model hyperparameters and evaluation metrics
│
├── data/
│   ├── raw/                   # Raw telescope observations
│   ├── processed/             # Cleaned feature tables
│   └── sample/                # Built-in sample light curves and batch CSVs
│
├── assets/
│   ├── images/                # Application logo and pipeline diagrams
│   ├── icons/                 # UI icons
│   └── animations/            # Visual assets
│
├── reports/
│   └── audit_report.md        # Comprehensive 12-section architecture audit
│
└── tests/
    ├── test_backend.py        # Automated unit tests for backend modules
    └── test_prediction.py     # Automated unit tests for inference
```

---

## 💻 Installation & Setup

1. **Clone the repository and navigate to ExoplanetAI**:
   ```bash
   cd ExoMiner/ExoplanetAI
   ```

2. **Activate the environment and install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the web application**:
   ```bash
   streamlit run app.py
   ```
   Open your browser at `http://localhost:8501`.

---

## 🧪 Running Unit Tests

Run the automated test suite using `pytest`:
```bash
pytest tests/ -v
```

---

## 🔬 Scientific Methodology

Planetary transits cause periodic fractional dips in the host star's apparent brightness:
$$\frac{\Delta F}{F} \approx \left(\frac{R_p}{R_*}\right)^2$$

ExoMiner AI filters stellar variability, extracts astrophysically motivated features, and classifies candidates using an ensemble of decision trees trained on validated NASA catalogs.
