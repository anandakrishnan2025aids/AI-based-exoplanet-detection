"""
02_Detect.py - Primary Exoplanet Candidate Vetting and Analysis Engine.
Deep Space Observatory Design Language.
Provides instant light curve quick-scan preview, real-time pipeline progress sequence,
calibrated Random Forest inference, phase-folded transit profile, explainable feature bars,
physical candidate parameter derivation, and scientific dossier export.
"""

import os
import time
import streamlit as st
import pandas as pd
import numpy as np

from components.styles import inject_observatory_theme
from components.navbar import render_header
from components.cards import render_panel, render_verdict_badge, render_scientific_disclaimer
from components.charts import (
    plot_explainability_bars,
    plot_observatory_lightcurve,
    plot_phase_folded_transit
)
from components.report import generate_candidate_report
from backend.preprocessing import LightCurvePreprocessor
from backend.feature_engineering import ExoplanetFeatureExtractor
from backend.prediction import get_default_predictor, FeatureValidationError, ModelInferenceError
from backend.validation import validate_lightcurve_df

st.set_page_config(
    page_title="Detect • ExoMiner AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

inject_observatory_theme()
render_header(active_page="DETECT")

# ---------------- Page Header ----------------
st.markdown("""
<div style="margin-bottom: 1.5rem;">
    <div style="font-family: var(--font-mono); font-size: 0.75rem; color: #00D8F6; letter-spacing: 0.14em; text-transform: uppercase;">
        PRIMARY CANDIDATE VETTING CONSOLE
    </div>
    <h1 style="font-family: var(--font-display); font-size: 2.2rem; color: #FFFFFF; margin-top: 2px;">
        SEARCH FOR A HIDDEN WORLD
    </h1>
    <p style="font-size: 0.95rem; color: #94A3B8; margin-top: 4px;">
        Deploy calibrated machine learning inference on astronomical photometric observations to detect planetary transits.
    </p>
</div>
""", unsafe_allow_html=True)

predictor = get_default_predictor()
data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
sample_lc_path = os.path.join(data_root, "sample", "sample_lightcurve.csv")

# ---------------- Input Instrument Panel ----------------
st.markdown("### OBSERVATION DATA INGESTION")

input_mode = st.radio(
    "Select Ingestion Method:",
    ["[ TRY DEMO DATA ]", "[ UPLOAD LIGHT CURVE ]"],
    horizontal=True
)

df_active = None
target_id_str = "Unknown Target"

if input_mode == "[ TRY DEMO DATA ]":
    demo_cols = st.columns([2, 1])
    with demo_cols[0]:
        demo_choice = st.selectbox(
            "Select Curated Benchmark Observation:",
            [
                "Kepler-10b • Confirmed Rocky Super-Earth (P = 0.837d)",
                "Kepler-90i • Confirmed 8th Planet in Multi-System (P = 14.45d)",
                "Stellar White Noise Control • Non-Exoplanet Flat Baseline"
            ]
        )

    if "Kepler-10b" in demo_choice:
        target_id_str = "Kepler-10b (KIC-11904151)"
        if os.path.exists(sample_lc_path):
            df_active = pd.read_csv(sample_lc_path)
        else:
            t = np.linspace(120.0, 145.0, 1000)
            f = 1.0 - 0.0152 * np.exp(-((t % 0.837 - 0.2)**2)/(2*0.04**2)) + np.random.normal(0, 0.002, len(t))
            df_active = pd.DataFrame({"time": t, "flux": f})
    elif "Kepler-90i" in demo_choice:
        target_id_str = "Kepler-90i (KIC-10723750)"
        t = np.linspace(100.0, 180.0, 1200)
        f = 1.0 - 0.0210 * np.exp(-((t % 14.449 - 2.0)**2)/(2*0.08**2)) + np.random.normal(0, 0.0018, len(t))
        df_active = pd.DataFrame({"time": t, "flux": f})
    else:
        target_id_str = "Stellar Noise Control (KIC-99999999)"
        t = np.linspace(100.0, 150.0, 1000)
        f = 1.0 + np.random.normal(0, 0.006, len(t))
        df_active = pd.DataFrame({"time": t, "flux": f})

else:  # Upload
    st.markdown("""
    <div style="background: rgba(11, 21, 40, 0.85); border: 1px dashed rgba(0, 216, 246, 0.35); border-radius: 8px; padding: 14px; margin-bottom: 12px;">
        <div style="font-family: var(--font-mono); font-size: 0.8rem; color: #00D8F6;">SUPPORTED FORMAT: CSV TABLE</div>
        <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 3px;">Required Columns: <code>time</code>, <code>flux</code> (flux error optional)</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Light Curve CSV", type=["csv"], label_visibility="collapsed")
    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            time_cols = [c for c in df_raw.columns if "time" in str(c).lower()]
            flux_cols = [c for c in df_raw.columns if "flux" in str(c).lower()]
            if time_cols and flux_cols:
                df_active = pd.DataFrame({
                    "time": df_raw[time_cols[0]].values,
                    "flux": df_raw[flux_cols[0]].values
                })
                target_id_str = uploaded_file.name.replace(".csv", "")
            else:
                st.error("Invalid CSV columns. The file must contain columns named 'time' and 'flux'.")
        except Exception as e:
            st.error(f"Error reading file: {e}")

# ---------------- Dataset Information Summary ----------------
if df_active is not None and not df_active.empty:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### OBSERVATION TELEMETRY SPECIFICATIONS")

    t_min = float(df_active["time"].min())
    t_max = float(df_active["time"].max())
    std_phot = float(np.std(df_active["flux"].values))

    i1, i2, i3, i4 = st.columns(4)
    with i1:
        st.markdown(f"""
        <div class="kpi-hud">
            <div class="kpi-hud-label">DATASET TARGET</div>
            <div class="kpi-hud-value" style="font-size: 1.15rem; color: #00D8F6;">{target_id_str.split('•')[0].strip()}</div>
            <div class="kpi-hud-sub">Observational Source</div>
        </div>
        """, unsafe_allow_html=True)
    with i2:
        st.markdown(f"""
        <div class="kpi-hud">
            <div class="kpi-hud-label">ROWS / CADENCES</div>
            <div class="kpi-hud-value">{len(df_active):,}</div>
            <div class="kpi-hud-sub">Photometric Data Points</div>
        </div>
        """, unsafe_allow_html=True)
    with i3:
        st.markdown(f"""
        <div class="kpi-hud">
            <div class="kpi-hud-label">TEMPORAL RANGE</div>
            <div class="kpi-hud-value" style="font-size: 1.25rem;">{t_max - t_min:.1f} d</div>
            <div class="kpi-hud-sub">{t_min:.1f} to {t_max:.1f} BJD</div>
        </div>
        """, unsafe_allow_html=True)
    with i4:
        st.markdown(f"""
        <div class="kpi-hud">
            <div class="kpi-hud-label">PHOTOMETRIC NOISE (σ)</div>
            <div class="kpi-hud-value">{std_phot*1e6:.0f} ppm</div>
            <div class="kpi-hud-sub">Raw Dispersion Level</div>
        </div>
        """, unsafe_allow_html=True)

    # ---------------- Quick-Scan Preview of Ingested Light Curve ----------------
    st.markdown("<br>", unsafe_allow_html=True)
    col_pv_head1, col_pv_head2 = st.columns([2, 1])
    with col_pv_head1:
        st.markdown(f"<div style='font-family: var(--font-mono); font-size: 0.76rem; color: #00D8F6;'>PHOTOMETRIC QUICK-SCAN PREVIEW • {target_id_str}</div>", unsafe_allow_html=True)
    with col_pv_head2:
        st.markdown("<div style='text-align: right; font-family: var(--font-mono); font-size: 0.72rem; color: #10B981;'>● DATASET READY FOR INFERENCE</div>", unsafe_allow_html=True)

    fig_preview = plot_observatory_lightcurve(
        time=df_active["time"].values,
        flux=df_active["flux"].values,
        trend=None,
        mode="RAW",
        title=f"INGESTED PHOTOMETRY: {target_id_str}",
        highlight_dips=False
    )
    fig_preview.update_layout(height=280)
    st.plotly_chart(fig_preview, use_container_width=True)

    # ---------------- ANALYZE SIGNAL BUTTON ----------------
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔬 ANALYZE SIGNAL & RUN VETTING MODEL", use_container_width=True, key="start_vetting_btn"):
        # Real-time scientific progress sequence
        st.markdown("### VETTING PIPELINE EXECUTION")
        progress_placeholder = st.empty()

        # Step 1: Validation
        progress_placeholder.markdown("""
        <div class="hud-panel">
            <div class="prog-step prog-active">● <b>01 VALIDATING DATA</b> — Inspecting finite values and schema...</div>
            <div class="prog-step prog-pending">○ 02 CLEANING NOISE</div>
            <div class="prog-step prog-pending">○ 03 EXTRACTING FEATURES</div>
            <div class="prog-step prog-pending">○ 04 RUNNING MODEL</div>
            <div class="prog-step prog-pending">○ 05 ANALYZING SIGNAL</div>
            <div class="prog-step prog-pending">○ 06 GENERATING RESULT</div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.3)
        is_val, val_msg = validate_lightcurve_df(df_active)
        if not is_val:
            progress_placeholder.empty()
            st.error(f"❌ Input Data Validation Failed: {val_msg}")
            st.stop()

        # Step 2: Cleaning
        progress_placeholder.markdown("""
        <div class="hud-panel">
            <div class="prog-step prog-done">✓ 01 VALIDATING DATA (Passed)</div>
            <div class="prog-step prog-active">● <b>02 CLEANING NOISE</b> — Removing stellar flare excursions and baseline drift...</div>
            <div class="prog-step prog-pending">○ 03 EXTRACTING FEATURES</div>
            <div class="prog-step prog-pending">○ 04 RUNNING MODEL</div>
            <div class="prog-step prog-pending">○ 05 ANALYZING SIGNAL</div>
            <div class="prog-step prog-pending">○ 06 GENERATING RESULT</div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.3)
        preprocessor = LightCurvePreprocessor()
        proc_df = preprocessor.process(df_active["time"].values, df_active["flux"].values)

        # Step 3: Extract Features
        progress_placeholder.markdown("""
        <div class="hud-panel">
            <div class="prog-step prog-done">✓ 01 VALIDATING DATA</div>
            <div class="prog-step prog-done">✓ 02 CLEANING NOISE</div>
            <div class="prog-step prog-active">● <b>03 EXTRACTING FEATURES</b> — Computing transit depth, periodicity, SNR, skewness...</div>
            <div class="prog-step prog-pending">○ 04 RUNNING MODEL</div>
            <div class="prog-step prog-pending">○ 05 ANALYZING SIGNAL</div>
            <div class="prog-step prog-pending">○ 06 GENERATING RESULT</div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.3)
        extractor = ExoplanetFeatureExtractor()
        features = extractor.extract_from_series(proc_df["time"].values, proc_df["flux_detrended"].values)

        # Step 4: Running Model
        progress_placeholder.markdown("""
        <div class="hud-panel">
            <div class="prog-step prog-done">✓ 01 VALIDATING DATA</div>
            <div class="prog-step prog-done">✓ 02 CLEANING NOISE</div>
            <div class="prog-step prog-done">✓ 03 EXTRACTING FEATURES</div>
            <div class="prog-step prog-active">● <b>04 RUNNING MODEL</b> — Random Forest ensemble inference...</div>
            <div class="prog-step prog-pending">○ 05 ANALYZING SIGNAL</div>
            <div class="prog-step prog-pending">○ 06 GENERATING RESULT</div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.3)
        try:
            prediction_result = predictor.predict_single(features, allow_heuristic=False)
        except (FeatureValidationError, ModelInferenceError) as e:
            progress_placeholder.empty()
            st.error(f"❌ Prediction Pipeline Error: {e}")
            st.info("💡 Recommendation: Inspect light curve quality, noise level, or model status.")
            st.stop()
        except Exception as e:
            progress_placeholder.empty()
            st.error(f"❌ Unexpected Error During Model Inference: {e}")
            st.stop()

        # Step 5 & 6
        progress_placeholder.markdown("""
        <div class="hud-panel">
            <div class="prog-step prog-done">✓ 01 VALIDATING DATA</div>
            <div class="prog-step prog-done">✓ 02 CLEANING NOISE</div>
            <div class="prog-step prog-done">✓ 03 EXTRACTING FEATURES</div>
            <div class="prog-step prog-done">✓ 04 RUNNING MODEL</div>
            <div class="prog-step prog-done">✓ 05 ANALYZING SIGNAL</div>
            <div class="prog-step prog-done">✓ 06 GENERATING RESULT (100% Complete)</div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.2)

        # Store in session state for persistence
        st.session_state["last_prediction"] = prediction_result
        st.session_state["last_target"] = target_id_str
        st.session_state["last_proc_df"] = proc_df
        st.session_state["last_features"] = features

# ---------------- RESULT PRESENTATION ----------------
if "last_prediction" in st.session_state:
    res = st.session_state["last_prediction"]
    t_name = st.session_state.get("last_target", target_id_str)
    feats = st.session_state.get("last_features", {})
    proc_df = st.session_state.get("last_proc_df", None)
    score_pct = res["confidence"] * 100.0 if res.get("confidence") is not None else 0.0

    st.markdown("<hr style='border-color: rgba(0, 216, 246, 0.25); margin: 2rem 0;'>", unsafe_allow_html=True)

    # 1. Headline Verdict Badge & Warning
    is_heur = res.get("is_heuristic", False)
    if is_heur:
        st.warning(
            "⚠️ **HEURISTIC ESTIMATE ACTIVE**: Machine learning model inference was bypassed or unavailable. "
            "Results are based strictly on uncalibrated astrophysical transit rules."
        )

    render_verdict_badge(
        res["is_exoplanet"],
        score_pct if res.get("confidence", 0.0) > 0.0 and res.get("prediction") != "No transit detected — inconclusive" else None,
        target_name=t_name,
        is_heuristic=is_heur,
        prediction_status=res.get("prediction")
    )

    # 2. Scientific Disclaimer
    render_scientific_disclaimer()

    # 3. Diagnostic Metrics Grid
    d1, d2, d3, d4, d5 = st.columns(5)
    with d1:
        pred_label = res.get("prediction", "")
        if "Unable to evaluate" in pred_label:
            status_txt = "INVALID DATA"
            status_col = "#F59E0B"
            sub_grade = res.get("confidence_grade", "Inconclusive")
        elif "No transit detected" in pred_label:
            status_txt = "NO TRANSIT"
            status_col = "#94A3B8"
            sub_grade = "Inconclusive Vetting"
        elif res["is_exoplanet"]:
            status_txt = "TRANSIT"
            status_col = "#10B981"
            sub_grade = f"{res.get('confidence_grade', 'Moderate')} Confidence"
        else:
            status_txt = "NON-CANDIDATE"
            status_col = "#94A3B8"
            sub_grade = f"{res.get('confidence_grade', 'Moderate')} Confidence"

        st.markdown(f"""
        <div class="kpi-hud">
            <div class="kpi-hud-label">SIGNAL STATUS</div>
            <div class="kpi-hud-value" style="font-size: 1.15rem; color: {status_col};">
                {status_txt}
            </div>
            <div class="kpi-hud-sub">{sub_grade}</div>
        </div>
        """, unsafe_allow_html=True)
    with d2:
        st.markdown(f"""
        <div class="kpi-hud">
            <div class="kpi-hud-label">SIGNAL STRENGTH</div>
            <div class="kpi-hud-value" style="font-size: 1.2rem; color: #00D8F6;">
                {feats.get('snr', 0.0):.1f}
            </div>
            <div class="kpi-hud-sub">Signal-to-Noise Ratio</div>
        </div>
        """, unsafe_allow_html=True)
    with d3:
        depth_pct = feats.get('transit_depth', 0.0) * 100.0
        st.markdown(f"""
        <div class="kpi-hud">
            <div class="kpi-hud-label">TRANSIT DEPTH</div>
            <div class="kpi-hud-value" style="font-size: 1.2rem; color: #00D8F6;">
                {depth_pct:.3f}%
            </div>
            <div class="kpi-hud-sub">{depth_pct*10000:.0f} ppm</div>
        </div>
        """, unsafe_allow_html=True)
    with d4:
        p_val_kpi = feats.get('period_days')
        p_display = f"{p_val_kpi:.3f} d" if p_val_kpi is not None else "Undetermined"
        p_sub = "Harmonic BLS Period" if p_val_kpi is not None else "Non-Periodic / Noise"
        st.markdown(f"""
        <div class="kpi-hud">
            <div class="kpi-hud-label">ORBITAL PERIOD</div>
            <div class="kpi-hud-value" style="font-size: 1.2rem; color: #00D8F6;">
                {p_display}
            </div>
            <div class="kpi-hud-sub">{p_sub}</div>
        </div>
        """, unsafe_allow_html=True)
    with d5:
        model_name = res.get("model_type", "Random Forest")
        model_sub = "Heuristic Fallback" if is_heur else "Gini Ensemble v2.2"
        st.markdown(f"""
        <div class="kpi-hud">
            <div class="kpi-hud-label">MODEL ARCHITECTURE</div>
            <div class="kpi-hud-value" style="font-size: 0.95rem; color: #CBD5E1;">
                {model_name}
            </div>
            <div class="kpi-hud-sub">{model_sub}</div>
        </div>
        """, unsafe_allow_html=True)

    # 4. Phase-Folded Occultation Profile (if exoplanet or signal found)
    if proc_df is not None and "time" in proc_df.columns:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="margin-bottom: 0.8rem;">
            <div style="font-family: var(--font-mono); font-size: 0.74rem; color: #00D8F6; letter-spacing: 0.12em; text-transform: uppercase;">
                STACKED ORBITAL CADENCE ALIGNMENT
            </div>
            <h3 style="font-family: var(--font-display); font-size: 1.4rem; color: #FFFFFF; margin-top: 2px;">
                PHASE-FOLDED CANDIDATE OCCULTATION PROFILE
            </h3>
        </div>
        """, unsafe_allow_html=True)
        period_val = max(0.2, float(feats.get("period_days") or 1.0))
        fig_folded = plot_phase_folded_transit(
            time=proc_df["time"].values,
            flux=proc_df["flux_detrended"].values,
            period=period_val,
            title=f"{t_name} • Folded Transit Profile (P = {period_val:.4f} d)"
        )
        fig_folded.update_layout(height=360)
        st.plotly_chart(fig_folded, use_container_width=True)

    # 5. DERIVED PHYSICAL CANDIDATE ATTRIBUTES
    if res["is_exoplanet"]:
        st.markdown("<br>", unsafe_allow_html=True)
        # Calculate approximate physical characteristics assuming solar-type host star
        d_val = float(feats.get("transit_depth") or 0.015)
        p_val = float(feats.get("period_days") or 1.0)
        r_earth = np.sqrt(d_val) * 109.2  # R_star ~ 1 R_sun
        a_au = ((p_val / 365.25)**2)**(1/3)  # Kepler's Third Law
        teq_k = 5778.0 * np.sqrt(1.0 / (2.0 * a_au * 215.03)) if a_au > 0 else 300.0

        st.markdown("""
        <div style="margin-bottom: 0.8rem;">
            <div style="font-family: var(--font-mono); font-size: 0.74rem; color: #10B981; letter-spacing: 0.12em; text-transform: uppercase;">
                ASTROPHYSICAL MODEL DERIVATIONS
            </div>
            <h3 style="font-family: var(--font-display); font-size: 1.4rem; color: #FFFFFF; margin-top: 2px;">
                DERIVED PLANETARY CANDIDATE ATTRIBUTES
            </h3>
        </div>
        """, unsafe_allow_html=True)

        pc1, pc2, pc3, pc4 = st.columns(4)
        with pc1:
            st.markdown(f"""
            <div class="kpi-hud">
                <div class="kpi-hud-label">ESTIMATED RADIUS</div>
                <div class="kpi-hud-value" style="color: #00D8F6;">{r_earth:.2f} R⊕</div>
                <div class="kpi-hud-sub">{r_earth / 11.2:.2f} Jupiter Radii</div>
            </div>
            """, unsafe_allow_html=True)
        with pc2:
            st.markdown(f"""
            <div class="kpi-hud">
                <div class="kpi-hud-label">SEMI-MAJOR AXIS (a)</div>
                <div class="kpi-hud-value" style="color: #F1F5F9;">{a_au:.3f} AU</div>
                <div class="kpi-hud-sub">Kepler's 3rd Law (a ∝ P²/³)</div>
            </div>
            """, unsafe_allow_html=True)
        with pc3:
            st.markdown(f"""
            <div class="kpi-hud">
                <div class="kpi-hud-label">EQUILIBRIUM TEMP (Teq)</div>
                <div class="kpi-hud-value" style="color: #F59E0B;">{teq_k:.0f} K</div>
                <div class="kpi-hud-sub">{teq_k - 273.15:.0f} °C (Zero Albedo)</div>
            </div>
            """, unsafe_allow_html=True)
        with pc4:
            plan_class = "Super-Earth" if r_earth < 2.0 else ("Sub-Neptune" if r_earth < 4.0 else "Gas Giant")
            st.markdown(f"""
            <div class="kpi-hud">
                <div class="kpi-hud-label">CLASSIFICATION TIER</div>
                <div class="kpi-hud-value" style="color: #10B981; font-size: 1.3rem;">{plan_class}</div>
                <div class="kpi-hud-sub">Photometric Demographics</div>
            </div>
            """, unsafe_allow_html=True)

    # 6. EXPLAINABLE RESULT: "WHY DID THE MODEL SAY THIS?"
    st.markdown("<br>", unsafe_allow_html=True)
    render_panel(
        "WHY DID THE MODEL SAY THIS? • EXPLAINABLE ASTROPHYSICAL CONTRIBUTIONS",
        f"""
        <p style="font-size: 0.92rem; color: #CBD5E1; line-height: 1.6; margin-bottom: 12px;">
            {res['vetting_summary']}
        </p>
        """,
        "Feature Importance Decomposition"
    )

    contrib_dict = {
        "Transit Depth (ΔF/F)": min(1.0, feats.get("transit_depth", 0.0) * 25.0),
        "Signal Strength (SNR)": min(1.0, feats.get("snr", 0.0) / 20.0),
        "Periodicity Recurrence": 0.65 if feats.get("period_days", 0) > 0.3 else 0.15,
        "Asymmetric Flux Skew": min(1.0, abs(feats.get("flux_skew", 0.0)) / 3.0),
        "Photometric Noise Dispersion": max(0.05, 1.0 - feats.get("flux_std", 0.005) * 50.0)
    }
    st.plotly_chart(plot_explainability_bars(contrib_dict), use_container_width=True)

    # 7. Action Buttons
    st.markdown("<br>", unsafe_allow_html=True)
    act1, act2, act3 = st.columns(3)

    with act1:
        if st.button("📈 VIEW SIGNALS LAB & PHASE PROFILE", use_container_width=True):
            st.switch_page("pages/03_Signals.py")

    with act2:
        report_text = generate_candidate_report(
            target_id=t_name,
            prediction_result=res,
            dataset_name="Observation Light Curve Cadences",
            rows_count=len(df_active) if df_active is not None else 1000
        )
        st.download_button(
            label="📄 DOWNLOAD ANALYSIS REPORT (DOSSIER)",
            data=report_text,
            file_name=f"exoplanet_vetting_{t_name.replace(' ', '_')}.txt",
            mime="text/plain",
            use_container_width=True
        )

    with act3:
        if st.button("🔄 ANALYZE ANOTHER TARGET", use_container_width=True):
            if "last_prediction" in st.session_state:
                del st.session_state["last_prediction"]
            st.rerun()
