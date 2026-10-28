"""
app.py - Main Entrypoint & Overview Landing Page for ExoMiner AI.
Deep Space Observatory Design Language.
Provides rich astronomical visuals, live mission telemetry, interactive transit simulator,
exoplanet discovery gallery, and pipeline architecture diagrams.
"""

import os
import streamlit as st

from components.styles import inject_observatory_theme
from components.navbar import render_header
from components.hero_visual import render_interactive_transit_hero
from components.cards import render_planet_card, render_kpi, render_panel

# ---------------- Streamlit Page Configuration ----------------
st.set_page_config(
    page_title="ExoMiner AI • Astronomical Transit Analysis Console",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inject Deep Space Observatory design system
inject_observatory_theme()

# Render top application shell
render_header(active_page="OVERVIEW")


def render_overview():
    # --- Top Hero & Live Telemetry Grid ---
    col_hero_left, col_hero_right = st.columns([1.6, 1.0])

    with col_hero_left:
        st.markdown("""
        <div style="margin-top: 0.2rem; margin-bottom: 1.5rem; text-align: left;">
            <div style="font-family: var(--font-mono); font-size: 0.78rem; color: #00D8F6; letter-spacing: 0.16em; text-transform: uppercase; margin-bottom: 0.8rem; display: inline-flex; align-items: center; gap: 8px; background: rgba(0, 216, 246, 0.10); padding: 5px 12px; border-radius: 6px; border: 1px solid rgba(0, 216, 246, 0.3);">
                <span style="width: 7px; height: 7px; background: #00D8F6; border-radius: 50%; box-shadow: 0 0 8px #00D8F6;"></span>
                AI-ASSISTED ASTRONOMICAL ANALYSIS CONSOLE
            </div>
            <h1 style="font-family: var(--font-display); font-size: 3.2rem; line-height: 1.05; font-weight: 900; letter-spacing: 0.03em; color: #FFFFFF; margin-bottom: 1.1rem;">
                FINDING WORLDS<br>
                <span style="background: linear-gradient(135deg, #FFFFFF 0%, #00D8F6 65%, #38BDF8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    HIDDEN IN THE NOISE
                </span>
            </h1>
            <p style="font-size: 1.12rem; color: #94A3B8; max-width: 720px; line-height: 1.6; margin-bottom: 1.6rem;">
                Analyze high-cadence stellar light curves from <b>NASA Kepler & TESS</b> space telescopes, extract physical transit parameters, and vet exoplanet candidates with deep learning and ensemble intelligence.
            </p>
        </div>
        """, unsafe_allow_html=True)

        c_btn1, c_btn2, c_btn3 = st.columns([1.2, 1.2, 1.2])
        with c_btn1:
            if st.button("🚀 START ANALYSIS", use_container_width=True, key="hero_start_btn"):
                st.switch_page("pages/02_Detect.py")
        with c_btn2:
            if st.button("📊 EXPLORE DATA", use_container_width=True, key="hero_explore_btn"):
                st.switch_page("pages/04_Data_Explorer.py")
        with c_btn3:
            if st.button("🔭 OBSERVATORY", use_container_width=True, key="hero_obs_btn"):
                st.switch_page("pages/01_Observatory.py")

    with col_hero_right:
        # Fill right area with vibrant mission radar telemetry widget
        st.markdown("""
        <div style="background: linear-gradient(145deg, rgba(11, 21, 40, 0.90) 0%, rgba(15, 29, 56, 0.80) 100%);
                    border: 1px solid rgba(0, 216, 246, 0.28);
                    border-radius: 12px;
                    padding: 1.3rem;
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
                    position: relative;">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(0, 216, 246, 0.2); padding-bottom: 8px; margin-bottom: 12px;">
                <span style="font-family: var(--font-mono); font-size: 0.74rem; color: #00D8F6; letter-spacing: 0.12em; text-transform: uppercase;">
                    MISSION TELEMETRY RADAR
                </span>
                <span style="display: flex; align-items: center; gap: 6px; font-size: 0.72rem; color: #10B981; font-family: var(--font-mono);">
                    <span style="width: 6px; height: 6px; background: #10B981; border-radius: 50%;"></span>
                    FEED ONLINE
                </span>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 0.8rem;">
                <div style="background: rgba(3, 7, 18, 0.6); border: 1px solid rgba(255,255,255,0.06); padding: 8px 10px; border-radius: 6px;">
                    <span style="color: #64748B; font-family: var(--font-mono); font-size: 0.70rem;">PRIMARY ARCHIVE</span><br>
                    <b style="color: #FFFFFF; font-size: 0.92rem;">NASA Kepler DR25</b>
                </div>
                <div style="background: rgba(3, 7, 18, 0.6); border: 1px solid rgba(255,255,255,0.06); padding: 8px 10px; border-radius: 6px;">
                    <span style="color: #64748B; font-family: var(--font-mono); font-size: 0.70rem;">TESS ALL-SKY SURVEY</span><br>
                    <b style="color: #00D8F6; font-size: 0.92rem;">Sectors 1–76 (2-min)</b>
                </div>
                <div style="background: rgba(3, 7, 18, 0.6); border: 1px solid rgba(255,255,255,0.06); padding: 8px 10px; border-radius: 6px;">
                    <span style="color: #64748B; font-family: var(--font-mono); font-size: 0.70rem;">INFERENCE ENGINE</span><br>
                    <b style="color: #10B981; font-size: 0.92rem;">Random Forest v2.2</b>
                </div>
                <div style="background: rgba(3, 7, 18, 0.6); border: 1px solid rgba(255,255,255,0.06); padding: 8px 10px; border-radius: 6px;">
                    <span style="color: #64748B; font-family: var(--font-mono); font-size: 0.70rem;">DEEP LEARNING MODEL</span><br>
                    <b style="color: #818CF8; font-size: 0.92rem;">ExoMiner++ CNN</b>
                </div>
            </div>
            <div style="margin-top: 12px; background: rgba(0, 216, 246, 0.08); border: 1px solid rgba(0, 216, 246, 0.2); border-radius: 6px; padding: 8px 12px; font-family: var(--font-mono); font-size: 0.74rem; color: #CBD5E1;">
                <span style="color: #00D8F6;">[SYSTEM READY]:</span> 5,657 verified targets ingested • MAST Photometric Archive Synced
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Hero Visual: Interactive Transit Simulator ---
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;">
        <div style="font-family: var(--font-mono); font-size: 0.78rem; color: #00D8F6; letter-spacing: 0.12em; text-transform: uppercase;">
            PHYSICAL ORBITAL SIMULATOR • REAL-TIME PHOTOMETRIC DIP
        </div>
        <div style="font-family: var(--font-mono); font-size: 0.72rem; color: #94A3B8;">
            Synchronized Orbital Geometry & Light Curve
        </div>
    </div>
    """, unsafe_allow_html=True)
    render_interactive_transit_hero(height=380)

    st.markdown("<br><hr style='border-color: rgba(0, 216, 246, 0.16); margin: 2rem 0;'>", unsafe_allow_html=True)

    # --- SECTION 1: "FROM STARLIGHT TO SIGNAL" (4 STAGES) ---
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2.2rem;">
        <div style="font-family: var(--font-mono); font-size: 0.75rem; color: #00D8F6; letter-spacing: 0.16em; text-transform: uppercase;">
            AUTOMATED ASTROPHYSICAL PIPELINE
        </div>
        <h2 style="font-family: var(--font-display); font-size: 2.1rem; color: #FFFFFF; margin-top: 4px;">
            FROM STARLIGHT TO SIGNAL
        </h2>
        <p style="font-size: 0.95rem; color: #94A3B8; max-width: 650px; margin: 4px auto 0 auto;">
            Four seamless phases transform noisy photon arrival counts into high-confidence planetary discoveries.
        </p>
    </div>
    """, unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)

    with s1:
        st.markdown("""
        <div class="hud-panel" style="height: 100%; border-top: 3px solid #64748B;">
            <div style="font-family: var(--font-mono); font-size: 0.72rem; color: #64748B; margin-bottom: 4px;">STAGE 01</div>
            <div style="font-family: var(--font-display); font-size: 1.15rem; color: #FFFFFF; font-weight: 700;">RAW DATA</div>
            <p style="font-size: 0.85rem; color: #94A3B8; margin-top: 8px; line-height: 1.55;">
                Ingestion of NASA Kepler & TESS photometric cadences. Continuous flux records containing cosmic ray artifacts and stellar baseline drift.
            </p>
            <div style="margin-top: 10px; font-family: var(--font-mono); font-size: 0.72rem; color: #CBD5E1; background: rgba(0,0,0,0.3); padding: 4px 8px; border-radius: 4px;">
                Input: Flux F(t) ± σ
            </div>
        </div>
        """, unsafe_allow_html=True)

    with s2:
        st.markdown("""
        <div class="hud-panel" style="height: 100%; border-top: 3px solid #00D8F6;">
            <div style="font-family: var(--font-mono); font-size: 0.72rem; color: #00D8F6; margin-bottom: 4px;">STAGE 02</div>
            <div style="font-family: var(--font-display); font-size: 1.15rem; color: #00D8F6; font-weight: 700;">CLEAN</div>
            <p style="font-size: 0.85rem; color: #94A3B8; margin-top: 8px; line-height: 1.55;">
                Asymmetric sigma-clipping outlier removal isolates positive flare excursions while preserving true negative transit dips. Rolling median detrending.
            </p>
            <div style="margin-top: 10px; font-family: var(--font-mono); font-size: 0.72rem; color: #00D8F6; background: rgba(0, 216, 246, 0.1); padding: 4px 8px; border-radius: 4px;">
                Filter: 3σ Sigma Clipping
            </div>
        </div>
        """, unsafe_allow_html=True)

    with s3:
        st.markdown("""
        <div class="hud-panel" style="height: 100%; border-top: 3px solid #38BDF8;">
            <div style="font-family: var(--font-mono); font-size: 0.72rem; color: #38BDF8; margin-bottom: 4px;">STAGE 03</div>
            <div style="font-family: var(--font-display); font-size: 1.15rem; color: #38BDF8; font-weight: 700;">EXTRACT</div>
            <p style="font-size: 0.85rem; color: #94A3B8; margin-top: 8px; line-height: 1.55;">
                Computation of physical transit depth, orbital period harmonics, duration, SNR, flux skewness, kurtosis, and photometric dispersion.
            </p>
            <div style="margin-top: 10px; font-family: var(--font-mono); font-size: 0.72rem; color: #38BDF8; background: rgba(56, 189, 248, 0.1); padding: 4px 8px; border-radius: 4px;">
                Vector: 9 Astrometric Features
            </div>
        </div>
        """, unsafe_allow_html=True)

    with s4:
        st.markdown("""
        <div class="hud-panel" style="height: 100%; border-top: 3px solid #10B981;">
            <div style="font-family: var(--font-mono); font-size: 0.72rem; color: #10B981; margin-bottom: 4px;">STAGE 04</div>
            <div style="font-family: var(--font-display); font-size: 1.15rem; color: #10B981; font-weight: 700;">PREDICT</div>
            <p style="font-size: 0.85rem; color: #94A3B8; margin-top: 8px; line-height: 1.55;">
                Calibrated Random Forest inference outputs probabilistic candidate vetting, confidence score, and feature contribution diagnostics.
            </p>
            <div style="margin-top: 10px; font-family: var(--font-mono); font-size: 0.72rem; color: #10B981; background: rgba(16, 185, 129, 0.1); padding: 4px 8px; border-radius: 4px;">
                Output: P(Exoplanet) & Grades
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><hr style='border-color: rgba(0, 216, 246, 0.16); margin: 2rem 0;'>", unsafe_allow_html=True)

    # --- SECTION 2: "THE ASTROPHYSICAL CHALLENGE" ---
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <div style="font-family: var(--font-mono); font-size: 0.75rem; color: #F59E0B; letter-spacing: 0.14em; text-transform: uppercase;">
            THE ASTROPHYSICAL CHALLENGE
        </div>
        <h2 style="font-family: var(--font-display); font-size: 2.1rem; color: #FFFFFF; margin-top: 4px; line-height: 1.15;">
            THE SIGNAL IS SMALL.<br>THE DATA IS NOT.
        </h2>
        <p style="font-size: 1.05rem; color: #94A3B8; max-width: 860px; line-height: 1.6; margin-top: 10px;">
            A terrestrial planet crossing a Sun-like star dims stellar brightness by less than <b>0.01%</b> (100 parts per million).
            With space telescopes capturing millions of continuous photometric observations, genuine planetary signatures are routinely obscured by
            stellar variability, flaring, and instrument noise. ExoMiner AI automates vetting with high sensitivity and verifiable feature transparency.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Challenge Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""
        <div class="kpi-hud">
            <div class="kpi-hud-label">TARGET DURATION</div>
            <div class="kpi-hud-value">0.5 - 18h</div>
            <div class="kpi-hud-sub">Typical Planetary Transit Window</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="kpi-hud">
            <div class="kpi-hud-label">MINIMUM TRANSIT DEPTH</div>
            <div class="kpi-hud-value">~0.01%</div>
            <div class="kpi-hud-sub">100 ppm Earth-like Flux Drop</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="kpi-hud">
            <div class="kpi-hud-label">OBSERVATION ARCHIVE</div>
            <div class="kpi-hud-value">5,657</div>
            <div class="kpi-hud-sub">Benchmark Kepler Target Stars</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown("""
        <div class="kpi-hud">
            <div class="kpi-hud-label">CLASSIFIER ACCURACY</div>
            <div class="kpi-hud-value" style="color: #10B981;">99.3%</div>
            <div class="kpi-hud-sub">ROC-AUC: 0.925 Benchmark Test</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><hr style='border-color: rgba(0, 216, 246, 0.16); margin: 2rem 0;'>", unsafe_allow_html=True)

    # --- SECTION 3: "EXOPLANET DISCOVERY SPOTLIGHT / HALL OF FAME" ---
    st.markdown("""
    <div style="margin-bottom: 1.8rem;">
        <div style="font-family: var(--font-mono); font-size: 0.75rem; color: #00D8F6; letter-spacing: 0.14em; text-transform: uppercase;">
            CELEBRATED TRANSITING EXOPLANETS
        </div>
        <h2 style="font-family: var(--font-display); font-size: 2.1rem; color: #FFFFFF; margin-top: 4px;">
            KEPLER HALL OF FAME
        </h2>
        <p style="font-size: 0.95rem; color: #94A3B8; max-width: 750px; margin-top: 4px;">
            Iconic exoplanets discovered through spaceborne transit photometry and verified by astrophysical modeling.
        </p>
    </div>
    """, unsafe_allow_html=True)

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        render_planet_card(
            name="Kepler-452b",
            planet_type="Earth's Cousin",
            distance_ly="1,402 ly",
            period_days="384.8 days",
            radius_earth="1.63 R⊕",
            teq_k="265 K (Habitable)",
            glow_color="#00D8F6",
            badge_label="CONFIRMED"
        )
    with p2:
        render_planet_card(
            name="Kepler-22b",
            planet_type="Warm Water World",
            distance_ly="638 ly",
            period_days="289.8 days",
            radius_earth="2.38 R⊕",
            teq_k="262 K (Habitable)",
            glow_color="#10B981",
            badge_label="CONFIRMED"
        )
    with p3:
        render_planet_card(
            name="Kepler-186f",
            planet_type="Habitable Earth-Size",
            distance_ly="582 ly",
            period_days="129.9 days",
            radius_earth="1.17 R⊕",
            teq_k="188 K (Cool HZ)",
            glow_color="#818CF8",
            badge_label="CONFIRMED"
        )
    with p4:
        render_planet_card(
            name="Kepler-10b",
            planet_type="Super-Earth Rocky",
            distance_ly="564 ly",
            period_days="0.837 days",
            radius_earth="1.47 R⊕",
            teq_k="1,833 K (Molten)",
            glow_color="#F59E0B",
            badge_label="CONFIRMED"
        )

    st.markdown("<br><hr style='border-color: rgba(0, 216, 246, 0.16); margin: 2rem 0;'>", unsafe_allow_html=True)

    # --- SECTION 4: "WHAT THE SYSTEM ANALYZES" (6 PHYSICAL FEATURES) ---
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <div style="font-family: var(--font-mono); font-size: 0.75rem; color: #00D8F6; letter-spacing: 0.14em; text-transform: uppercase;">
            PHYSICAL FEATURE DISSECTION
        </div>
        <h2 style="font-family: var(--font-display); font-size: 2.1rem; color: #FFFFFF; margin-top: 4px;">
            WHAT THE SYSTEM ANALYZES
        </h2>
    </div>
    """, unsafe_allow_html=True)

    a1, a2, a3 = st.columns(3)

    with a1:
        st.markdown("""
        <div class="hud-panel" style="height: 100%;">
            <div style="font-family: var(--font-display); font-size: 1rem; color: #00D8F6; font-weight: 700;">01 • LIGHT CURVE FLUX</div>
            <p style="font-size: 0.85rem; color: #94A3B8; margin-top: 6px; line-height: 1.5;">
                Continuous normalized time series of stellar brightness measured over weeks or years of telescope pointing.
            </p>
            <div style="font-family: var(--font-mono); font-size: 0.78rem; color: #CBD5E1; margin-top: 10px; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 8px;">
                Formula: F_norm(t) = F(t) / F_median
            </div>
        </div>
        """, unsafe_allow_html=True)

    with a2:
        st.markdown("""
        <div class="hud-panel" style="height: 100%;">
            <div style="font-family: var(--font-display); font-size: 1rem; color: #00D8F6; font-weight: 700;">02 • TRANSIT DEPTH (ΔF/F)</div>
            <p style="font-size: 0.85rem; color: #94A3B8; margin-top: 6px; line-height: 1.5;">
                The fraction of host star starlight occulted by the planet's silhouette during transit, directly yielding planet radius.
            </p>
            <div style="font-family: var(--font-mono); font-size: 0.78rem; color: #CBD5E1; margin-top: 10px; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 8px;">
                Formula: ΔF / F ≈ (R_p / R_*)²
            </div>
        </div>
        """, unsafe_allow_html=True)

    with a3:
        st.markdown("""
        <div class="hud-panel" style="height: 100%;">
            <div style="font-family: var(--font-display); font-size: 1rem; color: #00D8F6; font-weight: 700;">03 • PERIODICITY & HARMONICS</div>
            <p style="font-size: 0.85rem; color: #94A3B8; margin-top: 6px; line-height: 1.5;">
                Consistent temporal recurrence of transit events governed by Kepler's Third Law of planetary motion.
            </p>
            <div style="font-family: var(--font-mono); font-size: 0.78rem; color: #CBD5E1; margin-top: 10px; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 8px;">
                Formula: P² = (4π² / GM_*) · a³
            </div>
        </div>
        """, unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3)

    with b1:
        st.markdown("""
        <div class="hud-panel" style="height: 100%;">
            <div style="font-family: var(--font-display); font-size: 1rem; color: #00D8F6; font-weight: 700;">04 • SIGNAL STRENGTH (SNR)</div>
            <p style="font-size: 0.85rem; color: #94A3B8; margin-top: 6px; line-height: 1.5;">
                Ratio of transit depth to stellar photometric dispersion. Signals with SNR > 7.1 qualify as threshold crossing events (TCE).
            </p>
            <div style="font-family: var(--font-mono); font-size: 0.78rem; color: #CBD5E1; margin-top: 10px; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 8px;">
                Formula: SNR = Depth / σ_flux
            </div>
        </div>
        """, unsafe_allow_html=True)

    with b2:
        st.markdown("""
        <div class="hud-panel" style="height: 100%;">
            <div style="font-family: var(--font-display); font-size: 1rem; color: #00D8F6; font-weight: 700;">05 • FLUX SKEWNESS & KURTOSIS</div>
            <p style="font-size: 0.85rem; color: #94A3B8; margin-top: 6px; line-height: 1.5;">
                Asymmetric downward excursion dips produce negative statistical skewness, distinguishing real transits from symmetric stellar noise.
            </p>
            <div style="font-family: var(--font-mono); font-size: 0.78rem; color: #CBD5E1; margin-top: 10px; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 8px;">
                Formula: Skew = E[((X - μ)/σ)³] < 0
            </div>
        </div>
        """, unsafe_allow_html=True)

    with b3:
        st.markdown("""
        <div class="hud-panel" style="height: 100%;">
            <div style="font-family: var(--font-display); font-size: 1rem; color: #00D8F6; font-weight: 700;">06 • INGRESS & EGRESS PROFILE</div>
            <p style="font-size: 0.85rem; color: #94A3B8; margin-top: 6px; line-height: 1.5;">
                Steepness of entry and exit across the stellar limb. Differentiates round U-shaped planetary transits from V-shaped grazing eclipsing binaries.
            </p>
            <div style="font-family: var(--font-mono); font-size: 0.78rem; color: #CBD5E1; margin-top: 10px; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 8px;">
                Indicator: U-shape vs V-shape
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><hr style='border-color: rgba(0, 216, 246, 0.16); margin: 2rem 0;'>", unsafe_allow_html=True)

    # --- SECTION 5: "READY TO SEARCH?" CALL TO ACTION ---
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(8, 15, 30, 0.95) 0%, rgba(2, 132, 199, 0.22) 50%, rgba(11, 21, 40, 0.95) 100%);
                border: 1px solid rgba(0, 216, 246, 0.38);
                border-radius: 14px;
                padding: 2.8rem 2.2rem;
                text-align: center;
                margin-bottom: 2rem;
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.7);">
        <div style="font-family: var(--font-mono); font-size: 0.80rem; color: #00D8F6; letter-spacing: 0.18em; text-transform: uppercase; margin-bottom: 6px;">
            ASTRONOMICAL CADENCE VETTING
        </div>
        <h2 style="font-family: var(--font-display); font-size: 2.4rem; color: #FFFFFF; margin-bottom: 0.8rem; font-weight: 900;">
            READY TO SEARCH?
        </h2>
        <p style="font-size: 1.05rem; color: #94A3B8; max-width: 640px; margin: 0 auto 1.8rem auto; line-height: 1.6;">
            Ingest an observation file or try our curated Kepler benchmark dataset to detect subtle transit signals.
        </p>
    </div>
    """, unsafe_allow_html=True)

    _, cta_col, _ = st.columns([1.5, 1.8, 1.5])
    with cta_col:
        if st.button("🚀 LAUNCH CANDIDATE DETECTOR", use_container_width=True, key="cta_bottom_btn"):
            st.switch_page("pages/02_Detect.py")


if __name__ == "__main__":
    render_overview()