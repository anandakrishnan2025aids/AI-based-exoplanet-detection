"""
06_About.py - Project Background, Astrophysical Narrative, and System Architecture.
Deep Space Observatory Design Language.
Provides scientific context, mission timeline (Kepler -> TESS -> JWST),
astrophysics formulae cheatsheet, research citations, and engineering specifications.
"""

import streamlit as st

from components.styles import inject_observatory_theme
from components.navbar import render_header
from components.cards import render_panel, render_scientific_disclaimer

st.set_page_config(
    page_title="About & Science • ExoMiner AI",
    page_icon="ℹ️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

inject_observatory_theme()
render_header(active_page="ABOUT")

# ---------------- Page Header ----------------
st.markdown("""
<div style="margin-bottom: 2rem;">
    <div style="font-family: var(--font-mono); font-size: 0.75rem; color: #00D8F6; letter-spacing: 0.14em; text-transform: uppercase;">
        PROJECT ARCHITECTURE & SCIENTIFIC NARRATIVE
    </div>
    <h1 style="font-family: var(--font-display); font-size: 2.3rem; color: #FFFFFF; margin-top: 2px;">
        FINDING WORLDS BEYOND OUR SUN
    </h1>
    <p style="font-size: 1.05rem; color: #94A3B8; max-width: 820px; margin-top: 4px; line-height: 1.6;">
        How statistical learning and astronomical transit photometry collaborate to discover and vet exoplanetary candidates.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- THE QUESTION ----------------
st.markdown("""
<div style="background: linear-gradient(135deg, rgba(11, 21, 40, 0.90) 0%, rgba(15, 29, 56, 0.80) 100%);
            border-left: 4px solid #00D8F6;
            border-radius: 0 10px 10px 0;
            padding: 1.8rem;
            margin-bottom: 2rem;
            border-top: 1px solid rgba(0, 216, 246, 0.2);
            border-right: 1px solid rgba(0, 216, 246, 0.2);
            border-bottom: 1px solid rgba(0, 216, 246, 0.2);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5);">
    <div style="font-family: var(--font-mono); font-size: 0.75rem; color: #00D8F6; letter-spacing: 0.12em; text-transform: uppercase;">THE QUESTION</div>
    <h2 style="font-family: var(--font-display); font-size: 1.7rem; color: #FFFFFF; margin-top: 6px; margin-bottom: 10px;">
        "Can AI detect tiny planetary signals hidden inside noisy stellar observations?"
    </h2>
    <p style="font-size: 0.96rem; color: #CBD5E1; line-height: 1.7; margin: 0;">
        When an exoplanet crosses the disk of its parent star, it dims the starlight by an almost imperceptible fraction—often less than 0.01% for Earth-sized worlds.
        NASA space telescopes observe millions of stars continuously, generating petabytes of raw brightness measurements.
        Human vetting cannot keep pace with this deluge of data. ExoMiner AI answers this challenge with verifiable, physically grounded machine learning.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- THE STORY IN 4 CHAPTERS ----------------
st.markdown("### THE SCIENTIFIC ARCHITECTURE")

c1, c2 = st.columns(2)

with c1:
    render_panel(
        "01 • THE DATA: HIGH-PRECISION PHOTOMETRY",
        """
        The <b>NASA Kepler</b> and <b>TESS</b> missions deployed space telescopes into heliocentric and lunar-resonant orbits,
        eliminating atmospheric scintillation to measure stellar flux with parts-per-million precision.
        Observations span years of continuous 29.4-minute (Kepler) and 2-minute (TESS) integrations.
        """,
        "Space Telemetry"
    )

    render_panel(
        "03 • THE AI: ENSEMBLE DECISION TREES",
        """
        Rather than treating light curves as black-box image pixels, ExoMiner AI engineers astrophysically grounded features:
        transit depth, harmonic period, transit duration, SNR, skewness, and kurtosis.
        A <b>Random Forest Ensemble</b> combines decision boundaries across 100 decorrelated trees to maximize sensitivity while suppressing false alarms.
        """,
        "Algorithmic Architecture"
    )

with c2:
    render_panel(
        "02 • THE SIGNAL: TRANSIT PHOTOMETRY",
        """
        Under the transit method, the depth of the photometric dip is mathematically tied to the square of the planet-to-star radius ratio:
        <div style="background: rgba(3, 7, 18, 0.7); padding: 8px; border-radius: 6px; font-family: monospace; color: #00D8F6; text-align: center; margin: 10px 0; border: 1px solid rgba(0,216,246,0.2);">
            ΔF / F ≈ (R_planet / R_star)²
        </div>
        Because orbits repeat periodically under Kepler's laws, phase-folding multiple transit events enhances the signal-to-noise ratio by √N.
        """,
        "Astrophysical Principles"
    )

    render_panel(
        "04 • THE RESULT: VERIFIED CANDIDATES",
        """
        The model outputs calibrated probability scores, confidence grades, and explainability feature rankings.
        This isolates viable candidates for ground-based Doppler spectroscopy follow-up (e.g. Keck HIRES, HARPS-N, ESPRESSO),
        weeding out eclipsing binary false positives and instrumental glitches.
        """,
        "Candidate Discovery"
    )

# ---------------- EXOPLANET EXPLORATION TIMELINE ----------------
st.markdown("<br><hr style='border-color: rgba(0, 216, 246, 0.16); margin: 2rem 0;'>", unsafe_allow_html=True)
st.markdown("""
<div style="margin-bottom: 1.4rem;">
    <div style="font-family: var(--font-mono); font-size: 0.75rem; color: #00D8F6; letter-spacing: 0.14em; text-transform: uppercase;">
        DECISIVE MISSIONS & DISCOVERY HISTORY
    </div>
    <h2 style="font-family: var(--font-display); font-size: 1.8rem; color: #FFFFFF; margin-top: 2px;">
        THE SPACE ASTRONOMY TIMELINE
    </h2>
</div>
""", unsafe_allow_html=True)

tm1, tm2, tm3, tm4 = st.columns(4)

with tm1:
    st.markdown("""
    <div class="hud-panel" style="height: 100%; border-top: 3px solid #00D8F6;">
        <div style="font-family: var(--font-mono); font-size: 0.72rem; color: #00D8F6;">2009 – 2018</div>
        <div style="font-family: var(--font-display); font-size: 1.1rem; color: #FFFFFF; font-weight: 700; margin-top: 2px;">NASA KEPLER</div>
        <p style="font-size: 0.82rem; color: #94A3B8; margin-top: 8px; line-height: 1.5;">
            Monitored 150,000+ stars continuously in Cygnus-Lyra. Discovered 2,700+ confirmed planets, revealing small worlds are ubiquitous.
        </p>
    </div>
    """, unsafe_allow_html=True)

with tm2:
    st.markdown("""
    <div class="hud-panel" style="height: 100%; border-top: 3px solid #38BDF8;">
        <div style="font-family: var(--font-mono); font-size: 0.72rem; color: #38BDF8;">2018 – PRESENT</div>
        <div style="font-family: var(--font-display); font-size: 1.1rem; color: #38BDF8; font-weight: 700; margin-top: 2px;">NASA TESS</div>
        <p style="font-size: 0.82rem; color: #94A3B8; margin-top: 8px; line-height: 1.5;">
            Transiting Exoplanet Survey Satellite scanning 85% of the sky. Ingested over 400 confirmed exoplanets and 7,000+ candidates.
        </p>
    </div>
    """, unsafe_allow_html=True)

with tm3:
    st.markdown("""
    <div class="hud-panel" style="height: 100%; border-top: 3px solid #10B981;">
        <div style="font-family: var(--font-mono); font-size: 0.72rem; color: #10B981;">2021 – 2022</div>
        <div style="font-family: var(--font-display); font-size: 1.1rem; color: #10B981; font-weight: 700; margin-top: 2px;">NASA EXOMINER AI</div>
        <p style="font-size: 0.82rem; color: #94A3B8; margin-top: 8px; line-height: 1.5;">
            Deep learning neural classifier developed by NASA Ames, confirming 301 exoplanets in archived Kepler data that human vetters missed.
        </p>
    </div>
    """, unsafe_allow_html=True)

with tm4:
    st.markdown("""
    <div class="hud-panel" style="height: 100%; border-top: 3px solid #F59E0B;">
        <div style="font-family: var(--font-mono); font-size: 0.72rem; color: #F59E0B;">2022 – 2030+</div>
        <div style="font-family: var(--font-display); font-size: 1.1rem; color: #F59E0B; font-weight: 700; margin-top: 2px;">JWST & ROMAN</div>
        <p style="font-size: 0.82rem; color: #94A3B8; margin-top: 8px; line-height: 1.5;">
            Infrared transmission spectroscopy characterizing atmospheric water vapor, carbon dioxide, methane, and cloud profiles of transit targets.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- ASTROPHYSICS EQUATIONS CHEATSHEET ----------------
st.markdown("<br><hr style='border-color: rgba(0, 216, 246, 0.16); margin: 2rem 0;'>", unsafe_allow_html=True)
st.markdown("""
<div style="margin-bottom: 1.4rem;">
    <div style="font-family: var(--font-mono); font-size: 0.75rem; color: #00D8F6; letter-spacing: 0.14em; text-transform: uppercase;">
        THEORETICAL CORNERSTONES
    </div>
    <h2 style="font-family: var(--font-display); font-size: 1.8rem; color: #FFFFFF; margin-top: 2px;">
        ASTROPHYSICAL TRANSIT FORMULAE CHEATSHEET
    </h2>
</div>
""", unsafe_allow_html=True)

eq1, eq2, eq3 = st.columns(3)

with eq1:
    render_panel(
        "TRANSIT DEPTH & RADIUS RATIO",
        """
        Fractional drop in stellar flux during central planetary transit:
        <div style="background: rgba(3, 7, 18, 0.7); padding: 8px; border-radius: 6px; font-family: monospace; color: #00D8F6; text-align: center; margin: 8px 0; border: 1px solid rgba(0,216,246,0.2);">
            δ = ΔF / F = (R_p / R_*)²
        </div>
        Yields planetary radius R_p when host star radius R_* is known from spectroscopy or Gaia DR2 parallax.
        """,
        "Physical Dimension"
    )

with eq2:
    render_panel(
        "KEPLER'S THIRD LAW OF MOTION",
        """
        Governs orbital semi-major axis (a) from transit recurrence period (P):
        <div style="background: rgba(3, 7, 18, 0.7); padding: 8px; border-radius: 6px; font-family: monospace; color: #10B981; text-align: center; margin: 8px 0; border: 1px solid rgba(16,185,129,0.2);">
            a³ = (G · M_* / 4π²) · P²
        </div>
        Fixes orbital distance in astronomical units (AU) around a star of mass M_*.
        """,
        "Orbital Mechanics"
    )

with eq3:
    render_panel(
        "EQUILIBRIUM TEMPERATURE",
        """
        Radiation balance assuming zero Bond albedo and uniform heat redistribution:
        <div style="background: rgba(3, 7, 18, 0.7); padding: 8px; border-radius: 6px; font-family: monospace; color: #F59E0B; text-align: center; margin: 8px 0; border: 1px solid rgba(245,158,11,0.2);">
            T_eq = T_* · (R_* / 2a)^(1/2) · (1 - A_B)^(1/4)
        </div>
        Determines whether the candidate falls within the liquid water habitable zone (200 – 320 K).
        """,
        "Planetary Climate"
    )

# ---------------- CITATIONS & SCIENTIFIC DISCLAIMER ----------------
st.markdown("<br><hr style='border-color: rgba(0, 216, 246, 0.16); margin: 2rem 0;'>", unsafe_allow_html=True)
render_panel(
    "KEY SCIENTIFIC CITATIONS & DATA ACKNOWLEDGMENTS",
    """
    <ul style="margin: 0; padding-left: 18px; line-height: 1.8; color: #CBD5E1; font-size: 0.86rem;">
        <li><b>ExoMiner:</b> Valizadegan, H., et al. (2022). <i>"ExoMiner: A Highly Accurate and Explainable Deep Learning Classifier to Validate Kepler and TESS Exoplanets"</i>, The Astrophysical Journal, 926(2), 120.</li>
        <li><b>NASA Exoplanet Archive:</b> Akeson, R. L., et al. (2013). <i>"The NASA Exoplanet Archive: Data and Tools for Exoplanet Research"</i>, PASP, 125, 989.</li>
        <li><b>Lightkurve:</b> Lightkurve Collaboration (2018). <i>"Lightkurve: Kepler and TESS time series analysis in Python"</i>, Astrophysics Source Code Library, ascl:1812.013.</li>
        <li><b>Kepler DR25:</b> Thompson, S. E., et al. (2018). <i>"Planetary Candidates Observed by Kepler. VIII. A Fully Automated Catalog with DR25"</i>, ApJS, 235, 38.</li>
    </ul>
    """,
    "Academic Bibliography"
)

render_scientific_disclaimer()
