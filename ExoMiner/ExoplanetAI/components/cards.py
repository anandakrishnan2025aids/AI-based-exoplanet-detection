"""
cards.py - Scientific HUD Readouts, Console Cards, and Rich Astrophysical Widgets.
Provides colorful, high-tech telemetry cards, planet showcases, and data widgets.
"""

from typing import Optional
import streamlit as st


def render_kpi(label: str, value: str, subtext: Optional[str] = None):
    """Renders a high-tech scientific KPI readout box."""
    sub_html = f'<div class="kpi-hud-sub">{subtext}</div>' if subtext else ''
    html = f"""
    <div class="kpi-hud">
        <div class="kpi-hud-label">{label}</div>
        <div class="kpi-hud-value">{value}</div>
        {sub_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_panel(title: str, content_html: str, subtitle: Optional[str] = None):
    """Renders an observatory HUD panel with corner brackets."""
    sub = f'<div style="font-family: var(--font-mono); font-size: 0.75rem; color: #94A3B8; margin-top: 2px;">{subtitle}</div>' if subtitle else ''
    st.markdown(f"""
    <div class="hud-panel">
        <div style="font-family: var(--font-display); font-size: 1.05rem; font-weight: 700; color: #00D8F6; letter-spacing: 0.05em;">{title}</div>
        {sub}
        <div style="margin-top: 0.9rem; line-height: 1.6; color: #E2E8F0;">
            {content_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_planet_card(
    name: str,
    planet_type: str,
    distance_ly: str,
    period_days: str,
    radius_earth: str,
    teq_k: str,
    glow_color: str = "#00D8F6",
    badge_label: str = "CONFIRMED"
):
    """Renders an exoplanet spotlight card with colorful astronomical specs."""
    st.markdown(f"""
    <div style="background: linear-gradient(145deg, rgba(11, 21, 40, 0.92) 0%, rgba(15, 29, 56, 0.85) 100%);
                border: 1px solid rgba(0, 216, 246, 0.25);
                border-radius: 10px;
                padding: 1.2rem;
                position: relative;
                overflow: hidden;
                box-shadow: 0 6px 24px rgba(0,0,0,0.5);
                transition: transform 0.25s, border-color 0.25s;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
            <div>
                <div style="font-family: var(--font-display); font-size: 1.15rem; font-weight: 800; color: #FFFFFF;">{name}</div>
                <div style="font-family: var(--font-mono); font-size: 0.72rem; color: {glow_color}; letter-spacing: 0.08em; text-transform: uppercase;">{planet_type}</div>
            </div>
            <span style="background: rgba(0, 216, 246, 0.15); border: 1px solid {glow_color}; color: {glow_color}; font-size: 0.68rem; font-family: var(--font-mono); padding: 2px 7px; border-radius: 4px; font-weight: 600;">
                {badge_label}
            </span>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 10px; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 10px; font-size: 0.78rem;">
            <div>
                <span style="color: #64748B; font-family: var(--font-mono);">ORBITAL PERIOD:</span><br>
                <b style="color: #F1F5F9;">{period_days}</b>
            </div>
            <div>
                <span style="color: #64748B; font-family: var(--font-mono);">PLANET RADIUS:</span><br>
                <b style="color: #00D8F6;">{radius_earth}</b>
            </div>
            <div>
                <span style="color: #64748B; font-family: var(--font-mono);">EQUILIBRIUM TEMP:</span><br>
                <b style="color: #F59E0B;">{teq_k}</b>
            </div>
            <div>
                <span style="color: #64748B; font-family: var(--font-mono);">DISTANCE:</span><br>
                <b style="color: #CBD5E1;">{distance_ly}</b>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_scientific_disclaimer():
    """Renders the required subtle but clear scientific disclaimer."""
    st.markdown("""
    <div class="sci-disclaimer">
        <b>ASTROPHYSICAL DISCLAIMER:</b> Educational / research prototype.
        Predictions represent AI-assisted statistical candidates and do <b>not</b> constitute scientific confirmation of an exoplanet.
        Ground-truth confirmation requires high-resolution Doppler radial velocity follow-up spectroscopy.
    </div>
    """, unsafe_allow_html=True)


def render_verdict_badge(
    is_exoplanet: bool,
    model_score_pct: Optional[float] = None,
    target_name: str = "Target",
    is_heuristic: bool = False,
    prediction_status: Optional[str] = None
):
    """
    Renders the prominent candidate classification result badge.
    Explicitly reflects heuristic fallback status and scientific non-detection semantics.
    """
    # 1. Unable to evaluate / invalid data
    if prediction_status == "Unable to evaluate — insufficient or invalid data":
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(245, 158, 11, 0.16) 0%, rgba(120, 53, 15, 0.28) 100%);
                    border: 1.5px solid #F59E0B;
                    border-radius: 10px;
                    padding: 1.3rem 1.8rem;
                    margin-bottom: 1.4rem;
                    box-shadow: 0 0 20px rgba(245, 158, 11, 0.25);">
            <div style="font-family: var(--font-mono); font-size: 0.78rem; color: #F59E0B; text-transform: uppercase; letter-spacing: 0.12em;">
                ⚠️ DATA QUALITY NOTICE • TARGET: {target_name}
            </div>
            <div style="display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap; margin-top: 8px;">
                <div style="font-family: var(--font-display); font-size: 1.75rem; font-weight: 800; color: #FFFFFF; letter-spacing: 0.04em;">
                    UNABLE TO EVALUATE • INSUFFICIENT OR INVALID DATA
                </div>
                <div style="text-align: right;">
                    <span style="font-family: var(--font-mono); font-size: 0.82rem; color: #94A3B8;">STATUS: </span>
                    <span style="font-family: var(--font-display); font-size: 1.8rem; font-weight: 900; color: #F59E0B;">N/A</span>
                </div>
            </div>
            <div style="margin-top: 8px; font-size: 0.85rem; color: #FEF3C7; border-top: 1px solid rgba(245, 158, 11, 0.2); padding-top: 8px;">
                Photometric observation contains insufficient samples (&lt;10 cadences) or constant zero-variance flux. Candidate vetting cannot proceed.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # 2. Exoplanet / Transit Candidate
    if is_exoplanet:
        headline = "POTENTIAL CANDIDATE (HEURISTIC ESTIMATE)" if is_heuristic else "TRANSIT DETECTED • CANDIDATE FOR FURTHER ANALYSIS"
        score_label = "HEURISTIC SCORE: " if is_heuristic else "CONFIDENCE: "
        score_val_str = f"{model_score_pct:.1f}%" if model_score_pct is not None else "CANDIDATE"
        subtext = (
            "Calculated via uncalibrated astrophysical heuristics (Kepler transit depth/SNR/skew rules). "
            "Not a validated machine learning model inference."
            if is_heuristic else
            "Periodic transit signature consistent with planetary limb occultation. Candidate requires high-resolution Doppler spectroscopic follow-up for physical confirmation."
        )
        status_prefix = "● HEURISTIC ANALYSIS COMPLETE" if is_heuristic else "● TRANSIT CANDIDATE IDENTIFIED"
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.16) 0%, rgba(6, 78, 59, 0.28) 100%);
                    border: 1.5px solid #10B981;
                    border-radius: 10px;
                    padding: 1.3rem 1.8rem;
                    margin-bottom: 1.4rem;
                    box-shadow: 0 0 24px rgba(16, 185, 129, 0.25);">
            <div style="font-family: var(--font-mono); font-size: 0.78rem; color: #10B981; text-transform: uppercase; letter-spacing: 0.12em;">
                {status_prefix} • TARGET: {target_name}
            </div>
            <div style="display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap; margin-top: 8px;">
                <div style="font-family: var(--font-display); font-size: 1.75rem; font-weight: 800; color: #FFFFFF; letter-spacing: 0.04em;">
                    {headline}
                </div>
                <div style="text-align: right;">
                    <span style="font-family: var(--font-mono); font-size: 0.82rem; color: #94A3B8;">{score_label}</span>
                    <span style="font-family: var(--font-display); font-size: 2rem; font-weight: 900; color: #10B981; text-shadow: 0 0 12px rgba(16, 185, 129, 0.5);">{score_val_str}</span>
                </div>
            </div>
            <div style="margin-top: 8px; font-size: 0.85rem; color: #D1FAE5; border-top: 1px solid rgba(16, 185, 129, 0.2); padding-top: 8px;">
                {subtext}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # 3. No Transit Detected / Inconclusive
        headline = "NO TRANSIT DETECTED (HEURISTIC ESTIMATE)" if is_heuristic else "NO TRANSIT DETECTED • INCONCLUSIVE"
        score_label = "HEURISTIC SCORE: " if is_heuristic else "RESULT: "
        if model_score_pct is not None and model_score_pct > 0.0 and not (prediction_status == "No transit detected — inconclusive"):
            score_val_str = f"{model_score_pct:.1f}%"
        else:
            score_val_str = "INCONCLUSIVE"
        subtext = (
            "Heuristic rules indicate stellar noise or insufficient transit parameters. "
            "Model inference was bypassed."
            if is_heuristic else
            "No statistically significant periodic transit dips detected in this observation. Non-detection does not rule out non-transiting planets, long-period orbits, or low-amplitude transits below the noise threshold. Result is scientifically inconclusive."
        )
        status_prefix = "○ HEURISTIC ANALYSIS COMPLETE" if is_heuristic else "○ OBSERVATION VETTED"
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(148, 163, 184, 0.12) 0%, rgba(30, 41, 59, 0.35) 100%);
                    border: 1.5px solid #64748B;
                    border-radius: 10px;
                    padding: 1.3rem 1.8rem;
                    margin-bottom: 1.4rem;
                    box-shadow: 0 0 20px rgba(0, 0, 0, 0.4);">
            <div style="font-family: var(--font-mono); font-size: 0.78rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.12em;">
                {status_prefix} • TARGET: {target_name}
            </div>
            <div style="display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap; margin-top: 8px;">
                <div style="font-family: var(--font-display); font-size: 1.75rem; font-weight: 800; color: #E2E8F0; letter-spacing: 0.04em;">
                    {headline}
                </div>
                <div style="text-align: right;">
                    <span style="font-family: var(--font-mono); font-size: 0.82rem; color: #94A3B8;">{score_label}</span>
                    <span style="font-family: var(--font-display); font-size: 1.8rem; font-weight: 900; color: #CBD5E1;">{score_val_str}</span>
                </div>
            </div>
            <div style="margin-top: 8px; font-size: 0.85rem; color: #94A3B8; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 8px;">
                {subtext}
            </div>
        </div>
        """, unsafe_allow_html=True)
