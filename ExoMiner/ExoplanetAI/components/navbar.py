"""
navbar.py - Custom Deep Space Observatory Application Shell and Navigation.
Implements the top header layout with minimal orbital neural SVG logo,
page routing, and real-time system status.
"""

from typing import Optional
import streamlit as st


# Minimal orbital symbol combined with an AI/neural signal motif (SVG)
ORBITAL_NEURAL_SVG = """
<svg width="34" height="34" viewBox="0 0 44 44" fill="none" xmlns="http://www.w3.org/2000/svg">
    <!-- Star core -->
    <circle cx="22" cy="22" r="6" fill="#00D8F6" filter="drop-shadow(0 0 6px #00D8F6)"/>
    <!-- Outer orbital ring -->
    <ellipse cx="22" cy="22" rx="19" ry="8" stroke="#00D8F6" stroke-width="1.5" stroke-dasharray="3 3" opacity="0.65" transform="rotate(-25 22 22)"/>
    <!-- Transiting planet -->
    <circle cx="34" cy="16" r="3.2" fill="#FFFFFF" stroke="#00D8F6" stroke-width="1"/>
    <!-- Neural signal pulse line -->
    <path d="M4 22H14L17 17L20 27L24 16L27 24L30 22H40" stroke="#38BDF8" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.85"/>
</svg>
"""

NAV_ITEMS = [
    ("OVERVIEW", "app.py", "Overview / Home"),
    ("OBSERVATORY", "pages/01_Observatory.py", "Main Analysis Center"),
    ("DETECT", "pages/02_Detect.py", "Candidate Vetting"),
    ("SIGNALS", "pages/03_Signals.py", "Signals Lab"),
    ("DATA", "pages/04_Data_Explorer.py", "Data Explorer"),
    ("MODEL", "pages/05_Model_Lab.py", "Model Lab"),
    ("ABOUT", "pages/06_About.py", "About & Science")
]


def render_header(active_page: str = "OVERVIEW") -> str:
    """
    Renders the Deep Space Observatory top application shell across all pages.
    Includes branding, active navigation indicators, and system status telemetry.
    """
    # 1. Top Brand + Status Telemetry Bar
    st.markdown(f"""
    <div class="obs-shell-header">
        <div class="brand-wrapper">
            {ORBITAL_NEURAL_SVG}
            <div>
                <div class="brand-title">EXOPLANET AI</div>
                <div class="brand-subtitle">AI-ASSISTED ASTRONOMICAL SIGNAL ANALYSIS</div>
            </div>
        </div>
        <div class="status-hud">
            <div class="status-chip">
                <span class="status-pulse-green"></span>
                <span>SYSTEM ONLINE</span>
            </div>
            <div class="status-chip" style="color: #00D8F6; border-color: rgba(0, 216, 246, 0.3);">
                <span>RF v2.2 ACTIVE</span>
            </div>
            <div class="status-chip">
                <span>5,657 CADENCES</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Modern Navigation Bar
    cols = st.columns(len(NAV_ITEMS))
    selected_page = active_page

    for idx, (name, path, tooltip) in enumerate(NAV_ITEMS):
        is_active = (name == active_page)
        btn_label = f"◆ {name}" if is_active else name
        
        with cols[idx]:
            if st.button(
                btn_label,
                key=f"nav_btn_{name}_{active_page}",
                help=tooltip,
                use_container_width=True
            ):
                st.session_state["nav_page"] = name
                # Attempt seamless page switch if file exists
                try:
                    st.switch_page(path)
                except Exception:
                    # Fallback for single-page routing in app.py
                    st.rerun()

    st.markdown("<hr style='border-color: rgba(0, 216, 246, 0.15); margin: 0.8rem 0 1.5rem 0;'>", unsafe_allow_html=True)
    return selected_page
