"""
styles.py - Deep Space Observatory Global Design System.
Provides the scientific console theme, grid backgrounds, typography, and HUD styles.
Guarantees 100% dark cosmic theme coverage with zero unstyled white areas across all displays.
"""

import streamlit as st


DEEP_SPACE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');

:root {
    --space-bg: #030712;
    --space-deep: #050B18;
    --panel-bg: rgba(11, 21, 40, 0.78);
    --panel-hover: rgba(15, 29, 56, 0.90);
    --panel-border: rgba(0, 216, 246, 0.24);
    --panel-border-bright: rgba(0, 216, 246, 0.70);
    --cyan-primary: #00D8F6;
    --cyan-glow: rgba(0, 216, 246, 0.45);
    --blue-accent: #0284C7;
    --blue-glow: rgba(2, 132, 199, 0.35);
    --purple-nebula: #818CF8;
    --amber-warning: #F59E0B;
    --green-success: #10B981;
    --red-alert: #EF4444;
    --text-pure: #FFFFFF;
    --text-bright: #F1F5F9;
    --text-muted: #94A3B8;
    --text-dim: #64748B;
    --font-display: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    --font-ui: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    --font-mono: 'JetBrains Mono', 'Inter', monospace;
    color-scheme: dark;
}

/* Universal Typography Enforcement: Inter */
*, *::before, *::after {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    box-sizing: border-box;
}

/* Eliminate All Streamlit Default White Artifacts */
html, body, #root, .stApp, 
[data-testid="stAppViewContainer"], 
[data-testid="stAppViewBlockContainer"],
[data-testid="stVerticalBlock"],
section.main {
    background-color: var(--space-bg) !important;
    color: var(--text-bright) !important;
    color-scheme: dark !important;
}

/* Hide Default Chrome but Style Visible Top Header */
#MainMenu, footer {
    visibility: hidden;
    height: 0;
}

[data-testid="stSidebarNav"] {
    display: none !important;
}

[data-testid="stHeader"] {
    background: rgba(3, 7, 18, 0.75) !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
    border-bottom: 1px solid rgba(0, 216, 246, 0.12) !important;
}

[data-testid="stToolbar"], [data-testid="stDecoration"] {
    display: none !important;
}

[data-testid="stBottom"] {
    background: transparent !important;
}

[data-testid="stSidebar"] {
    background-color: #070F22 !important;
    border-right: 1px solid rgba(0, 216, 246, 0.2) !important;
}

/* Fluid Responsive Container (fills wide monitors smoothly without huge empty gaps) */
.main .block-container {
    padding-top: 1rem !important;
    padding-bottom: 4.5rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    max-width: 95% !important;
    width: 95% !important;
}

/* Cosmic Nebula Grid Background on Entire Application */
.stApp {
    background-color: var(--space-bg) !important;
    background-image: 
        radial-gradient(circle at 50% 0%, rgba(2, 132, 199, 0.16) 0%, transparent 60%),
        radial-gradient(circle at 88% 25%, rgba(129, 140, 248, 0.08) 0%, transparent 45%),
        radial-gradient(circle at 12% 70%, rgba(0, 216, 246, 0.06) 0%, transparent 50%),
        linear-gradient(rgba(0, 216, 246, 0.028) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 216, 246, 0.028) 1px, transparent 1px) !important;
    background-size: 100% 100%, 100% 100%, 100% 100%, 42px 42px, 42px 42px !important;
    background-attachment: fixed !important;
}

/* Typography Hierarchy */
h1, h2, h3, h4, .hud-title {
    font-family: var(--font-display) !important;
    font-weight: 700;
    letter-spacing: 0.03em;
    color: var(--text-pure) !important;
}

p, span, div, label {
    color: var(--text-bright);
}

.mono-text {
    font-family: var(--font-mono) !important;
}

/* Top Navigation & Brand Header Shell */
.obs-shell-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: linear-gradient(135deg, rgba(8, 15, 30, 0.92) 0%, rgba(15, 29, 56, 0.85) 100%);
    border: 1px solid rgba(0, 216, 246, 0.28);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 12px;
    padding: 0.9rem 1.6rem;
    margin-bottom: 1.6rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(0, 216, 246, 0.2);
}

.brand-wrapper {
    display: flex;
    align-items: center;
    gap: 14px;
}

.brand-title {
    font-family: var(--font-display) !important;
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    color: var(--cyan-primary) !important;
    margin: 0;
    line-height: 1.1;
    text-shadow: 0 0 14px rgba(0, 216, 246, 0.55);
}

.brand-subtitle {
    font-family: var(--font-mono) !important;
    font-size: 0.70rem;
    color: var(--text-dim) !important;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin: 3px 0 0 0;
}

/* System Status HUD */
.status-hud {
    display: flex;
    align-items: center;
    gap: 12px;
    font-family: var(--font-mono) !important;
    font-size: 0.76rem;
}

.status-chip {
    display: flex;
    align-items: center;
    gap: 7px;
    background: rgba(11, 21, 40, 0.9);
    border: 1px solid rgba(0, 216, 246, 0.22);
    padding: 5px 12px;
    border-radius: 8px;
    color: var(--text-muted);
}

.status-pulse-green {
    width: 8px;
    height: 8px;
    background-color: var(--green-success);
    border-radius: 50%;
    box-shadow: 0 0 10px var(--green-success);
    animation: pulseGlow 2s infinite;
}

@keyframes pulseGlow {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.85); }
}

/* Scientific HUD Panels / Glass Console */
.hud-panel {
    background: var(--panel-bg);
    border: 1px solid var(--panel-border);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-radius: 10px;
    padding: 1.4rem;
    margin-bottom: 1.3rem;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(0, 216, 246, 0.1);
    transition: all 0.25s ease;
    position: relative;
}

.hud-panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 9px; height: 9px;
    border-top: 2px solid var(--cyan-primary);
    border-left: 2px solid var(--cyan-primary);
    border-top-left-radius: 3px;
}

.hud-panel::after {
    content: '';
    position: absolute;
    bottom: 0; right: 0;
    width: 9px; height: 9px;
    border-bottom: 2px solid var(--cyan-primary);
    border-right: 2px solid var(--cyan-primary);
    border-bottom-right-radius: 3px;
}

.hud-panel:hover {
    border-color: var(--panel-border-bright);
    box-shadow: 0 10px 35px rgba(0, 216, 246, 0.15), inset 0 1px 0 rgba(0, 216, 246, 0.25);
}

/* Metric / KPI Readout HUD */
.kpi-hud {
    background: linear-gradient(145deg, rgba(11, 21, 40, 0.90) 0%, rgba(15, 29, 56, 0.75) 100%);
    border: 1px solid rgba(0, 216, 246, 0.22);
    border-radius: 10px;
    padding: 1.2rem 1.3rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.45);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.kpi-hud:hover {
    transform: translateY(-2px);
    border-color: rgba(0, 216, 246, 0.5);
    box-shadow: 0 8px 28px rgba(0, 216, 246, 0.18);
}

.kpi-hud::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--cyan-primary), transparent);
}

.kpi-hud-label {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-muted);
    margin-bottom: 0.4rem;
}

.kpi-hud-value {
    font-family: var(--font-display) !important;
    font-size: 1.9rem;
    font-weight: 800;
    color: var(--text-pure);
    letter-spacing: 0.02em;
    line-height: 1.1;
    margin-bottom: 0.35rem;
}

.kpi-hud-sub {
    font-family: var(--font-mono) !important;
    font-size: 0.75rem;
    color: var(--cyan-primary);
}

/* Buttons: High-Tech Scientific Console */
.stButton > button {
    font-family: var(--font-display) !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    background: linear-gradient(180deg, rgba(0, 216, 246, 0.18) 0%, rgba(2, 132, 199, 0.30) 100%) !important;
    color: var(--text-pure) !important;
    border: 1px solid var(--cyan-primary) !important;
    border-radius: 8px !important;
    padding: 0.65rem 1.4rem !important;
    box-shadow: 0 0 14px rgba(0, 216, 246, 0.22) !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.stButton > button:hover {
    background: linear-gradient(180deg, rgba(0, 216, 246, 0.38) 0%, rgba(2, 132, 199, 0.50) 100%) !important;
    border-color: #38BDF8 !important;
    box-shadow: 0 0 22px rgba(0, 216, 246, 0.50) !important;
    transform: translateY(-1px) !important;
}

.stButton > button:active {
    transform: translateY(1px) !important;
}

/* Navigation Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent !important;
    border-bottom: 1px solid rgba(0, 216, 246, 0.25) !important;
    padding-bottom: 4px;
}

.stTabs [data-baseweb="tab"] {
    font-family: var(--font-display) !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
    background-color: rgba(15, 23, 42, 0.7) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 0.6rem 1.4rem !important;
    transition: all 0.2s ease !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-pure) !important;
    border-color: rgba(0, 216, 246, 0.3) !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: var(--cyan-primary) !important;
    background-color: rgba(0, 216, 246, 0.15) !important;
    border: 1px solid var(--cyan-primary) !important;
    border-bottom: none !important;
    box-shadow: 0 0 14px rgba(0, 216, 246, 0.3) !important;
}

.stTabs [data-baseweb="tab-panel"] {
    background-color: transparent !important;
    padding-top: 1.2rem !important;
}

/* Form Controls & Selectboxes (Zero White Dropdown) */
div[data-baseweb="select"] > div {
    background-color: rgba(11, 21, 40, 0.90) !important;
    border: 1px solid rgba(0, 216, 246, 0.30) !important;
    border-radius: 8px !important;
    color: #FFFFFF !important;
}

div[data-baseweb="popover"], 
div[data-baseweb="menu"], 
ul[role="listbox"] {
    background-color: #0B1528 !important;
    border: 1px solid rgba(0, 216, 246, 0.35) !important;
    border-radius: 8px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.8) !important;
}

li[role="option"] {
    background-color: #0B1528 !important;
    color: #F1F5F9 !important;
    font-size: 0.85rem !important;
    padding: 8px 12px !important;
}

li[role="option"]:hover, 
li[role="option"][aria-selected="true"] {
    background-color: rgba(0, 216, 246, 0.22) !important;
    color: #00D8F6 !important;
}

/* Text & Number Inputs */
input[type="text"], input[type="number"] {
    background-color: rgba(11, 21, 40, 0.85) !important;
    border: 1px solid rgba(0, 216, 246, 0.28) !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
}

input[type="text"]:focus, input[type="number"]:focus {
    border-color: var(--cyan-primary) !important;
    box-shadow: 0 0 10px rgba(0, 216, 246, 0.35) !important;
}

/* Radio & Checkbox */
div[data-testid="stRadio"] label, 
div[data-testid="stCheckbox"] label {
    color: var(--text-bright) !important;
}

/* Sliders */
div[data-testid="stSlider"] {
    color: var(--cyan-primary) !important;
}

/* Streamlit DataFrames */
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(0, 216, 246, 0.25) !important;
    border-radius: 10px !important;
    background: rgba(11, 21, 40, 0.95) !important;
    overflow: hidden;
}

/* Expanders */
div[data-testid="stExpander"] {
    background: rgba(11, 21, 40, 0.80) !important;
    border: 1px solid rgba(0, 216, 246, 0.22) !important;
    border-radius: 10px !important;
    margin-top: 1rem !important;
}

div[data-testid="stExpander"] summary {
    font-family: var(--font-display) !important;
    color: var(--cyan-primary) !important;
    font-weight: 600 !important;
}

/* Download Button */
.stDownloadButton > button {
    background: linear-gradient(180deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.35) 100%) !important;
    border: 1px solid var(--green-success) !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    box-shadow: 0 0 12px rgba(16, 185, 129, 0.25) !important;
}

.stDownloadButton > button:hover {
    background: linear-gradient(180deg, rgba(16, 185, 129, 0.35) 0%, rgba(5, 150, 105, 0.5) 100%) !important;
    box-shadow: 0 0 20px rgba(16, 185, 129, 0.45) !important;
}

/* iframe Container Reset */
iframe, iframe[data-testid="stCustomComponentV1"] {
    background-color: transparent !important;
    color-scheme: dark !important;
    border: none !important;
    border-radius: 10px;
}

/* Scientific Progress Sequence Bar */
.prog-step {
    font-family: var(--font-mono) !important;
    font-size: 0.82rem;
    padding: 9px 14px;
    border-radius: 6px;
    margin-bottom: 7px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.prog-active {
    background: rgba(0, 216, 246, 0.16);
    border: 1px solid var(--cyan-primary);
    color: var(--cyan-primary);
    box-shadow: 0 0 12px rgba(0, 216, 246, 0.25);
}
.prog-done {
    background: rgba(16, 185, 129, 0.14);
    border: 1px solid var(--green-success);
    color: var(--green-success);
}
.prog-pending {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    color: var(--text-dim);
}

/* Scientific Disclaimer Tag */
.sci-disclaimer {
    background: rgba(245, 158, 11, 0.10);
    border-left: 3px solid var(--amber-warning);
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    font-size: 0.84rem;
    color: #FDE68A;
    margin: 1.2rem 0;
    line-height: 1.6;
}

/* Feature Badge Card */
.feat-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 8px;
    border-radius: 4px;
    font-family: var(--font-mono) !important;
    font-size: 0.72rem;
    background: rgba(0, 216, 246, 0.12);
    border: 1px solid rgba(0, 216, 246, 0.3);
    color: #00D8F6;
}

/* Custom Scrollbars */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #030712;
}
::-webkit-scrollbar-thumb {
    background: rgba(0, 216, 246, 0.28);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(0, 216, 246, 0.55);
}
</style>
"""


def inject_observatory_theme():
    """Injects the complete Deep Space Observatory CSS theme into the Streamlit session."""
    st.markdown(DEEP_SPACE_CSS, unsafe_allow_html=True)
