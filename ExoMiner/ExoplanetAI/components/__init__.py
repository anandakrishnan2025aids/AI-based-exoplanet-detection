"""
ExoplanetAI UI Components Package.
Deep Space Observatory Design System.
"""

try:
    from .styles import inject_observatory_theme
    from .navbar import render_header, NAV_ITEMS
    from .cards import render_kpi, render_panel, render_scientific_disclaimer, render_verdict_badge
    from .charts import (
        plot_observatory_lightcurve,
        plot_before_after_cleaning,
        plot_phase_folded_transit,
        plot_explainability_bars,
        plot_model_confusion_matrix,
        OBSERVATORY_LAYOUT
    )
    from .hero_visual import render_interactive_transit_hero
except ImportError:
    # Allows headless / test environments without streamlit to import non-UI submodules (e.g. report)
    inject_observatory_theme = None
    render_header = None
    NAV_ITEMS = []
    render_kpi = None
    render_panel = None
    render_scientific_disclaimer = None
    render_verdict_badge = None
    plot_observatory_lightcurve = None
    plot_before_after_cleaning = None
    plot_phase_folded_transit = None
    plot_explainability_bars = None
    plot_model_confusion_matrix = None
    OBSERVATORY_LAYOUT = {}
    render_interactive_transit_hero = None

from .report import generate_candidate_report

__all__ = [
    "inject_observatory_theme",
    "render_header",
    "NAV_ITEMS",
    "render_kpi",
    "render_panel",
    "render_scientific_disclaimer",
    "render_verdict_badge",
    "plot_observatory_lightcurve",
    "plot_before_after_cleaning",
    "plot_phase_folded_transit",
    "plot_explainability_bars",
    "plot_model_confusion_matrix",
    "OBSERVATORY_LAYOUT",
    "render_interactive_transit_hero",
    "generate_candidate_report",
]
