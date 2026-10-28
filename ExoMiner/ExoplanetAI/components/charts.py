"""
charts.py - Scientific Plotly Visualizations for Deep Space Observatory.
Provides interactive light curves, before/after signal cleaning comparisons,
phase folding, explainability bars, and diagnostic matrices.
"""

from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go


OBSERVATORY_LAYOUT = dict(
    template="plotly_dark",
    plot_bgcolor="rgba(6, 12, 24, 0.85)",
    paper_bgcolor="rgba(0, 0, 0, 0)",
    font=dict(family="Inter, -apple-system, sans-serif", color="#E2E8F0", size=11),
    margin=dict(l=55, r=30, t=45, b=45)
)


def plot_observatory_lightcurve(
    time: np.ndarray,
    flux: np.ndarray,
    trend: Optional[np.ndarray] = None,
    mode: str = "NORMALIZED",
    title: str = "STELLAR FLUX PHOTOMETRIC CADENCES",
    highlight_dips: bool = True
) -> go.Figure:
    """
    Renders the primary interactive observatory light curve with mode-specific overlays.
    Modes: RAW, CLEANED, NORMALIZED, TRANSIT
    """
    fig = go.Figure()

    # Base light curve scatter/line
    line_color = "#00D8F6" if mode in ["NORMALIZED", "TRANSIT"] else ("#38BDF8" if mode == "CLEANED" else "#94A3B8")

    fig.add_trace(go.Scatter(
        x=time,
        y=flux,
        mode="lines",
        name=f"Flux ({mode})",
        line=dict(color=line_color, width=1.5),
        opacity=0.9
    ))

    # Trend line overlay
    if trend is not None and mode in ["RAW", "TRANSIT"]:
        fig.add_trace(go.Scatter(
            x=time,
            y=trend,
            mode="lines",
            name="Stellar Baseline Trend",
            line=dict(color="#F59E0B", width=1.8, dash="dot")
        ))

    # Highlight transit excursions (< 2 sigma)
    if highlight_dips and mode in ["NORMALIZED", "TRANSIT"]:
        median = np.median(flux)
        std = np.std(flux)
        dip_thresh = median - 2.0 * std
        dip_mask = flux < dip_thresh

        if np.any(dip_mask):
            fig.add_trace(go.Scatter(
                x=time[dip_mask],
                y=flux[dip_mask],
                mode="markers",
                name="Potential Transit Excursions (< 2σ)",
                marker=dict(color="#EF4444", size=5, symbol="circle-open")
            ))

            # Threshold line
            fig.add_hline(
                y=dip_thresh,
                line=dict(color="rgba(239, 68, 68, 0.4)", width=1, dash="dash"),
                annotation_text="2σ Transit Dip Threshold",
                annotation_font=dict(color="#EF4444", size=10)
            )

    fig.update_layout(
        title=dict(text=title, font=dict(family="Inter, -apple-system, sans-serif", size=14, color="#00D8F6")),
        xaxis=dict(
            title="Time Cadence (BJD - 2454833 / Days)",
            showgrid=True,
            gridcolor="rgba(0, 216, 246, 0.08)",
            linecolor="rgba(0, 216, 246, 0.2)"
        ),
        yaxis=dict(
            title="Relative Flux",
            showgrid=True,
            gridcolor="rgba(0, 216, 246, 0.08)",
            linecolor="rgba(0, 216, 246, 0.2)"
        ),
        hovermode="x unified",
        height=480,
        **OBSERVATORY_LAYOUT
    )
    return fig


def plot_before_after_cleaning(
    t_raw: np.ndarray,
    f_raw: np.ndarray,
    t_clean: np.ndarray,
    f_clean: np.ndarray
) -> go.Figure:
    """
    Renders side-by-side comparison of raw noisy signal vs clean detrended signal.
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("RAW UNCLEANED SIGNAL (NOISE & DRIFT)", "AFTER PREPROCESSING (DETRENDED & NORMALIZED)"),
        horizontal_spacing=0.08
    )

    # Raw signal
    fig.add_trace(
        go.Scatter(x=t_raw, y=f_raw, mode="lines", name="Raw Flux", line=dict(color="#94A3B8", width=1.2)),
        row=1, col=1
    )

    # Cleaned signal
    fig.add_trace(
        go.Scatter(x=t_clean, y=f_clean, mode="lines", name="Clean Flux", line=dict(color="#00D8F6", width=1.5)),
        row=1, col=2
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title_text="Time (Days)", row=1, col=1)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title_text="Time (Days)", row=1, col=2)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title_text="Raw Flux", row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title_text="Normalized Flux", row=1, col=2)

    fig.update_layout(
        height=320,
        showlegend=False,
        **OBSERVATORY_LAYOUT
    )
    return fig


def plot_phase_folded_transit(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float = 0.0,
    title: str = "PHASE-FOLDED TRANSIT PROFILE"
) -> go.Figure:
    """
    Folds light curve on orbital period to stack all transit occurrences.
    """
    if period <= 0:
        period = 3.524

    phase = ((time - t0) % period) / period
    phase = np.where(phase > 0.5, phase - 1.0, phase)

    sort_idx = np.argsort(phase)
    phase_sorted = phase[sort_idx]
    flux_sorted = flux[sort_idx]

    fig = go.Figure()

    # Individual cadences
    fig.add_trace(go.Scatter(
        x=phase_sorted,
        y=flux_sorted,
        mode="markers",
        name="Folded Cadences",
        marker=dict(color="rgba(148, 163, 184, 0.45)", size=3.5)
    ))

    # Binned transit average profile
    bins = np.linspace(-0.5, 0.5, 60)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    digitized = np.digitize(phase_sorted, bins)
    bin_means = [flux_sorted[digitized == i].mean() if np.any(digitized == i) else np.nan for i in range(1, len(bins))]

    fig.add_trace(go.Scatter(
        x=bin_centers,
        y=bin_means,
        mode="lines+markers",
        name="Binned Transit Profile",
        line=dict(color="#00D8F6", width=2.5),
        marker=dict(color="#00D8F6", size=5)
    ))

    fig.update_layout(
        title=dict(text=f"{title} (Period = {period:.4f} days)", font=dict(family="Inter, -apple-system, sans-serif", size=13, color="#00D8F6")),
        xaxis=dict(title="Orbital Phase (-0.5 to +0.5)", range=[-0.5, 0.5], showgrid=True, gridcolor="rgba(0, 216, 246, 0.08)"),
        yaxis=dict(title="Normalized Flux", showgrid=True, gridcolor="rgba(0, 216, 246, 0.08)"),
        hovermode="x unified",
        height=380,
        **OBSERVATORY_LAYOUT
    )
    return fig


def plot_explainability_bars(feature_contributions: Dict[str, float]) -> go.Figure:
    """
    Renders explainable feature importance horizontal bars answering 'WHY DID THE MODEL SAY THIS?'.
    """
    items = sorted(feature_contributions.items(), key=lambda x: x[1], reverse=False)
    features = [k.replace('_', ' ').title() for k, v in items]
    weights = [v for k, v in items]

    fig = go.Figure(go.Bar(
        x=weights,
        y=features,
        orientation="h",
        marker=dict(
            color=weights,
            colorscale=[[0, "#0284C7"], [0.5, "#00D8F6"], [1.0, "#38BDF8"]],
            line=dict(color="rgba(0, 216, 246, 0.5)", width=1)
        )
    ))

    fig.update_layout(
        title=dict(text="FEATURE CONTRIBUTION TO PREDICTION", font=dict(family="Inter, -apple-system, sans-serif", size=13, color="#00D8F6")),
        xaxis=dict(title="Relative Feature Importance", showgrid=True, gridcolor="rgba(0, 216, 246, 0.08)"),
        yaxis=dict(title=""),
        height=260,
        **OBSERVATORY_LAYOUT
    )
    return fig


def plot_model_confusion_matrix(cm_matrix: List[List[int]], labels: List[str]) -> go.Figure:
    """
    Renders confusion matrix heatmap with counts and true/false annotations.
    """
    cm = np.array(cm_matrix)
    text_annotations = [[str(val) for val in row] for row in cm]

    fig = go.Figure(go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=text_annotations,
        texttemplate="%{text}",
        textfont=dict(family="Inter, -apple-system, sans-serif", size=18, color="#FFFFFF"),
        colorscale=[[0, "#060D1A"], [0.5, "#0284C7"], [1, "#00D8F6"]],
        showscale=False
    ))

    fig.update_layout(
        title=dict(text="BENCHMARK VALIDATION CONFUSION MATRIX", font=dict(family="Inter, -apple-system, sans-serif", size=13, color="#00D8F6")),
        xaxis=dict(title="Predicted Outcome"),
        yaxis=dict(title="Ground Truth (NASA Archive)", autorange="reversed"),
        height=320,
        **OBSERVATORY_LAYOUT
    )
    return fig


def plot_roc_curve(
    fpr: List[float],
    tpr: List[float],
    roc_auc: Optional[float] = None,
    title: str = "EMPIRICAL RECEIVER OPERATING CHARACTERISTIC (ROC)"
) -> go.Figure:
    """
    Renders empirical ROC curve derived from real prediction probabilities.
    Includes no-skill diagonal reference line.
    """
    fig = go.Figure()

    auc_str = f" (AUC = {roc_auc:.4f})" if roc_auc is not None else ""
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode="lines+markers",
        name=f"Empirical ROC{auc_str}",
        line=dict(color="#00D8F6", width=2.5),
        marker=dict(size=4, color="#38BDF8")
    ))

    # Diagonal chance line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Guess Chance (AUC = 0.50)",
        line=dict(color="#64748B", width=1.5, dash="dash")
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(family="Inter, -apple-system, sans-serif", size=13, color="#00D8F6")),
        xaxis=dict(
            title="False Positive Rate (1 - Specificity)",
            range=[-0.02, 1.02],
            showgrid=True,
            gridcolor="rgba(0, 216, 246, 0.08)"
        ),
        yaxis=dict(
            title="True Positive Rate (Sensitivity / Recall)",
            range=[-0.02, 1.02],
            showgrid=True,
            gridcolor="rgba(0, 216, 246, 0.08)"
        ),
        height=340,
        **OBSERVATORY_LAYOUT
    )
    return fig


def plot_precision_recall_curve(
    precision: List[float],
    recall: List[float],
    avg_precision: Optional[float] = None,
    baseline_prevalence: Optional[float] = None,
    title: str = "EMPIRICAL PRECISION-RECALL (PR) CURVE"
) -> go.Figure:
    """
    Renders empirical Precision-Recall curve derived from real prediction probabilities.
    Includes horizontal baseline prevalence line.
    """
    fig = go.Figure()

    ap_str = f" (AP = {avg_precision:.4f})" if avg_precision is not None else ""
    fig.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode="lines+markers",
        name=f"Empirical PR{ap_str}",
        line=dict(color="#10B981", width=2.5),
        marker=dict(size=4, color="#34D399")
    ))

    if baseline_prevalence is not None:
        fig.add_hline(
            y=baseline_prevalence,
            line=dict(color="#F59E0B", width=1.5, dash="dash"),
            annotation_text=f"Class Prevalence ({baseline_prevalence*100:.2f}%)",
            annotation_font=dict(color="#F59E0B", size=10)
        )

    fig.update_layout(
        title=dict(text=title, font=dict(family="Inter, -apple-system, sans-serif", size=13, color="#00D8F6")),
        xaxis=dict(
            title="Recall (Sensitivity)",
            range=[-0.02, 1.02],
            showgrid=True,
            gridcolor="rgba(0, 216, 246, 0.08)"
        ),
        yaxis=dict(
            title="Precision (Positive Predictive Value)",
            range=[-0.02, 1.02],
            showgrid=True,
            gridcolor="rgba(0, 216, 246, 0.08)"
        ),
        height=340,
        **OBSERVATORY_LAYOUT
    )
    return fig

