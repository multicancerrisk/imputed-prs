"""Diagnostic plotting functions for PRS imputation model visualization."""

from typing import TYPE_CHECKING, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from imputed_prs.core.types import ImputedVariantModel, PredictionResult

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

# Quality tier colors matching summarize_imputation_quality()
QUALITY_TIER_COLORS = {
    "excellent": "#2ecc71",  # green
    "good": "#3498db",  # blue
    "moderate": "#f39c12",  # orange
    "poor": "#e74c3c",  # red
}

# Quality tier boundaries
QUALITY_TIER_BOUNDARIES = {
    "excellent": 0.8,
    "good": 0.6,
    "moderate": 0.4,
}


def _import_matplotlib():
    """Import matplotlib, raising informative error if not installed."""
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting functions. "
            "Install it with: pip install imputed-prs[plotting]"
        )


def _get_quality_tier(r2: float) -> str:
    """Return quality tier name for a given R² value.

    Args:
        r2: Imputation R² value.

    Returns:
        Quality tier name: "excellent", "good", "moderate", or "poor".
    """
    if r2 > 0.8:
        return "excellent"
    elif r2 > 0.6:
        return "good"
    elif r2 > 0.4:
        return "moderate"
    else:
        return "poor"


def _get_quality_tier_color(r2: float) -> str:
    """Return color for R² quality tier.

    Args:
        r2: Imputation R² value.

    Returns:
        Hex color string for the quality tier.
    """
    tier = _get_quality_tier(r2)
    return QUALITY_TIER_COLORS[tier]


def _create_figure_if_needed(
    ax: Optional["matplotlib.axes.Axes"],
) -> Tuple["matplotlib.figure.Figure", "matplotlib.axes.Axes"]:
    """Create figure/axes if ax is None.

    Args:
        ax: Optional existing axes to use.

    Returns:
        Tuple of (figure, axes).
    """
    plt = _import_matplotlib()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()
    return fig, ax


def plot_calibration(
    s_imputed: np.ndarray,
    s_true: np.ndarray,
    ax: Optional["matplotlib.axes.Axes"] = None,
    title: Optional[str] = None,
    show_identity: bool = True,
    show_regression: bool = True,
    alpha: float = 0.5,
) -> Tuple["matplotlib.figure.Figure", "matplotlib.axes.Axes"]:
    """Create calibration scatter plot comparing imputed vs true PRS.

    Generates a scatter plot of imputed PRS values against true PRS values
    with optional identity line (y=x) and regression line.

    Args:
        s_imputed: Array of imputed PRS values.
        s_true: Array of true PRS values.
        ax: Optional matplotlib axes to plot on. If None, creates new figure.
        title: Optional plot title. Defaults to "PRS Calibration".
        show_identity: Whether to show y=x identity line. Defaults to True.
        show_regression: Whether to show regression line. Defaults to True.
        alpha: Transparency for scatter points. Defaults to 0.5.

    Returns:
        Tuple of (figure, axes) for further customization or saving.

    Raises:
        ValueError: If input arrays have different lengths or fewer than 2 samples.
    """
    s_imputed = np.asarray(s_imputed)
    s_true = np.asarray(s_true)

    if len(s_imputed) != len(s_true):
        raise ValueError(
            f"Input arrays must have same length. "
            f"Got s_imputed={len(s_imputed)}, s_true={len(s_true)}"
        )

    if len(s_imputed) < 2:
        raise ValueError("Need at least 2 samples for calibration plot")

    fig, ax = _create_figure_if_needed(ax)

    # Scatter plot
    ax.scatter(s_imputed, s_true, alpha=alpha, edgecolors="none", label="Samples")

    # Get axis limits for lines
    all_values = np.concatenate([s_imputed, s_true])
    margin = 0.05 * (np.max(all_values) - np.min(all_values))
    min_val = np.min(all_values) - margin
    max_val = np.max(all_values) + margin

    # Identity line
    if show_identity:
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            color="gray",
            linestyle="--",
            linewidth=1,
            label="Identity (y=x)",
        )

    # Regression line and statistics
    slope, intercept, r_value, p_value, std_err = stats.linregress(s_imputed, s_true)
    r2 = r_value**2

    if show_regression:
        x_line = np.array([min_val, max_val])
        y_line = slope * x_line + intercept
        ax.plot(
            x_line, y_line, color="red", linewidth=1.5, label=f"Regression (R²={r2:.3f})"
        )

    # Annotation
    n_samples = len(s_imputed)
    annotation_text = f"R² = {r2:.3f}\nSlope = {slope:.3f}\nn = {n_samples}"
    ax.annotate(
        annotation_text,
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Labels and title
    ax.set_xlabel("Imputed PRS")
    ax.set_ylabel("True PRS")
    ax.set_title(title or "PRS Calibration")
    ax.legend(loc="lower right")

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    return fig, ax


def plot_imputation_quality(
    models: List[ImputedVariantModel],
    ax: Optional["matplotlib.axes.Axes"] = None,
    title: Optional[str] = None,
    bins: int = 20,
    show_tiers: bool = True,
) -> Tuple["matplotlib.figure.Figure", "matplotlib.axes.Axes"]:
    """Create histogram of imputation R² values by quality tier.

    Visualizes the distribution of imputation quality across all variant
    models, with colors indicating quality tiers.

    Args:
        models: List of ImputedVariantModel instances.
        ax: Optional matplotlib axes to plot on. If None, creates new figure.
        title: Optional plot title. Defaults to "Imputation Quality Distribution".
        bins: Number of histogram bins. Defaults to 20.
        show_tiers: Whether to show tier boundary lines. Defaults to True.

    Returns:
        Tuple of (figure, axes) for further customization or saving.
    """
    fig, ax = _create_figure_if_needed(ax)

    if not models:
        warnings.warn("Empty models list provided to plot_imputation_quality")
        ax.set_xlabel("Imputation R²")
        ax.set_ylabel("Count")
        ax.set_title(title or "Imputation Quality Distribution")
        ax.set_xlim(0, 1)
        ax.text(
            0.5,
            0.5,
            "No models to display",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig, ax

    # Extract R² values
    r2_values = np.array([m.imputation_r2 for m in models])

    # Count by tier
    tier_counts = {
        "excellent": np.sum(r2_values > 0.8),
        "good": np.sum((r2_values > 0.6) & (r2_values <= 0.8)),
        "moderate": np.sum((r2_values > 0.4) & (r2_values <= 0.6)),
        "poor": np.sum(r2_values <= 0.4),
    }

    # Create histogram with colored bars by tier
    bin_edges = np.linspace(0, 1, bins + 1)
    counts, _ = np.histogram(r2_values, bins=bin_edges)

    # Color each bar based on its bin center's quality tier
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    colors = [_get_quality_tier_color(bc) for bc in bin_centers]

    ax.bar(
        bin_centers,
        counts,
        width=bin_edges[1] - bin_edges[0],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )

    # Add tier boundary lines
    if show_tiers:
        for boundary in [0.4, 0.6, 0.8]:
            ax.axvline(
                x=boundary, color="black", linestyle="--", linewidth=1, alpha=0.7
            )

    # Legend with tier counts
    legend_elements = []
    plt = _import_matplotlib()
    from matplotlib.patches import Patch

    for tier_name in ["excellent", "good", "moderate", "poor"]:
        count = tier_counts[tier_name]
        legend_elements.append(
            Patch(
                facecolor=QUALITY_TIER_COLORS[tier_name],
                edgecolor="black",
                label=f"{tier_name.capitalize()}: {count}",
            )
        )
    ax.legend(handles=legend_elements, loc="upper left")

    # Labels and title
    ax.set_xlabel("Imputation R²")
    ax.set_ylabel("Count")
    ax.set_title(title or "Imputation Quality Distribution")
    ax.set_xlim(0, 1)

    return fig, ax


def plot_variance_contribution(
    models: List[ImputedVariantModel],
    top_n: int = 20,
    ax: Optional["matplotlib.axes.Axes"] = None,
    title: Optional[str] = None,
    color_by_quality: bool = True,
) -> Tuple["matplotlib.figure.Figure", "matplotlib.axes.Axes"]:
    """Create bar chart of top variance-contributing variants.

    Shows the variants that contribute most to prediction variance,
    calculated as beta² × residual_variance.

    Args:
        models: List of ImputedVariantModel instances.
        top_n: Number of top contributing variants to show. Defaults to 20.
        ax: Optional matplotlib axes to plot on. If None, creates new figure.
        title: Optional plot title. Defaults to "Top Variance Contributors".
        color_by_quality: Whether to color bars by quality tier. Defaults to True.

    Returns:
        Tuple of (figure, axes) for further customization or saving.
    """
    fig, ax = _create_figure_if_needed(ax)

    if not models:
        warnings.warn("Empty models list provided to plot_variance_contribution")
        ax.set_xlabel("Variance Contribution")
        ax.set_ylabel("Variant")
        ax.set_title(title or "Top Variance Contributors")
        ax.text(
            0.5,
            0.5,
            "No models to display",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig, ax

    # Calculate variance contribution for each model
    contributions = []
    for m in models:
        var_contrib = (m.beta**2) * m.residual_variance
        # Create label: use variant_id or fallback to CHR:POS
        label = m.variant_id if m.variant_id else f"{m.chromosome}:{m.position}"
        contributions.append(
            {
                "label": label,
                "contribution": var_contrib,
                "r2": m.imputation_r2,
            }
        )

    # Sort by contribution descending and take top_n
    contributions.sort(key=lambda x: x["contribution"], reverse=True)
    contributions = contributions[:top_n]

    # Reverse for horizontal bar chart (highest at top)
    contributions = contributions[::-1]

    labels = [c["label"] for c in contributions]
    values = [c["contribution"] for c in contributions]

    if color_by_quality:
        colors = [_get_quality_tier_color(c["r2"]) for c in contributions]
    else:
        colors = "#3498db"  # default blue

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Variance Contribution")
    ax.set_ylabel("Variant")
    ax.set_title(title or "Top Variance Contributors")

    # Add legend for quality tiers if coloring by quality
    if color_by_quality:
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(
                facecolor=QUALITY_TIER_COLORS[tier],
                edgecolor="black",
                label=tier.capitalize(),
            )
            for tier in ["excellent", "good", "moderate", "poor"]
        ]
        ax.legend(handles=legend_elements, loc="lower right")

    return fig, ax


def plot_truncation_diagnostics(
    predictions: Union[List[PredictionResult], pd.DataFrame],
    ax: Optional["matplotlib.axes.Axes"] = None,
    title: Optional[str] = None,
) -> Tuple["matplotlib.figure.Figure", "matplotlib.axes.Axes"]:
    """Visualize dosage truncation patterns across predictions.

    Creates a histogram showing the distribution of truncated dosage counts
    across samples, with summary statistics.

    Args:
        predictions: Either a list of PredictionResult objects or a DataFrame
            with 'n_truncated' and 'n_variants_imputed' columns.
        ax: Optional matplotlib axes to plot on. If None, creates new figure.
        title: Optional plot title. Defaults to "Truncation Diagnostics".

    Returns:
        Tuple of (figure, axes) for further customization or saving.

    Raises:
        ValueError: If predictions is empty or has invalid format.
    """
    fig, ax = _create_figure_if_needed(ax)

    # Extract truncation counts
    if isinstance(predictions, pd.DataFrame):
        if "n_truncated" not in predictions.columns:
            raise ValueError("DataFrame must have 'n_truncated' column")
        truncation_counts = predictions["n_truncated"].values
    elif isinstance(predictions, list):
        if not predictions:
            raise ValueError("predictions list cannot be empty")
        truncation_counts = np.array([p.n_truncated for p in predictions])
    else:
        raise ValueError(
            "predictions must be a list of PredictionResult or a DataFrame"
        )

    if len(truncation_counts) == 0:
        raise ValueError("No truncation data available")

    # Calculate statistics
    mean_truncated = np.mean(truncation_counts)
    median_truncated = np.median(truncation_counts)
    pct_with_truncation = 100 * np.mean(truncation_counts > 0)
    max_truncated = np.max(truncation_counts)

    # Create histogram
    if max_truncated == 0:
        # All zeros - show a single bar
        ax.bar([0], [len(truncation_counts)], color="#2ecc71", edgecolor="black")
        ax.set_xlim(-0.5, 1)
    else:
        # Determine bins
        n_bins = min(20, int(max_truncated) + 1)
        ax.hist(
            truncation_counts,
            bins=n_bins,
            color="#3498db",
            edgecolor="black",
            linewidth=0.5,
        )

    # Add summary annotation
    annotation_text = (
        f"Mean: {mean_truncated:.1f}\n"
        f"Median: {median_truncated:.0f}\n"
        f"Samples with truncation: {pct_with_truncation:.1f}%\n"
        f"n = {len(truncation_counts)}"
    )
    ax.annotate(
        annotation_text,
        xy=(0.95, 0.95),
        xycoords="axes fraction",
        verticalalignment="top",
        horizontalalignment="right",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Labels and title
    ax.set_xlabel("Number of Truncated Dosages")
    ax.set_ylabel("Count")
    ax.set_title(title or "Truncation Diagnostics")

    return fig, ax
