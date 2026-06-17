"""Cross-metric correlation figures.

For each channel, correlate the per-bin single-channel metrics against each
other (Spearman, robust to outliers and monotonic-but-nonlinear relationships)
and render the correlation matrix as a heatmap. This surfaces structure that the
per-metric distribution plots cannot show -- e.g. whether bins with longer
periods also tend to have larger amplitudes or steeper rising edges.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from waveanalysis.housekeeping.housekeeping_functions import get_channel_name
from waveanalysis.plotting.style import style_context, apply_dark

# Minimum paired observations before a correlation is computed for a cell.
_MIN_PAIRS = 3

# Single-channel per-bin metrics, grouped by family for a readable axis order.
# Inter-channel (combo) metrics like shifts are intentionally excluded -- they
# have a different shape and a different meaning. Only those present in a given
# run are used.
_SINGLE_CHANNEL_METRICS = [
    'Period',
    'Peak Amp', 'Peak Rel Amp', 'Peak Max', 'Peak Min', 'Peak Width',
    'Peak Area', 'Peak Offset',
    'Rise Duration', 'Fall Duration', 'Rise minus Fall Duration',
    'Rising Slope', 'Falling Slope', 'Max Rising Slope', 'Max Falling Slope',
    'Rising/Falling Slope Ratio',
]


def plot_metric_correlation_workflow(
    img_metrics: dict,
    img_props: dict,
    dark_plots: bool = False
) -> dict:
    """
    Build a per-bin cross-metric Spearman correlation heatmap for each channel.

    Parameters:
        img_metrics (dict): metric name -> array of shape (num_channels, num_bins).
        img_props (dict): image properties (needs 'num_channels').
        dark_plots (bool): dark theme toggle.

    Returns:
        dict: { 'Ch N Metric Correlations': Figure } for channels with enough data.
    """
    num_channels = img_props['num_channels']
    metric_names = [m for m in _SINGLE_CHANNEL_METRICS if m in img_metrics]

    corr_figs = {}
    for channel in range(num_channels):
        columns = [np.asarray(img_metrics[m][channel], dtype=float) for m in metric_names]
        corr = _spearman_matrix(columns)
        fig = _return_correlation_figure(
            corr,
            labels=metric_names,
            channel_name=get_channel_name(img_props.get('channel_names'), channel),
            dark_plots=dark_plots,
        )
        if fig is not None:
            corr_figs[f'Ch {channel + 1} Metric Correlations'] = fig

    return corr_figs


def _spearman_matrix(columns: list) -> np.ndarray:
    """
    Pairwise Spearman correlation matrix. Each pair is computed on the bins
    where both metrics are finite (pairwise-complete), so a few NaN bins don't
    drop the whole row. Cells without enough paired data or with no variance
    are left as NaN.
    """
    n_metrics = len(columns)
    corr = np.full((n_metrics, n_metrics), np.nan)
    for i in range(n_metrics):
        for j in range(i, n_metrics):
            a, b = columns[i], columns[j]
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() >= _MIN_PAIRS and np.std(a[mask]) > 0 and np.std(b[mask]) > 0:
                rho, _ = stats.spearmanr(a[mask], b[mask])
                corr[i, j] = corr[j, i] = rho
    np.fill_diagonal(corr, 1.0)
    return corr


def _return_correlation_figure(
    corr: np.ndarray,
    labels: list,
    channel_name: str,
    dark_plots: bool = False
) -> plt.Figure:
    """Render a correlation matrix as an annotated heatmap, or None if empty."""
    # Nothing meaningful to show if every off-diagonal cell is NaN.
    off_diag = corr[~np.eye(len(labels), dtype=bool)]
    if off_diag.size == 0 or np.all(np.isnan(off_diag)):
        return None

    n = len(labels)
    with style_context(dark_plots):
        size = 0.6 * n + 3
        fig, ax = plt.subplots(figsize=(size, size), constrained_layout=True)
        apply_dark(fig, ax, dark_plots)

        im = ax.imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r')

        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=8)

        # Annotate each cell with the correlation; white text on strong cells.
        for i in range(n):
            for j in range(n):
                value = corr[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                            fontsize=6, color='white' if abs(value) > 0.6 else 'black')

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Spearman ρ')
        ax.set_title(f'{channel_name}: cross-metric correlations (per bin)')

        plt.close(fig)

    return fig
