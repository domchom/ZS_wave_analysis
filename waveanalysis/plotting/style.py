"""Shared plotting helpers and theme.

Centralizes the bits every figure used to re-implement by hand: NaN filtering,
dark-mode background toggling, and the box+points "raincloud" distribution
panel that replaces the old histogram+boxplot pairs.

Using these keeps every figure visually consistent and cuts a lot of repeated
boilerplate. Prefer `constrained_layout=True` on the figure (handled by the
caller) over manual `subplots_adjust`.
"""
import numpy as np
import matplotlib.pyplot as plt

# Minimum sample size before a violin (KDE) is drawn behind the box. Below
# this, a KDE is more misleading than informative, so we show box+points only.
_MIN_VIOLIN_N = 10


def style_context(dark_plots: bool = False):
    """Return the matplotlib style context to use for a figure."""
    return plt.style.context('dark_background' if dark_plots else 'default')


def clean_array(values) -> np.ndarray:
    """Return a flat float array with NaN/inf removed."""
    arr = np.asarray(values, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def apply_dark(fig: plt.Figure, axes, dark_plots: bool = False) -> None:
    """Force black backgrounds for a figure and its axes when in dark mode."""
    if not dark_plots:
        return
    fig.patch.set_facecolor('black')
    for ax in np.atleast_1d(np.asarray(axes, dtype=object)).ravel():
        ax.set_facecolor('black')


def raincloud(
    ax: plt.Axes,
    data: list,
    labels: list,
    colors: list = None,
    dark_plots: bool = False,
    point_color: str = None,
    show_violin: bool = True,
    zero_line: bool = False,
    annotate_n: bool = True,
    rotation: float = 0,
) -> None:
    """Distribution panel that replaces the old histogram + boxplot pair.

    For each category it draws a box (no fliers) with the raw points jittered
    on top, plus a faint violin behind it when there are enough points. This
    shows distribution shape, the five-number summary, and the sample size in
    a single panel.

    Parameters:
        ax: target axes.
        data: list of 1-D array-likes, one per category (NaNs are dropped).
        labels: x-axis label per category.
        colors: optional per-category color for the box edge / violin.
        dark_plots: dark theme toggle.
        point_color: override for the jittered points (defaults by theme).
        show_violin: draw a faint violin behind boxes with >= _MIN_VIOLIN_N pts.
        zero_line: draw a dashed horizontal line at y=0 (useful for signed metrics).
        annotate_n: append "(n=...)" to each category label.
        rotation: x tick label rotation.
    """
    groups = [clean_array(g) for g in data]
    n_groups = len(groups)
    colors = list(colors) if colors is not None else [None] * n_groups
    pts_color = point_color or ('lightblue' if dark_plots else '0.25')
    positions = np.arange(1, n_groups + 1)

    # Faint violin behind the box, only where there are enough points for a
    # KDE to be meaningful.
    if show_violin:
        for pos, vals, col in zip(positions, groups, colors):
            if vals.size >= _MIN_VIOLIN_N:
                parts = ax.violinplot(
                    [vals], positions=[pos], widths=0.7,
                    showmeans=False, showmedians=False, showextrema=False,
                )
                for body in parts['bodies']:
                    body.set_facecolor(col or ('gray' if dark_plots else 'lightgray'))
                    body.set_alpha(0.25)

    # Box (no fliers): outline only so the points and violin stay visible.
    boxes = ax.boxplot(
        groups, positions=positions, widths=0.25,
        showfliers=False, patch_artist=True,
    )
    for box, col in zip(boxes['boxes'], colors):
        box.set_facecolor('none')
        if col:
            box.set_edgecolor(col)

    # Raw points jittered on top.
    for pos, vals in zip(positions, groups):
        if vals.size == 0:
            continue
        jitter = (np.random.rand(vals.size) - 0.5) * 0.12
        ax.scatter(
            np.full(vals.size, pos) + jitter, vals,
            color=pts_color, alpha=0.5, s=12, zorder=3,
        )

    if zero_line:
        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')

    if annotate_n:
        labels = [f'{lab} (n={vals.size})' for lab, vals in zip(labels, groups)]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=rotation, ha='right' if rotation else 'center')
