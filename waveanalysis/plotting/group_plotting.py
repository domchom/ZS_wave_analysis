import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from .mean_plot_creation import (
    CCF_SHIFT_NOTE,
    PHASE_SHIFT_NOTE,
    landmark_metric_note,
    _add_figure_note,
    _annotate_lead_direction,
)
from .style import style_context, apply_dark
from waveanalysis.housekeeping.housekeeping_functions import relabel_metric_text, get_channel_name

# Metrics whose value is a signed inter-channel offset, so a positive/negative
# note is meaningful. Matched as a substring of the column name.
_SIGNED_METRIC_KEYS = ('Shift', 'Diff')

# Minimum observations per group before a statistical test is run.
_MIN_N_FOR_STATS = 3

# Substring identifying detection-failure columns: the percentage of bins with
# no measurable period/peak/shift. These are quality signals (higher = weaker,
# noisier signal) rather than biological measurements.
_DETECTION_FAILURE_KEY = 'Pcnt No'

# A group whose median detection-failure rate exceeds this is flagged: its wave
# metrics rest on relatively few usable bins.
_DETECTION_CAUTION_PCNT = 50.0


def _resolve_display_and_order(
    summary_df: pd.DataFrame,
    group_order: list = None,
    group_labels: dict = None,
) -> tuple:
    '''Apply display-name overrides and resolve left-to-right group order.

    Returns (possibly-relabeled copy of summary_df, ordered list of group names).
    Renaming is applied on a copy so the caller's frame is untouched, and the
    returned order is expressed in the (possibly renamed) display space. Order is
    caller-specified first, then any remaining groups in order of first
    appearance (more meaningful than seaborn's alphabetical default).
    '''
    if group_labels:
        summary_df = summary_df.copy()
        summary_df['Group Name'] = summary_df['Group Name'].map(
            lambda g: group_labels.get(g, g)
        )
        if group_order is not None:
            group_order = [group_labels.get(g, g) for g in group_order]

    present_groups = list(dict.fromkeys(summary_df['Group Name'].tolist()))
    if group_order is not None:
        order = [g for g in group_order if g in present_groups]
        order += [g for g in present_groups if g not in order]
    else:
        order = present_groups
    return summary_df, order


def _p_to_stars(p: float) -> str:
    '''Conventional significance markers for a p-value.'''
    if p < 1e-4:
        return '****'
    if p < 1e-3:
        return '***'
    if p < 1e-2:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


def _add_significance_bracket(ax, x1, x2, y, text, dark_plots=False):
    '''Draw a significance bracket spanning x1..x2 at height y with a label.'''
    color = 'white' if dark_plots else 'black'
    h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, color=color)
    ax.text((x1 + x2) / 2, y + h, text, ha='center', va='bottom',
            color=color, fontsize=10)


def _group_comparison_stats(ax, groups_data, dark_plots):
    '''Run a non-parametric group test and annotate the axes.

    Two groups -> Mann-Whitney U with a rank-biserial effect size and a
    significance bracket. More than two -> Kruskal-Wallis omnibus test. Returns
    a short text summary for the title, or None if a test could not be run.
    '''
    ns = [len(d) for d in groups_data]
    if any(n < _MIN_N_FOR_STATS for n in ns):
        return None
    try:
        if len(groups_data) == 2:
            u_stat, p = stats.mannwhitneyu(
                groups_data[0], groups_data[1], alternative='two-sided'
            )
            # rank-biserial correlation: 0 = full overlap, ±1 = full separation
            rbc = 1 - (2.0 * u_stat) / (ns[0] * ns[1])
            dmax = max(np.nanmax(groups_data[0]), np.nanmax(groups_data[1]))
            dmin = min(np.nanmin(groups_data[0]), np.nanmin(groups_data[1]))
            span = (dmax - dmin) or 1.0
            ax.set_ylim(top=dmax + 0.20 * span)
            _add_significance_bracket(ax, 0, 1, dmax + 0.08 * span,
                                      _p_to_stars(p), dark_plots)
            return f'Mann–Whitney p={p:.3g} ({_p_to_stars(p)}), rank-biserial r={rbc:.2f}'
        else:
            _, p = stats.kruskal(*groups_data)
            return f'Kruskal–Wallis p={p:.3g} ({_p_to_stars(p)})'
    except ValueError:
        # e.g. all values identical -> test undefined
        return None


def generate_group_comparison(
    summary_df: pd.DataFrame,
    log_params: dict,
    dark_plots: bool = False,
    channel_names: list = None,
    edge_height_fraction: float = 0.5,
    group_order: list = None,
    group_labels: dict = None,
    add_stats: bool = True,
) -> dict:
    """
    Generate group comparison plots for each parameter in the summary dataframe.

    Each plot shows a box + swarm per group with the sample size in the x label
    and, when there are enough observations, a non-parametric significance test
    (Mann-Whitney for two groups, Kruskal-Wallis for more).

    Parameters:
        summary_df (pd.DataFrame): The summary dataframe containing the data for comparison.
                                   Must contain a column 'Group Name' and columns with 'Mean' in their names.
        log_params (dict): A dictionary to log any errors encountered during plotting.
                           Expects a key 'Plotting errors' with a list as value.
        dark_plots (bool): If True, use a dark theme with black background.
        group_order (list): Explicit left-to-right group order, given as the
                            original group names. Defaults to order of first
                            appearance in the dataframe.
        group_labels (dict): Optional mapping of original group name -> display
                             name. Renaming is applied before ordering and is
                             reflected in the x-axis tick labels.
        add_stats (bool): If True, annotate a non-parametric group test.

    Returns:
        dict: A dictionary mapping parameter name -> matplotlib Figure.
    """
    print('Generating group comparisons...')
    group_mean_parameter_figs = {}

    # get the parameters to compare
    parameters_to_compare = [column for column in summary_df.columns if 'Mean' in column]

    summary_df, order = _resolve_display_and_order(summary_df, group_order, group_labels)

    point_color = 'lightgray' if dark_plots else '.25'

    for param in parameters_to_compare:
        try:
            # skip if column is completely NaN or empty after dropna
            if summary_df[param].dropna().empty:
                raise ValueError("No data to compare")

            with style_context(dark_plots):
                fig, ax = plt.subplots()
                apply_dark(fig, ax, dark_plots)

                # boxplot (colorblind-safe palette)
                sns.boxplot(
                    x='Group Name',
                    y=param,
                    data=summary_df,
                    order=order,
                    showfliers=False,
                    palette='colorblind',
                    ax=ax
                )

                # swarmplot (jittered points)
                sns.swarmplot(
                    x='Group Name',
                    y=param,
                    data=summary_df,
                    order=order,
                    color=point_color,
                    ax=ax
                )

                # Per-group values and sample sizes
                groups_data = [
                    summary_df.loc[summary_df['Group Name'] == g, param].dropna().values
                    for g in order
                ]

                # Non-parametric group test (drawn before tick labels so any
                # y-limit headroom for the bracket is already applied).
                stat_line = None
                if add_stats and len(order) >= 2:
                    stat_line = _group_comparison_stats(ax, groups_data, dark_plots)

                display_param = relabel_metric_text(param, channel_names, edge_height_fraction)
                title = f'Group comparison: {display_param}'
                if stat_line:
                    title += f'\n{stat_line}'
                ax.set_title(title, fontsize=10)
                ax.set_xlabel('Group')
                ax.set_ylabel(display_param)

                # Sample size per group in the tick labels
                ax.set_xticks(range(len(order)))
                ax.set_xticklabels(
                    [f'{g}\n(n={len(d)})' for g, d in zip(order, groups_data)],
                    rotation=45, ha='right'
                )

                # Signed offset metrics get metric-specific captions; others
                # (durations, widths, periods, amplitudes) have no +/- meaning.
                if any(key in param for key in _SIGNED_METRIC_KEYS):
                    # Mark which channel leads at each end of the axis for actual
                    # shift metrics (a "Diff" is a difference of shifts, so the
                    # leading/trailing framing does not apply to it).
                    combo_match = re.search(r'Ch(\d+)-Ch(\d+)', param)
                    if 'Shift' in param and 'Diff' not in param and combo_match:
                        ch1_idx = int(combo_match.group(1)) - 1
                        ch2_idx = int(combo_match.group(2)) - 1
                        _annotate_lead_direction(
                            ax,
                            get_channel_name(channel_names, ch1_idx),
                            get_channel_name(channel_names, ch2_idx),
                            axis='y',
                            dark_plots=dark_plots,
                        )
                    note = landmark_metric_note(param, edge_height_fraction)
                    if not note and '% Phase Shift' in param:
                        note = PHASE_SHIFT_NOTE
                    elif not note and 'Shift' in param:
                        note = CCF_SHIFT_NOTE
                    if note:
                        _add_figure_note(fig, note, dark_plots, bottom=0.30)
                    else:
                        fig.tight_layout()
                else:
                    fig.tight_layout()
                group_mean_parameter_figs[param] = fig
                plt.close(fig)

        except ValueError:
            # make sure the key exists
            if 'Plotting errors' not in log_params:
                log_params['Plotting errors'] = []
            log_params['Plotting errors'].append(f'No data to compare for {param}')

    return group_mean_parameter_figs


def generate_group_metric_scatter(
    summary_df: pd.DataFrame,
    x_param: str,
    y_param: str,
    dark_plots: bool = False,
    channel_names: list = None,
    edge_height_fraction: float = None,
    add_stats: bool = False,
) -> plt.Figure:
    """
    Generate one grouped scatter plot comparing two summary metrics.

    Each point is one processed image/sample. Groups are overlaid on the same
    axes and separated by both color and marker.
    """
    required = {'Group Name', x_param, y_param}
    missing = [col for col in required if col not in summary_df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    plot_df = summary_df[['Group Name', x_param, y_param]].copy()
    plot_df[x_param] = pd.to_numeric(plot_df[x_param], errors='coerce')
    plot_df[y_param] = pd.to_numeric(plot_df[y_param], errors='coerce')
    plot_df = plot_df.dropna(subset=[x_param, y_param])
    if plot_df.empty:
        raise ValueError("No paired numeric data for the selected metrics")

    order = list(dict.fromkeys(plot_df['Group Name'].astype(str).tolist()))
    markers = ['o', 's', '^', 'D', 'P', 'X', 'v', '<', '>', '*']
    marker_map = {group: markers[i % len(markers)] for i, group in enumerate(order)}
    palette = dict(zip(order, sns.color_palette('colorblind', n_colors=len(order))))

    x_label = relabel_metric_text(x_param, channel_names, edge_height_fraction)
    y_label = relabel_metric_text(y_param, channel_names, edge_height_fraction)

    with style_context(dark_plots):
        fig, ax = plt.subplots(figsize=(6.2, 5) if add_stats else (7, 5))
        apply_dark(fig, ax, dark_plots)

        sns.scatterplot(
            data=plot_df,
            x=x_param,
            y=y_param,
            hue='Group Name',
            style='Group Name',
            hue_order=order,
            style_order=order,
            markers=marker_map,
            palette=palette,
            s=70,
            edgecolor='black' if not dark_plots else 'white',
            linewidth=0.5,
            ax=ax,
        )

        ax.set_title(f'Group scatter:\n{x_label} vs {y_label}', fontsize=10)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25)

        stats_text = None
        if add_stats:
            stats_text = _add_scatter_fit_stats(
                ax,
                plot_df,
                x_param=x_param,
                y_param=y_param,
                group_order=order,
                palette=palette,
                dark_plots=dark_plots,
            )

        legend = ax.legend(title='Group', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        legend.set_in_layout(False)
        if stats_text:
            fig.subplots_adjust(left=0.14, right=0.72, top=0.84, bottom=0.14)
            _add_scatter_stats_text(ax, stats_text, dark_plots)
        else:
            fig.tight_layout()
        plt.close(fig)

    return fig


def _add_scatter_fit_stats(
    ax,
    plot_df: pd.DataFrame,
    x_param: str,
    y_param: str,
    group_order: list,
    palette: dict,
    dark_plots: bool = False,
) -> str:
    """
    Add an overall linear fit and correlation summary to a scatter axis.

    R² comes from ordinary least-squares linear regression. Pearson describes
    linear association; Spearman describes monotonic association and is less
    sensitive to non-linear scaling/outliers.
    """
    x_values = np.asarray(plot_df[x_param].values, dtype=float)
    y_values = np.asarray(plot_df[y_param].values, dtype=float)
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[mask]
    y_values = y_values[mask]
    if len(x_values) < 3 or np.std(x_values) == 0 or np.std(y_values) == 0:
        return f'n={len(x_values)}\nNot enough variation for fit'

    lin = stats.linregress(x_values, y_values)
    r_squared = lin.rvalue ** 2
    rho, spearman_p = stats.spearmanr(x_values, y_values)

    x_fit = np.linspace(np.nanmin(x_values), np.nanmax(x_values), 100)
    fit_color = 'white' if dark_plots else 'black'
    ax.plot(x_fit, lin.intercept + lin.slope * x_fit, color=fit_color, linewidth=1.7,
            linestyle='--', alpha=0.85, label='Overall fit')

    group_lines = []
    for group in group_order:
        group_df = plot_df.loc[plot_df['Group Name'].astype(str) == str(group)]
        gx = pd.to_numeric(group_df[x_param], errors='coerce').values
        gy = pd.to_numeric(group_df[y_param], errors='coerce').values
        gmask = np.isfinite(gx) & np.isfinite(gy)
        gx = gx[gmask]
        gy = gy[gmask]
        if len(gx) < 3 or np.std(gx) == 0 or np.std(gy) == 0:
            group_lines.append(f'{group}: R² n/a')
            continue
        glin = stats.linregress(gx, gy)
        gx_fit = np.linspace(np.nanmin(gx), np.nanmax(gx), 100)
        ax.plot(
            gx_fit,
            glin.intercept + glin.slope * gx_fit,
            color=palette.get(group, fit_color),
            linewidth=1.3,
            linestyle=':',
            alpha=0.9,
            label=f'{group} fit',
        )
        group_lines.append(f'{group}: R²={glin.rvalue ** 2:.2f}')

    text_color = 'lightgray' if dark_plots else 'dimgray'
    box_face = 'black' if dark_plots else 'white'
    stat_text = (
        f'n={len(x_values)}\n'
        f'Overall R²={r_squared:.2f}\n'
        f'Pearson r={lin.rvalue:.2f}, p={lin.pvalue:.3g}\n'
        f'Spearman ρ={rho:.2f}, p={spearman_p:.3g}'
    )
    if group_lines:
        stat_text += '\n' + '\n'.join(group_lines)
    return stat_text


def _add_scatter_stats_text(ax: plt.Axes, stat_text: str, dark_plots: bool = False) -> None:
    text_color = 'lightgray' if dark_plots else 'dimgray'
    box_face = 'black' if dark_plots else 'white'
    ax.text(
        1.02,
        0.40,
        stat_text,
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=8,
        color=text_color,
        bbox=dict(boxstyle='round,pad=0.3', facecolor=box_face, alpha=0.65, edgecolor='none'),
    )


def generate_group_metric_correlations(
    summary_df: pd.DataFrame,
    dark_plots: bool = False,
    channel_names: list = None,
    edge_height_fraction: float = None,
    min_pairs: int = 3,
) -> dict:
    """
    Generate Spearman cross-metric correlation heatmaps from the summary table,
    separately for each group.

    Each row in the summary table is one processed image/sample, and each metric
    column is a per-image mean measurement.
    """
    if 'Group Name' not in summary_df.columns:
        raise ValueError("Missing required column: Group Name")

    metric_cols = [
        col for col in summary_df.columns
        if 'Mean' in col and pd.to_numeric(summary_df[col], errors='coerce').notna().any()
    ]
    if len(metric_cols) < 2:
        raise ValueError("At least two numeric mean metrics are required")

    work_df = summary_df[['Group Name'] + metric_cols].copy()
    for col in metric_cols:
        work_df[col] = pd.to_numeric(work_df[col], errors='coerce')

    figs = {}
    groups = list(dict.fromkeys(work_df['Group Name'].astype(str).tolist()))
    for group in groups:
        group_df = work_df.loc[work_df['Group Name'].astype(str) == group, metric_cols]
        usable_cols = [
            col for col in metric_cols
            if group_df[col].notna().sum() >= min_pairs and group_df[col].nunique(dropna=True) > 1
        ]
        if len(usable_cols) < 2:
            continue

        corr = _spearman_summary_matrix(group_df, usable_cols, min_pairs)
        off_diag = corr[~np.eye(len(usable_cols), dtype=bool)]
        if off_diag.size == 0 or np.all(np.isnan(off_diag)):
            continue

        labels = [relabel_metric_text(col, channel_names, edge_height_fraction) for col in usable_cols]
        figs[f'{group} Metric Correlations'] = _return_group_correlation_figure(
            corr,
            labels=labels,
            group_name=group,
            dark_plots=dark_plots,
        )

    if not figs:
        raise ValueError("No groups had enough paired metric data for correlations")
    return figs


def _spearman_summary_matrix(group_df: pd.DataFrame, cols: list, min_pairs: int) -> np.ndarray:
    corr = np.full((len(cols), len(cols)), np.nan)
    for i, col_a in enumerate(cols):
        corr[i, i] = 1.0
        for j, col_b in enumerate(cols[i + 1:], start=i + 1):
            paired = group_df[[col_a, col_b]].dropna()
            if len(paired) >= min_pairs and paired[col_a].std() > 0 and paired[col_b].std() > 0:
                rho, _ = stats.spearmanr(paired[col_a], paired[col_b])
                corr[i, j] = corr[j, i] = rho
    return corr


def _return_group_correlation_figure(
    corr: np.ndarray,
    labels: list,
    group_name: str,
    dark_plots: bool = False,
) -> plt.Figure:
    n = len(labels)
    with style_context(dark_plots):
        size = min(max(0.45 * n + 3, 7), 16)
        fig, ax = plt.subplots(figsize=(size, size), constrained_layout=True)
        apply_dark(fig, ax, dark_plots)

        im = ax.imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r')
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=7)

        for i in range(n):
            for j in range(n):
                value = corr[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                            fontsize=6, color='white' if abs(value) > 0.6 else 'black')

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Spearman rho')
        ax.set_title(f'{group_name}: metric correlations')
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Dataset-quality plots (group comparison level)
#
# These describe how trustworthy a group's data is, rather than its biology:
#  - detection failure: % of bins where no period/peak/shift was measurable
#  - coverage: number of bins contributing to each image's per-image means
#  - effective sample size: how many images actually carry a usable value for
#    each metric in each group (the "can I even run stats here?" map)
# ---------------------------------------------------------------------------


def _quality_box_swarm(
    summary_df: pd.DataFrame,
    param: str,
    order: list,
    ylabel: str,
    title: str,
    dark_plots: bool = False,
    caution_level: float = None,
    caution_label: str = None,
    add_stats: bool = True,
) -> plt.Figure:
    '''Draw one box + swarm-per-group figure for a single quality column.

    Mirrors the look of generate_group_comparison (colorblind boxes, jittered
    points, per-group n in the tick labels, optional non-parametric group test)
    but is aimed at quality columns. An optional caution band shades the region
    beyond caution_level to flag groups in the danger zone.
    '''
    point_color = 'lightgray' if dark_plots else '.25'
    with style_context(dark_plots):
        fig, ax = plt.subplots()
        apply_dark(fig, ax, dark_plots)

        sns.boxplot(
            x='Group Name', y=param, data=summary_df, order=order,
            showfliers=False, palette='colorblind', ax=ax,
        )
        sns.swarmplot(
            x='Group Name', y=param, data=summary_df, order=order,
            color=point_color, ax=ax,
        )

        groups_data = [
            summary_df.loc[summary_df['Group Name'] == g, param].dropna().values
            for g in order
        ]

        # Non-parametric group test (run before tick labels so any y-limit
        # headroom for the significance bracket is already applied).
        stat_line = None
        if add_stats and len(order) >= 2:
            stat_line = _group_comparison_stats(ax, groups_data, dark_plots)

        # Shade the caution region (e.g. >50% of bins with no detection).
        if caution_level is not None:
            top = max(ax.get_ylim()[1], caution_level)
            ax.axhspan(caution_level, top, color='red', alpha=0.08, zorder=0)
            ax.axhline(caution_level, color='red', alpha=0.45, lw=1, ls='--')
            if caution_label:
                ax.text(
                    0.99, caution_level, caution_label, transform=ax.get_yaxis_transform(),
                    ha='right', va='bottom', fontsize=7,
                    color='salmon' if dark_plots else 'firebrick',
                )

        full_title = title
        if stat_line:
            full_title += f'\n{stat_line}'
        ax.set_title(full_title, fontsize=10)
        ax.set_xlabel('Group')
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(
            [f'{g}\n(n={len(d)})' for g, d in zip(order, groups_data)],
            rotation=45, ha='right',
        )
        fig.tight_layout()
        plt.close(fig)
    return fig


def generate_group_detection_quality(
    summary_df: pd.DataFrame,
    log_params: dict,
    dark_plots: bool = False,
    channel_names: list = None,
    edge_height_fraction: float = 0.5,
    group_order: list = None,
    group_labels: dict = None,
    add_stats: bool = True,
) -> dict:
    '''
    Generate detection-failure comparisons per group.

    For every "Pcnt No ..." column (the percentage of bins where no period,
    peak, or shift could be measured) draw a box + swarm per group. A group
    sitting high here has weak/noisy signal, so its wave metrics rest on few
    usable bins. A caution band marks the >50% region.

    Returns a dict mapping column name -> matplotlib Figure.
    '''
    print('Generating detection-quality comparisons...')
    figs = {}
    if 'Group Name' not in summary_df.columns:
        return figs

    summary_df, order = _resolve_display_and_order(summary_df, group_order, group_labels)
    failure_cols = [c for c in summary_df.columns if _DETECTION_FAILURE_KEY in c]

    for param in failure_cols:
        try:
            if summary_df[param].dropna().empty:
                raise ValueError('No data to compare')
            display_param = relabel_metric_text(param, channel_names, edge_height_fraction)
            figs[param] = _quality_box_swarm(
                summary_df,
                param,
                order,
                ylabel=f'{display_param} (%)',
                title=f'Detection failure: {display_param}',
                dark_plots=dark_plots,
                caution_level=_DETECTION_CAUTION_PCNT,
                caution_label=f'caution: >{_DETECTION_CAUTION_PCNT:.0f}% undetected',
                add_stats=add_stats,
            )
        except ValueError:
            log_params.setdefault('Plotting errors', []).append(
                f'No data to compare for {param}'
            )
    return figs


def generate_group_coverage(
    summary_df: pd.DataFrame,
    log_params: dict,
    dark_plots: bool = False,
    group_order: list = None,
    group_labels: dict = None,
    add_stats: bool = True,
) -> dict:
    '''
    Generate a per-group coverage comparison from the 'Num Bins' column.

    Each point is one image; low values flag under-sampled images whose
    per-image means are statistically thin. Returns a dict mapping a single
    plot name -> matplotlib Figure (empty if the column is absent/empty).
    '''
    print('Generating coverage comparison...')
    figs = {}
    if 'Group Name' not in summary_df.columns or 'Num Bins' not in summary_df.columns:
        return figs

    summary_df, order = _resolve_display_and_order(summary_df, group_order, group_labels)
    work_df = summary_df.copy()
    work_df['Num Bins'] = pd.to_numeric(work_df['Num Bins'], errors='coerce')
    if work_df['Num Bins'].dropna().empty:
        log_params.setdefault('Plotting errors', []).append('No data to compare for Num Bins')
        return figs

    figs['Num Bins'] = _quality_box_swarm(
        work_df,
        'Num Bins',
        order,
        ylabel='Number of bins per image',
        title='Coverage: bins per image',
        dark_plots=dark_plots,
        add_stats=add_stats,
    )
    return figs


def generate_group_effective_n_heatmap(
    summary_df: pd.DataFrame,
    log_params: dict,
    dark_plots: bool = False,
    channel_names: list = None,
    edge_height_fraction: float = None,
    group_order: list = None,
    group_labels: dict = None,
) -> dict:
    '''
    Generate an effective sample-size heatmap (group x metric).

    Each cell is the number of images in a group that carry a usable (non-NaN)
    value for a metric -- the "can I even run stats here?" map. Cells below
    _MIN_N_FOR_STATS are outlined so under-supported group/metric pairs stand
    out. Returns a dict mapping a single plot name -> matplotlib Figure.
    '''
    print('Generating effective sample-size heatmap...')
    figs = {}
    if 'Group Name' not in summary_df.columns:
        return figs

    summary_df, order = _resolve_display_and_order(summary_df, group_order, group_labels)
    metric_cols = [
        c for c in summary_df.columns
        if 'Mean' in c and pd.to_numeric(summary_df[c], errors='coerce').notna().any()
    ]
    if not order or not metric_cols:
        log_params.setdefault('Plotting errors', []).append(
            'Not enough data for effective sample-size heatmap'
        )
        return figs

    counts = np.zeros((len(order), len(metric_cols)), dtype=int)
    for i, group in enumerate(order):
        group_df = summary_df.loc[summary_df['Group Name'] == group]
        for j, col in enumerate(metric_cols):
            counts[i, j] = int(pd.to_numeric(group_df[col], errors='coerce').notna().sum())

    labels = [relabel_metric_text(c, channel_names, edge_height_fraction) for c in metric_cols]
    figs['Effective Sample Size'] = _return_effective_n_figure(
        counts, row_labels=[str(g) for g in order], col_labels=labels, dark_plots=dark_plots,
    )
    return figs


def _return_effective_n_figure(
    counts: np.ndarray,
    row_labels: list,
    col_labels: list,
    dark_plots: bool = False,
) -> plt.Figure:
    n_rows, n_cols = counts.shape
    with style_context(dark_plots):
        width = min(max(0.5 * n_cols + 4, 8), 22)
        height = min(max(0.5 * n_rows + 2, 4), 16)
        fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
        apply_dark(fig, ax, dark_plots)

        vmax = max(counts.max(), 1)
        im = ax.imshow(counts, vmin=0, vmax=vmax, cmap='viridis', aspect='auto')
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels, fontsize=8)

        for i in range(n_rows):
            for j in range(n_cols):
                value = int(counts[i, j])
                # Annotate count; outline cells too thin to run a group test.
                ax.text(j, i, str(value), ha='center', va='center', fontsize=7,
                        color='white' if value < 0.6 * vmax else 'black')
                if value < _MIN_N_FOR_STATS:
                    ax.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, fill=False,
                        edgecolor='red', lw=1.5,
                    ))

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Usable images (non-NaN)')
        ax.set_title(
            f'Effective sample size per group x metric\n'
            f'(red outline: n < {_MIN_N_FOR_STATS}, too few for a group test)',
            fontsize=10,
        )
        plt.close(fig)
    return fig


def generate_group_per_image_reliability(
    summary_df: pd.DataFrame,
    log_params: dict,
    dark_plots: bool = False,
    channel_names: list = None,
    edge_height_fraction: float = 0.5,
    group_order: list = None,
    group_labels: dict = None,
) -> dict:
    '''
    Generate per-image reliability (caterpillar) plots, one per metric.

    Each marker is one image's per-image mean for the metric, with a vertical
    error bar of +/- the within-image StdDev across its bins. Images are blocked
    by group (group dividers + a group label) and sorted by mean within each
    block, so both the central tendency and the within-image spread of every
    individual image are visible at a glance. Long error bars flag images whose
    mean rests on highly variable bins.

    Returns a dict mapping metric column name -> matplotlib Figure.
    '''
    print('Generating per-image reliability plots...')
    figs = {}
    if 'Group Name' not in summary_df.columns or 'File Name' not in summary_df.columns:
        return figs

    summary_df, order = _resolve_display_and_order(summary_df, group_order, group_labels)
    mean_cols = [
        c for c in summary_df.columns
        if 'Mean' in c and pd.to_numeric(summary_df[c], errors='coerce').notna().any()
    ]

    palette = dict(zip(order, sns.color_palette('colorblind', n_colors=len(order))))

    for param in mean_cols:
        sd_param = param.replace('Mean', 'StdDev')
        try:
            cols = ['Group Name', 'File Name', param]
            if sd_param in summary_df.columns:
                cols.append(sd_param)
            plot_df = summary_df[cols].copy()
            plot_df[param] = pd.to_numeric(plot_df[param], errors='coerce')
            if sd_param in plot_df.columns:
                plot_df[sd_param] = pd.to_numeric(plot_df[sd_param], errors='coerce')
            plot_df = plot_df.dropna(subset=[param])
            if plot_df.empty:
                raise ValueError('No data to plot')

            display_param = relabel_metric_text(param, channel_names, edge_height_fraction)
            figs[param] = _return_per_image_reliability_figure(
                plot_df,
                param=param,
                sd_param=sd_param if sd_param in plot_df.columns else None,
                order=order,
                palette=palette,
                ylabel=display_param,
                title=f'Per-image reliability: {display_param}',
                dark_plots=dark_plots,
            )
        except ValueError:
            log_params.setdefault('Plotting errors', []).append(
                f'No data to plot for {param}'
            )
    return figs


def _return_per_image_reliability_figure(
    plot_df: pd.DataFrame,
    param: str,
    sd_param: str,
    order: list,
    palette: dict,
    ylabel: str,
    title: str,
    dark_plots: bool = False,
) -> plt.Figure:
    # Lay images out left-to-right: grouped by group order, sorted by mean within
    # each group so each block reads as a rising "caterpillar".
    blocks = []
    for group in order:
        block = plot_df.loc[plot_df['Group Name'] == group].sort_values(param)
        if not block.empty:
            blocks.append((group, block))
    ordered = pd.concat([b for _, b in blocks]) if blocks else plot_df
    n = len(ordered)

    with style_context(dark_plots):
        width = min(max(0.18 * n + 3, 7), 26)
        fig, ax = plt.subplots(figsize=(width, 5.5))
        apply_dark(fig, ax, dark_plots)

        edge_color = 'white' if dark_plots else 'black'
        x = 0
        group_spans = []
        for group, block in blocks:
            xs = np.arange(x, x + len(block))
            means = block[param].values
            yerr = block[sd_param].values if sd_param else None
            ax.errorbar(
                xs, means, yerr=yerr, fmt='o', markersize=5,
                color=palette.get(group, edge_color),
                ecolor=palette.get(group, edge_color),
                elinewidth=1, capsize=2, markeredgecolor=edge_color,
                markeredgewidth=0.4, alpha=0.9, label=str(group),
            )
            group_spans.append((group, x, x + len(block) - 1))
            x += len(block)
            # divider between group blocks
            if x < n:
                ax.axvline(x - 0.5, color='gray', alpha=0.4, lw=0.8, ls='--')

        # group label centered under each block
        for group, x0, x1 in group_spans:
            ax.text((x0 + x1) / 2, 1.01, str(group), transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=9,
                    color=palette.get(group, edge_color))

        # per-image file-name ticks, but only when there are few enough to read
        if n <= 60:
            ax.set_xticks(range(n))
            ax.set_xticklabels(ordered['File Name'].astype(str).tolist(),
                               rotation=90, fontsize=6)
        else:
            ax.set_xticks([])
            ax.set_xlabel(f'Images (n={n}, sorted by mean within group)')

        ax.set_ylabel(f'{ylabel} (mean ± SD)' if sd_param else ylabel)
        # extra pad so the title clears the centered group labels above each block
        ax.set_title(title, fontsize=10, pad=22)
        ax.grid(True, axis='y', alpha=0.25)
        ax.margins(x=0.01)
        fig.tight_layout()
        plt.close(fig)
    return fig
