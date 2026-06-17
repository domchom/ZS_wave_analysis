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
        group_order (list): Explicit left-to-right group order. Defaults to order
                            of first appearance in the dataframe.
        add_stats (bool): If True, annotate a non-parametric group test.

    Returns:
        dict: A dictionary mapping parameter name -> matplotlib Figure.
    """
    print('Generating group comparisons...')
    group_mean_parameter_figs = {}

    # get the parameters to compare
    parameters_to_compare = [column for column in summary_df.columns if 'Mean' in column]

    # Determine left-to-right group order: caller-specified, else order of first
    # appearance (more meaningful than seaborn's default alphabetical sort).
    present_groups = list(dict.fromkeys(summary_df['Group Name'].tolist()))
    if group_order is not None:
        order = [g for g in group_order if g in present_groups]
        order += [g for g in present_groups if g not in order]
    else:
        order = present_groups

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

    x_label = relabel_metric_text(x_param, channel_names, edge_height_fraction)
    y_label = relabel_metric_text(y_param, channel_names, edge_height_fraction)

    with style_context(dark_plots):
        fig, ax = plt.subplots(figsize=(7, 5))
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
            palette='colorblind',
            s=70,
            edgecolor='black' if not dark_plots else 'white',
            linewidth=0.5,
            ax=ax,
        )

        ax.set_title(f'Group scatter: {x_label} vs {y_label}', fontsize=10)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25)
        ax.legend(title='Group', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        fig.tight_layout()
        plt.close(fig)

    return fig


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
