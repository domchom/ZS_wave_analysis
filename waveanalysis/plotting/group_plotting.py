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
