import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from .mean_plot_creation import (
    CCF_SHIFT_NOTE,
    PHASE_SHIFT_NOTE,
    landmark_metric_note,
    _add_figure_note,
    _annotate_lead_direction,
)
from waveanalysis.housekeeping.housekeeping_functions import relabel_metric_text, get_channel_name

# Metrics whose value is a signed inter-channel offset, so a positive/negative
# note is meaningful. Matched as a substring of the column name.
_SIGNED_METRIC_KEYS = ('Shift', 'Diff')


def generate_group_comparison(
    summary_df: pd.DataFrame,
    log_params: dict,
    dark_plots: bool = False,
    channel_names: list = None,
    edge_height_fraction: float = 0.5,
) -> dict:
    """
    Generate group comparison plots for each parameter in the summary dataframe.

    Parameters:
        summary_df (pd.DataFrame): The summary dataframe containing the data for comparison.
                                   Must contain a column 'Group Name' and columns with 'Mean' in their names.
        log_params (dict): A dictionary to log any errors encountered during plotting.
                           Expects a key 'Plotting errors' with a list as value.
        dark_plots (bool): If True, use a dark theme with black background.

    Returns:
        dict: A dictionary mapping parameter name -> matplotlib Figure.
    """
    print('Generating group comparisons...')
    group_mean_parameter_figs = {}

    # get the parameters to compare
    parameters_to_compare = [column for column in summary_df.columns if 'Mean' in column]

    # choose style consistent with your other functions
    style = 'dark_background' if dark_plots else 'default'

    for param in parameters_to_compare:
        try:
            # skip if column is completely NaN or empty after dropna
            if summary_df[param].dropna().empty:
                raise ValueError("No data to compare")

            with plt.style.context(style):
                fig, ax = plt.subplots()

                # Force background to black in dark mode
                if dark_plots:
                    fig.patch.set_facecolor('black')
                    ax.set_facecolor('black')

                # boxplot
                sns.boxplot(
                    x='Group Name',
                    y=param,
                    data=summary_df,
                    showfliers=False,
                    ax=ax
                )

                # swarmplot (jittered points)
                sns.swarmplot(
                    x='Group Name',
                    y=param,
                    data=summary_df,
                    color=".25",
                    ax=ax
                )

                display_param = relabel_metric_text(param, channel_names, edge_height_fraction)
                ax.set_title(f'Group comparison: {display_param}')
                ax.set_xlabel('Group')
                ax.set_ylabel(display_param)

                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

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
