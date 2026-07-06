import pandas as pd
import matplotlib.pyplot as plt
from waveanalysis.housekeeping.housekeeping_functions import relabel_channels, metric_unit
from waveanalysis.plotting.style import style_context, apply_dark

# Spell units out on axis labels to match the standard/kymograph summary plots
# (which say "seconds", not "s"). Anything not listed is used as-is.
_UNIT_DISPLAY = {'s': 'seconds', 'AU·s': 'AU·seconds'}


def _rolling_labels(prefix: str, metric: str, channel_names: list = None) -> tuple:
    '''
    Build a (y_label, title) pair for a rolling time-course plot, styled to match
    the standard/kymograph summaries: a short "metric (unit)" y-axis label and a
    descriptive, channel-named title.
    '''
    unit = metric_unit(metric)
    unit = _UNIT_DISPLAY.get(unit, unit)
    y_label = f'{metric} ({unit})' if unit else metric
    title = relabel_channels(f'{prefix}: mean {metric} ± SD over rolling windows', channel_names)
    return y_label, title

def plot_rolling_summary(
    num_channels: int,
    fullmovie_summary: pd.DataFrame,
    channel_combos: list[tuple[int, int]],
    dark_plots: bool = False,
    channel_names: list = None,
    live_injection_dict: dict = {}
):
    '''
    Generate rolling summary plots for wave analysis.

    Parameters:
    - num_channels (int): The number of channels.
    - fullmovie_summary (pd.DataFrame): The summary data for the full movie.
    - channel_combos (list[tuple[int, int]]): The combinations of channels.
    - dark_plots (bool): Inidicates whether the plots should be dark.
    - live_injection_dict (dict): Stores info relevant for plots used when imaging during injections.

    Returns:
    - rolling_mean_plots_dict (dict): A dictionary containing the rolling mean plots.
    '''
    # Initialize the dictionary to store the rolling mean plots
    rolling_mean_plots_dict = {}
    rolling_mean_periods = {}
    rolling_mean_shifts = {}
    rolling_mean_peak_props = {}

    # Generate the rolling mean plots for the mean period
    for channel in range(num_channels):
        y_label, title = _rolling_labels(f'Ch {channel + 1}', 'Period', channel_names)
        rolling_mean_periods[f'Ch{channel + 1} Period'] = _return_mean_periods_shifts_props_plots(
            independent_variable='Submovie',
            dependent_variable=f'Ch {channel + 1} Mean Period',
            dependent_error=f'Ch {channel + 1} StdDev Period',
            y_label=y_label,
            title=title,
            fullmovie_summary=fullmovie_summary,
            dark_plots=dark_plots,
            injection_channels=live_injection_dict.get("injection_channels"),
            injection_submovie=live_injection_dict.get("injection_submovie"),
            channel_names=channel_names,
            )
            
    # Update the dictionary with the rolling mean plots for the mean period
    rolling_mean_plots_dict.update(rolling_mean_periods)

    # Generate the rolling mean plots for the mean shifts
    if num_channels > 1:
        for combo_number, combo in enumerate(channel_combos):
            base = f'Ch{combo[0]+1}-Ch{combo[1]+1}'
            y_label, title = _rolling_labels(base, 'CCF Shift', channel_names)
            rolling_mean_shifts[f'{base} CCF Shift'] = _return_mean_periods_shifts_props_plots(
                independent_variable='Submovie',
                dependent_variable=f'{base} Mean CCF Shift',
                dependent_error=f'{base} StdDev CCF Shift',
                y_label=y_label,
                title=title,
                fullmovie_summary=fullmovie_summary,
                dark_plots=dark_plots,
                injection_channels=live_injection_dict.get("injection_channels"),
                injection_submovie=live_injection_dict.get("injection_submovie"),
                channel_names=channel_names,
                )
            
    # Update the dictionary with the rolling mean plots for the mean shifts
    rolling_mean_plots_dict.update(rolling_mean_shifts)

    # Generate the rolling mean plots for the peak properties
    for channel in range(num_channels):
        # Suffixes match the renamed output columns (Peak Max/Min/Offset are now
        # Peak Apex/Baseline/Apex Offset), so 'Mean Peak {suffix}' still resolves.
        for prop_name in ['Width', 'Apex', 'Baseline', 'Amp', 'Rel Amp', 'Apex Offset', 'Area']:
            y_label, title = _rolling_labels(f'Ch {channel+1}', f'Peak {prop_name}', channel_names)
            rolling_mean_peak_props[f'Ch{channel+1} Peak {prop_name}'] = _return_mean_periods_shifts_props_plots(
                independent_variable='Submovie',
                dependent_variable=f'Ch {channel+1} Mean Peak {prop_name}',
                dependent_error=f'Ch {channel+1} StdDev Peak {prop_name}',
                y_label=y_label,
                title=title,
                fullmovie_summary=fullmovie_summary,
                dark_plots=dark_plots,
                injection_channels=live_injection_dict.get("injection_channels"),
                injection_submovie=live_injection_dict.get("injection_submovie"),
                channel_names=channel_names,
                )
                    
    # Update the dictionary with the rolling mean plots for the peak properties
    rolling_mean_plots_dict.update(rolling_mean_peak_props)

    # Generate the rolling mean plots for the per-channel edge timing and slopes
    rolling_mean_edge_slope = {}
    for channel in range(num_channels):
        for metric in ['Rise Duration', 'Fall Duration', 'Rise minus Fall Duration',
                       'Rising Slope', 'Falling Slope', 'Max Rising Slope',
                       'Max Falling Slope', 'Rising/Falling Slope Ratio']:
            y_label, title = _rolling_labels(f'Ch {channel+1}', metric, channel_names)
            rolling_mean_edge_slope[f'Ch{channel+1} {metric}'] = _return_mean_periods_shifts_props_plots(
                independent_variable='Submovie',
                dependent_variable=f'Ch {channel+1} Mean {metric}',
                dependent_error=f'Ch {channel+1} StdDev {metric}',
                y_label=y_label,
                title=title,
                fullmovie_summary=fullmovie_summary,
                dark_plots=dark_plots,
                injection_channels=live_injection_dict.get("injection_channels"),
                injection_submovie=live_injection_dict.get("injection_submovie"),
                channel_names=channel_names,
                )
    rolling_mean_plots_dict.update(rolling_mean_edge_slope)

    # Generate the rolling mean plots for the per-combo phase shift and
    # landmark (peak-apex / edge) inter-channel shifts
    if num_channels > 1:
        rolling_mean_combo = {}
        for combo in channel_combos:
            base = f'Ch{combo[0]+1}-Ch{combo[1]+1}'
            for metric in ['CCF % Phase Shift', 'Peak-Apex Shift', 'Rising-Edge Shift',
                           'Falling-Edge Shift', 'Rise-Apex Shift Diff', 'Fall-Apex Shift Diff']:
                y_label, title = _rolling_labels(base, metric, channel_names)
                rolling_mean_combo[f'{base} {metric}'] = _return_mean_periods_shifts_props_plots(
                    independent_variable='Submovie',
                    dependent_variable=f'{base} Mean {metric}',
                    dependent_error=f'{base} StdDev {metric}',
                    y_label=y_label,
                    title=title,
                    fullmovie_summary=fullmovie_summary,
                    dark_plots=dark_plots,
                    injection_channels=live_injection_dict.get("injection_channels"),
                    injection_submovie=live_injection_dict.get("injection_submovie"),
                    channel_names=channel_names,
                    )
        rolling_mean_plots_dict.update(rolling_mean_combo)

    return rolling_mean_plots_dict

# Distinct overlay colors for the injection-channel signals, indexed by
# channel number (1-4). Chosen to stand out from the blue metric trace in both
# light and dark themes.
_INJECTION_COLORS = {
    False: {1: 'darkorange', 2: 'green', 3: 'purple', 4: 'saddlebrown'},
    True:  {1: 'yellow', 2: 'lightgreen', 3: 'violet', 4: 'sandybrown'},
}


def _return_mean_periods_shifts_props_plots(
    independent_variable: str,
    dependent_variable: str,
    dependent_error: str,
    y_label: str,
    title: str,
    fullmovie_summary: pd.DataFrame,
    dark_plots: bool = False,
    injection_channels: list = None,
    injection_submovie: float = None,
    channel_names: list = None,
) -> plt.Figure:
    '''
    Space saving function to generate the rolling summary plots.

    injection_channels is a 1-indexed list of channels whose mean signal is
    overlaid (each rescaled to the metric's y-range so its shape can be read
    against the metric). injection_submovie draws a vertical line at the
    injection time, in submovie-index units.
    '''
    with style_context(dark_plots):
        fig, ax = plt.subplots()
        apply_dark(fig, ax, dark_plots)

        # plot the dataframe
        ax.plot(fullmovie_summary[independent_variable],
                fullmovie_summary[dependent_variable],
                color = 'blue' if not dark_plots else 'lightblue')

        # fill between the ± standard deviation of the dependent variable
        ax.fill_between(x = fullmovie_summary[independent_variable],
                        y1 = fullmovie_summary[dependent_variable] - fullmovie_summary[dependent_error],
                        y2 = fullmovie_summary[dependent_variable] + fullmovie_summary[dependent_error],
                        color = 'blue' if not dark_plots else 'lightblue',
                        alpha = 0.25)

        # Overlay the mean signal of each selected injection channel, rescaled to
        # the metric's y-range (shape, not absolute intensity, is what matters).
        labeled = False
        if injection_channels:
            ymin = (fullmovie_summary[dependent_variable] - fullmovie_summary[dependent_error]).min()
            ymax = (fullmovie_summary[dependent_variable] + fullmovie_summary[dependent_error]).max()
            palette = _INJECTION_COLORS[bool(dark_plots)]
            for ch in injection_channels:
                col = f'Ch {ch} Mean Signal'
                if col not in fullmovie_summary.columns:
                    continue
                overlay_signal = fullmovie_summary[col]
                signal_range = overlay_signal.max() - overlay_signal.min()
                if signal_range == 0:
                    continue
                signal_scaled = (overlay_signal - overlay_signal.min()) / signal_range
                signal_scaled = signal_scaled * (ymax - ymin) + ymin
                ax.plot(
                    fullmovie_summary[independent_variable],
                    signal_scaled,
                    color=palette.get(ch, 'gray'),
                    linewidth=2,
                    alpha=0.8,
                    label=relabel_channels(f'Ch {ch} signal', channel_names),
                )
                labeled = True

        # plot vertical line indicating when injection takes place
        if injection_submovie is not None and injection_submovie != 0:
            ax.axvline(x=injection_submovie, color='red', linestyle='--', alpha=0.6,
                       label='injection')
            labeled = True

        if labeled:
            ax.legend(loc='best', fontsize=8)

        # set axis labels
        ax.set_xlabel('Rolling submovie index')
        ax.set_ylabel(y_label)
        ax.set_title(title)
        plt.close(fig)

    return fig
