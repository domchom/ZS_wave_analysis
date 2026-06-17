import textwrap
import numpy as np
import matplotlib.pyplot as plt
from waveanalysis.signal_processing.correlation_functions import calc_indv_edge_lag_profile, _get_signal
from waveanalysis.housekeeping.housekeeping_functions import (
    get_channel_name,
    get_channel_combo_name,
)

# Explanatory captions added to the bottom of summary plots. Signed pair metrics
# use first-channel minus second-channel order from the figure title.
CCF_SHIFT_NOTE = ("CCF shift is the time offset that best aligns the two channels' oscillations (the lag at the peak of "
                  "their cross-correlation), i.e. the overall phase difference between the whole signals.")
PHASE_SHIFT_NOTE = ("Phase shift is the CCF shift normalized by the channel pair's mean period and reported as percent "
                    "of one cycle.")
EDGE_TIME_NOTE = ("Each wave's own shape (no comparison between channels). Rise duration = selected-height rising edge "
                  "up to the apex. Fall duration = apex down to the selected-height falling edge. Peak width = full "
                  "width at half maximum. A fall longer than the rise means an asymmetric, slow-decaying wave.")
EDGE_LAG_NOTE = ("This plots inter-channel lag at multiple heights along the rising and falling edges; 100% is the "
                 "peak-apex shift. If a curve is flat, the channel offset is the same from edge to peak, consistent "
                 "with a constant phase shift. Example: a rising-edge curve that goes from 10 s at 25% to 2 s at "
                 "100% means the first channel is much more delayed at wave onset, then catches up by the peak. A "
                 "sloped falling-edge curve similarly means the channels decay at different rates.")

def _edge_percent(edge_height_fraction: float) -> str:
    return f'{int(round(float(edge_height_fraction) * 100))}%'

def _add_figure_note(fig: plt.Figure, note: str, dark_plots: bool = False, bottom: float = 0.12) -> None:
    '''
    Reserve space at the bottom of a figure and add a centered explanatory caption.
    Call this instead of fig.tight_layout() so the note is not clipped.
    '''
    color = 'lightgray' if dark_plots else 'dimgray'
    fig.tight_layout(rect=[0, bottom, 1, 1])
    fig.text(0.5, bottom * 0.3, note, ha='center', va='bottom', fontsize=8, color=color, wrap=True)

def _add_panel_notes(fig: plt.Figure, notes: list, dark_plots: bool = False,
                     bottom: float = 0.22, wrap_width: int = 58) -> None:
    '''
    Place a caption beneath each panel of a multi-panel figure so every panel
    carries only the note relevant to what it shows.

    notes: list of (x_center, text), with x_center in figure-fraction coords
    roughly under each panel. Text is hard-wrapped to wrap_width characters so
    side-by-side captions stay within their own half of the figure.
    '''
    color = 'lightgray' if dark_plots else 'dimgray'
    fig.tight_layout(rect=[0, bottom, 1, 1])
    for x_center, text in notes:
        fig.text(x_center, bottom * 0.30, textwrap.fill(text, width=wrap_width),
                 ha='center', va='bottom', fontsize=8, color=color)

def _annotate_lead_direction(
    ax: plt.Axes,
    ch1_name: str,
    ch2_name: str,
    axis: str = 'y',
    dark_plots: bool = False,
) -> None:
    '''
    Mark which channel leads at each end of a signed-shift axis.

    Signed pair metrics are first channel minus second, so a negative value means
    the first channel leads and a positive value means the second channel leads.
    The (optional) custom channel names are used for the labels.
    '''
    neg_label = f'{ch1_name} leads'   # negative end (first channel earlier)
    pos_label = f'{ch2_name} leads'   # positive end (second channel earlier)
    color = 'lightgray' if dark_plots else 'dimgray'
    kw = dict(xycoords='axes fraction', textcoords='offset points',
              fontsize=7, fontstyle='italic', color=color)
    if axis == 'y':
        ax.annotate(f'↑ {pos_label}', xy=(0, 1), xytext=(4, -4), ha='left', va='top', **kw)
        ax.annotate(f'↓ {neg_label}', xy=(0, 0), xytext=(4, 4), ha='left', va='bottom', **kw)
    else:
        ax.annotate(f'{neg_label} ←', xy=(0, 1), xytext=(4, -4), ha='left', va='top', **kw)
        ax.annotate(f'→ {pos_label}', xy=(1, 1), xytext=(-4, -4), ha='right', va='top', **kw)

def plot_mean_acf_workflow(
    img_metrics: dict,
    img_props: dict,
    indv_acfs: np.ndarray,
    dark_plots: bool = False
) -> dict:
    '''
    Plot the mean autocorrelation function (ACF) for each channel.

    Args:
        img_metrics (dict): A dictionary containing image parameters.
        img_props (dict): A dictionary containing image properties.
        indv_acfs (np.ndarray): An array of individual autocorrelation functions.

    Returns:
        dict: A dictionary containing the mean ACF figures for each channel.
    '''
    # Extract image properties from the dictionary
    num_channels = img_props['num_channels']
    num_frames = img_props['num_frames']
    indv_periods = img_metrics['Period']

    # Initialize dictionary to store the mean ACF figures
    mean_acf_figs = {}

    # Loop through each channel and generate the mean ACF figure
    for channel in range(num_channels):
        # Generate and store the figure for the current channel
        mean_acf_figs[f'Ch {channel + 1} Mean ACF'] = _return_mean_acf_figure(
            signal=indv_acfs[channel],
            periods=indv_periods[channel],
            channel=get_channel_name(img_props.get('channel_names'), channel),
            num_frames= num_frames,
            frame_interval=img_props['frame_interval'],
            dark_plots=dark_plots)    

    return mean_acf_figs

def _return_mean_acf_figure(
    signal: np.ndarray, 
    periods: np.ndarray, 
    channel: str,
    num_frames: int,
    frame_interval: float,
    dark_plots: bool = False 
) -> plt.Figure:
    '''
    Space saving function to return mean ACF figures
    '''
    # Plot mean autocorrelation curve with shaded area representing standard deviation
    signal_mean = np.nanmean(signal, axis = 0)
    signal_std = np.nanstd(signal, axis = 0)
    x_axis = np.arange(-num_frames + 1, num_frames) * frame_interval

    style = 'dark_background' if dark_plots else 'default'
    with plt.style.context(style):
        # Create the figure with subplots
        fig, ax = plt.subplot_mosaic(mosaic = '''
                                                AA
                                                BC
                                                ''')
        
        # Plot mean autocorrelation curve with shaded area representing standard deviation
        ax['A'].plot(x_axis, signal_mean, color='blue' if not dark_plots else 'lightblue')
        ax['A'].fill_between(x_axis, 
                                signal_mean - signal_std, 
                                signal_mean + signal_std, 
                                color='blue' if not dark_plots else 'lightblue', 
                                alpha=0.2)
        ax['A'].set_title(f'{channel}: mean autocorrelation curve ± SD')

        # Plot histogram of period values
        periods = periods[~np.isnan(periods)]
        ax['B'].hist(periods, color='gray')
        ax['B'].set_xlabel('Detected period (seconds)')
        ax['B'].set_ylabel('Bin count')
        

        # Plot boxplot of period values
        ax['C'].boxplot(periods)
        ax['C'].set_xlabel('Detected period distribution')
        ax['C'].set_ylabel('Period (seconds)')

        fig.subplots_adjust(hspace=0.25, wspace=0.5)  
        plt.close(fig)

    return fig

def plot_mean_peak_props_workflow(
    img_metrics: dict,
    img_props: dict,
    dark_plots: bool = False
) -> dict:
    '''
    Plot Mean Peak Properties Workflow.

    This function takes in the image parameters dictionary and image properties dictionary
    and returns a dictionary of mean peak property figures for each channel.

    Parameters:
    - img_metrics (dict): A dictionary containing the image parameters for each channel.
    - img_props (dict): A dictionary containing the image properties.

    Returns:
    - mean_peak_figs (dict): A dictionary of mean peak property figures for each channel.
    '''
    # Extract peak properties from the image parameters dictionary
    indv_peak_mins = img_metrics['Peak Min']
    indv_peak_maxs = img_metrics['Peak Max']
    indv_peak_amps = img_metrics['Peak Amp']
    indv_peak_widths = img_metrics['Peak Width']
    indv_peak_offsets = img_metrics['Peak Offset']
    num_channels = img_props['num_channels']

    # Initialize dictionary to store the mean peak property figures
    mean_peak_figs = {}

    # Loop through each channel and generate the mean peak property figure
    for channel in range(num_channels):
        # Generate and store the figure for the current channel
        mean_peak_figs[f'Ch {channel + 1} Peak Props'] = _return_mean_prop_peaks_figure(
            min_array=indv_peak_mins[channel],
            max_array=indv_peak_maxs[channel],
            amp_array=indv_peak_amps[channel],
            width_array=indv_peak_widths[channel],
            offsets_array=indv_peak_offsets[channel],
            channel_name=get_channel_name(img_props.get('channel_names'), channel),
            dark_plots=dark_plots
            )

    return mean_peak_figs

def _return_mean_prop_peaks_figure(
    min_array: np.ndarray, 
    max_array: np.ndarray, 
    amp_array: np.ndarray, 
    width_array: np.ndarray,
    offsets_array: np.ndarray,
    channel_name: str,
    dark_plots: bool = False
) -> plt.Figure:
    """
    Space saving function to return mean peak property figures
    """
    style = 'dark_background' if dark_plots else 'default'
    with plt.style.context(style):
        # Create subplots for histograms and boxplots INSIDE the style context
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)

        # Optionally force all backgrounds to black
        if dark_plots:
            fig.patch.set_facecolor('black')
            for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
                ax.set_facecolor('black')

        # Filter out NaN values from arrays
        min_array = [val for val in min_array if not np.isnan(val)]
        max_array = [val for val in max_array if not np.isnan(val)]
        amp_array = [val for val in amp_array if not np.isnan(val)]
        width_array = [val for val in width_array if not np.isnan(val)]
        offsets_array = [val for val in offsets_array if not np.isnan(val)]

        # Define plot parameters for histograms and boxplots
        plot_params = {
            'amplitude': (amp_array, 'blue' if not dark_plots else 'lightblue'),
            'minimum': (min_array, 'purple' if not dark_plots else 'plum'),
            'maximum': (max_array, 'orange' if not dark_plots else 'lightcoral')
        }

        # Plot histograms for peak properties
        for label, (arr, arr_color) in plot_params.items():
            ax1.hist(arr, color=arr_color, label=label, alpha=0.75)

        # Plot boxplots for peak properties
        boxes = ax2.boxplot(
            [val[0] for val in plot_params.values()],
            patch_artist=True
        )
        ax2.set_xticklabels(plot_params.keys())
        for box, box_color in zip(boxes['boxes'], [val[1] for val in plot_params.values()]):
            box.set_edgecolor(box_color)
            box.set_facecolor('none')  # or same color if you want filled boxes

        # Set labels and legends for histograms and boxplots
        ax1.legend(loc='upper right', fontsize='small', ncol=1)
        ax1.set_xlabel(f'{channel_name}: peak amplitude, minimum, and maximum (AU)')
        ax1.set_ylabel('Peak count')
        ax2.set_xlabel(f'{channel_name}: peak value distributions')
        ax2.set_ylabel('Intensity (AU)')

        # Peak widths
        ax3.hist(width_array, color='dimgray', alpha=0.75)
        ax3.set_xlabel(f'{channel_name}: peak full width at half maximum (seconds)')
        ax3.set_ylabel('Peak count')

        bp = ax4.boxplot(width_array, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('dimgray')
        ax4.set_xlabel(f'{channel_name}: peak width distribution')
        ax4.set_ylabel('Full width at half maximum (seconds)')

        # Peak offsets
        ax5.hist(offsets_array, color='dimgray', alpha=0.75)
        ax5.set_xlabel(f'{channel_name}: peak apex offset from waveform midpoint (seconds)')
        ax5.set_ylabel('Peak count')

        bp1 = ax6.boxplot(offsets_array, vert=True, patch_artist=True)
        bp1['boxes'][0].set_facecolor('dimgray')
        ax6.set_xlabel(f'{channel_name}: peak apex offset distribution')
        ax6.set_ylabel('Apex offset from midpoint (seconds)')

        fig.subplots_adjust(hspace=0.6, wspace=0.6)
        plt.close(fig)

    return fig


def plot_mean_ccf_workflow(
    img_metrics: dict,
    img_props: dict,
    indv_ccfs: np.ndarray,
    dark_plots: bool = False
) -> dict:
    '''
    Plot the mean cross-correlation function (CCF) for each channel combination.

    Args:
        img_metrics (dict): A dictionary containing image parameters.
        img_props (dict): A dictionary containing image properties.
        indv_ccfs (np.ndarray): An array of individual cross-correlation functions.

    Returns:
        dict: A dictionary containing the mean CCF figures for each channel combination.
    '''
    # Extract cross-correlation functions and shifts from the image parameters dictionary
    indv_shifts = img_metrics['Shift']
    channel_combos = img_props['channel_combos']
    num_frames = img_props['num_frames']

    # Initialize dictionary to store the mean CCF figures
    mean_ccf_figs = {}

    # Loop through each channel combination and generate the mean CCF figure
    for combo_number, combo in enumerate(channel_combos):
        # Generate and store the figure for the current channel combination
        mean_ccf_figs[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} Mean CCF'] = _return_mean_ccf_figure(
        signal=indv_ccfs[combo_number],
        shifts=indv_shifts[combo_number],
        channel_combo=get_channel_combo_name(img_props.get('channel_names'), combo),
        ch1_name=get_channel_name(img_props.get('channel_names'), combo[0]),
        ch2_name=get_channel_name(img_props.get('channel_names'), combo[1]),
        num_frames= num_frames,
        frame_interval=img_props['frame_interval'],
        dark_plots=dark_plots)

    return mean_ccf_figs

def _return_mean_ccf_figure(
    signal: np.ndarray,
    shifts: np.ndarray,
    channel_combo: str,
    num_frames: int,
    frame_interval: float,
    ch1_name: str = 'Ch1',
    ch2_name: str = 'Ch2',
    dark_plots: bool = False
) -> plt.Figure:
    '''
    Space saving function to return mean CCF figures
    '''
    # Plot mean cross-correlation curve with shaded area representing standard deviation
    arr_mean = np.nanmean(signal, axis = 0)
    arr_std = np.nanstd(signal, axis = 0)
    x_axis = np.arange(-num_frames + 1, num_frames) * frame_interval

    style = 'dark_background' if dark_plots else 'default'
    with plt.style.context(style):
        # Calculate mean and standard deviation of cross-correlation curves
        fig, ax = plt.subplot_mosaic(mosaic = '''
                                                AA
                                                BC
                                                ''')
        
        # Plot mean cross-correlation curve with shaded area representing standard deviation
        ax['A'].plot(x_axis, arr_mean, color='blue' if not dark_plots else 'lightblue')
        ax['A'].fill_between(x_axis, 
                                arr_mean - arr_std, 
                                arr_mean + arr_std, 
                                color='blue' if not dark_plots else 'lightblue', 
                                alpha=0.2)
        ax['A'].set_title(f'{channel_combo}: mean cross-correlation curve ± SD')

        # Plot histogram of period values
        ax['B'].hist(shifts, color='gray')
        shifts = [val for val in shifts if not np.isnan(val)]
        ax['B'].set_xlabel('CCF shift per bin (seconds)')
        ax['B'].set_ylabel('Bin count')
        _annotate_lead_direction(ax['B'], ch1_name, ch2_name, axis='x', dark_plots=dark_plots)

        # Plot boxplot of period values
        ax['C'].boxplot(shifts)
        ax['C'].set_xlabel('CCF shift distribution')
        ax['C'].set_ylabel('CCF shift (seconds)')
        _annotate_lead_direction(ax['C'], ch1_name, ch2_name, axis='y', dark_plots=dark_plots)

        _add_figure_note(fig, f'{CCF_SHIFT_NOTE}', dark_plots, bottom=0.20)
        plt.close(fig)

    return fig

# Edge offsets share a y-axis; the apex-relative differences are smaller and
# get their own panel so they are not flattened by the larger edge offsets.
_LANDMARK_EDGE_METRICS = ['Peak Shift', 'Rise Shift', 'Fall Shift']
_LANDMARK_DIFF_METRICS = ['Rise-Peak Diff', 'Fall-Peak Diff']
def _landmark_labels(edge_height_fraction: float) -> dict:
    pct = _edge_percent(edge_height_fraction)
    return {
        'Peak Shift': 'Peak-apex shift',
        'Rise Shift': f'{pct} rising-edge shift',
        'Fall Shift': f'{pct} falling-edge shift',
        'Rise-Peak Diff': 'Rise shift - peak shift',
        'Fall-Peak Diff': 'Fall shift - peak shift',
    }

def landmark_metric_note(metric_key: str, edge_height_fraction: float) -> str:
    '''
    Caption for a single landmark-shift metric, so each plot only carries the
    note relevant to the metric it shows. *metric_key* may be a bare metric name
    ('Rise Shift') or a longer column/title containing it ('... Mean Rise Shift').
    Returns '' when no landmark metric matches.
    '''
    pct = _edge_percent(edge_height_fraction)
    # Check the apex-relative differences first: 'Rise-Peak Diff' also contains
    # the substrings 'Peak' and 'Rise'.
    if 'Rise-Peak' in metric_key:
        return (f'Rise-Peak diff = the {pct} rising-edge shift minus the apex shift. It is zero when the two channels '
                f'are the same waveform offset by a fixed delay, so a non-zero value reflects different rising kinetics. '
                f'Positive = the rising edge is more delayed than the peak (the first channel reaches the rising edge '
                f'even later, relative to the second channel, than it reaches its apex); negative = less delayed.')
    if 'Fall-Peak' in metric_key:
        return (f'Fall-Peak diff = the {pct} falling-edge shift minus the apex shift. It is zero when the two channels '
                f'are the same waveform offset by a fixed delay, so a non-zero value reflects different falling kinetics. '
                f'Positive = the falling edge is more delayed than the peak (the first channel reaches the falling edge '
                f'even later, relative to the second channel, than it reaches its apex); negative = less delayed.')
    if 'Peak Shift' in metric_key:
        return ('Peak shift = the time between the two channels reaching their apex (peak).')
    if 'Rise Shift' in metric_key:
        return (f'Rise shift = the time between the two channels crossing {pct} of peak height on the rising edge. ')
    if 'Fall Shift' in metric_key:
        return (f'Fall shift = the time between the two channels crossing {pct} of peak height on the falling edge. ')
    return ''

def plot_mean_landmark_shift_workflow(
    img_metrics: dict,
    img_props: dict,
    dark_plots: bool = False
) -> dict:
    '''
    Plot the distribution of the landmark-based shift metrics for each channel combination.

    For every combo this produces one figure with two boxplot panels: the edge
    offsets (peak apex, rising edge, falling edge) and the apex-relative
    differences (rise - peak, fall - peak). Each box is overlaid with the
    per-bin measurements so the spread across bins is visible.

    Returns an empty dict if the landmark metrics were not computed (e.g. a
    single-channel image).
    '''
    channel_combos = img_props['channel_combos']
    mean_landmark_figs = {}

    if not all(metric in img_metrics for metric in _LANDMARK_EDGE_METRICS):
        return mean_landmark_figs

    edge_height_fraction = img_props.get('edge_height_fraction', 0.5)
    labels = _landmark_labels(edge_height_fraction)
    for combo_number, combo in enumerate(channel_combos):
        combo_label = f'Ch{combo[0] + 1}-Ch{combo[1] + 1}'
        combo_display = get_channel_combo_name(img_props.get('channel_names'), combo)
        edge_data = [img_metrics[m][combo_number] for m in _LANDMARK_EDGE_METRICS]
        diff_data = [img_metrics[m][combo_number] for m in _LANDMARK_DIFF_METRICS
                     if m in img_metrics]
        mean_landmark_figs[f'{combo_label} Landmark Shifts'] = _return_landmark_shift_figure(
            edge_data=edge_data,
            diff_data=diff_data,
            diff_labels=[labels[m] for m in _LANDMARK_DIFF_METRICS if m in img_metrics],
            combo_label=combo_display,
            ch1_name=get_channel_name(img_props.get('channel_names'), combo[0]),
            ch2_name=get_channel_name(img_props.get('channel_names'), combo[1]),
            edge_height_fraction=edge_height_fraction,
            dark_plots=dark_plots,
        )

    return mean_landmark_figs

def _boxplot_with_points(
    ax: plt.Axes,
    data: list,
    labels: list,
    dark_plots: bool = False
) -> None:
    '''
    Space saving helper: boxplot of each group with the raw points jittered on top.
    '''
    clean = [np.asarray(group, dtype=float)[np.isfinite(group)] for group in data]
    ax.boxplot(clean, showfliers=False)
    point_color = 'lightblue' if dark_plots else 'gray'
    for position, values in enumerate(clean, start=1):
        if values.size == 0:
            continue
        jitter = (np.random.rand(values.size) - 0.5) * 0.15
        ax.scatter(np.full(values.size, position) + jitter, values,
                   color=point_color, alpha=0.5, s=12, zorder=3)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20, ha='right')

def _return_landmark_shift_figure(
    edge_data: list,
    diff_data: list,
    diff_labels: list,
    combo_label: str,
    edge_height_fraction: float,
    ch1_name: str = 'Ch1',
    ch2_name: str = 'Ch2',
    dark_plots: bool = False
) -> plt.Figure:
    '''
    Space saving function to return the landmark-shift distribution figure.
    '''
    style = 'dark_background' if dark_plots else 'default'
    with plt.style.context(style):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        if dark_plots:
            fig.patch.set_facecolor('black')
            ax1.set_facecolor('black')
            ax2.set_facecolor('black')

        labels = _landmark_labels(edge_height_fraction)
        _boxplot_with_points(ax1, edge_data, [labels[m] for m in _LANDMARK_EDGE_METRICS], dark_plots)
        ax1.set_ylabel('Time difference (seconds)')
        ax1.set_title(f'{combo_label}: absolute landmark timing')
        _annotate_lead_direction(ax1, ch1_name, ch2_name, axis='y', dark_plots=dark_plots)

        _boxplot_with_points(ax2, diff_data, diff_labels, dark_plots)
        ax2.set_ylabel('Difference from apex shift (seconds)')
        ax2.set_title(f'{combo_label}: edge timing relative to apex')

        pct = _edge_percent(edge_height_fraction)
        # One caption per panel: the left panel shows the raw landmark shifts,
        # the right panel shows each edge shift relative to the apex shift.
        left_note = (f'Absolute timing: peak shift = time between the two channels reaching their apex; rise and fall '
                     f'shifts = time between them crossing {pct} of peak height on the rising and falling edges. '
                     f'Arrows mark which channel leads (negative) versus trails (positive).')
        right_note = (f'Relative to apex: each value is the edge shift minus the apex shift. It is zero when the two '
                      f'channels are the same waveform offset by a fixed delay; positive = that edge is more delayed '
                      f'than the peak, negative = less delayed. This isolates differences in rise/fall kinetics from '
                      f'the overall phase offset.')
        _add_panel_notes(fig, [(0.28, left_note), (0.76, right_note)], dark_plots, bottom=0.24)
        plt.close(fig)

    return fig

# Per-channel peak-shape durations shown together (Rise + Fall sum to the width).
_EDGE_DURATION_METRICS = ['Rise Duration', 'Fall Duration', 'Rise minus Fall Duration', 'Peak Width']
def _edge_time_labels(edge_height_fraction: float) -> dict:
    pct = _edge_percent(edge_height_fraction)
    return {
        'Rise Duration': f'Rise duration ({pct} to apex)',
        'Fall Duration': f'Fall duration (apex to {pct})',
        'Rise minus Fall Duration': 'Rise duration - fall duration',
        'Peak Width': 'Peak width (full half-max)',
    }

def plot_mean_edge_times_workflow(
    img_metrics: dict,
    img_props: dict,
    dark_plots: bool = False
) -> dict:
    '''
    Plot the distribution of each channel's own peak edge timings (no comparison
    between channels): the rising-edge duration, falling-edge duration, and the
    selected-height width they sum to. One figure per channel, each box overlaid with
    the per-bin measurements.

    Returns an empty dict if the edge-time metrics were not computed.
    '''
    num_channels = img_props['num_channels']
    edge_time_figs = {}

    if not all(metric in img_metrics for metric in ('Rise Duration', 'Fall Duration')):
        return edge_time_figs

    edge_height_fraction = img_props.get('edge_height_fraction', 0.5)
    label_map = _edge_time_labels(edge_height_fraction)
    metrics = [m for m in _EDGE_DURATION_METRICS if m in img_metrics]
    for channel in range(num_channels):
        data = [img_metrics[m][channel] for m in metrics]
        edge_time_figs[f'Ch{channel + 1} Edge Times'] = _return_edge_times_figure(
            data=data,
            labels=[label_map[m] for m in metrics],
            channel_label=get_channel_name(img_props.get('channel_names'), channel),
            edge_height_fraction=edge_height_fraction,
            dark_plots=dark_plots,
        )

    return edge_time_figs

def _return_edge_times_figure(
    data: list,
    labels: list,
    channel_label: str,
    edge_height_fraction: float,
    dark_plots: bool = False
) -> plt.Figure:
    '''
    Space saving function to return the per-channel edge-time distribution figure.
    '''
    style = 'dark_background' if dark_plots else 'default'
    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=(6, 5))

        if dark_plots:
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')

        _boxplot_with_points(ax, data, labels, dark_plots)
        ax.set_ylabel('Duration / asymmetry within each peak (seconds)')
        ax.set_title(f'{channel_label}: peak edge durations')

        pct = _edge_percent(edge_height_fraction)
        note = (f'Each wave\'s own shape (no comparison between channels). Rise duration = time from {pct} of peak '
                f'height up to the apex. Fall duration = time from the apex down to {pct} of peak height. Peak width = '
                f'full width at half maximum. Rise - fall is positive for a slower rise than fall, and negative for a '
                f'faster rise than fall.')
        _add_figure_note(fig, note, dark_plots)
        plt.close(fig)

    return fig

def plot_lag_threshold_profile_workflow(
    bin_values: np.ndarray,
    img_metrics: dict,
    img_props: dict,
    dark_plots: bool = False
) -> dict:
    '''
    Plot the inter-channel lag as a function of amplitude fraction along the
    rising and falling edges, for each channel combination.

    Per combo, the lag profile is computed for every bin (via
    calc_indv_edge_lag_profile) and averaged across bins (mean +/- std). A flat
    line means a pure phase shift; a sloped line means the channels' edges move
    at different rates. The apex (fraction 1.0) equals the peak shift.

    Returns an empty dict if there is only one channel or no periods to derive
    the peak-matching tolerance.
    '''
    channel_combos = img_props['channel_combos']
    num_bins = img_props['num_bins']
    analysis_type = img_props['analysis_type']
    frame_interval = img_props['frame_interval']
    peak_prominence_fraction = img_props.get('peak_prominence_fraction', 0.1)
    periods = img_metrics.get('Period')

    lag_profile_figs = {}
    if img_props['num_channels'] < 2 or periods is None:
        return lag_profile_figs

    for combo_number, combo in enumerate(channel_combos):
        rise_per_bin, fall_per_bin = [], []
        fractions = None
        for bin in range(num_bins):
            signal1 = _get_signal(bin_values, combo[0], bin, analysis_type)
            signal2 = _get_signal(bin_values, combo[1], bin, analysis_type)
            # periods are in seconds at plot time; the matcher needs frames.
            period_frames = np.nanmean(periods[[combo[0], combo[1]], bin]) / frame_interval
            fractions, rise_lags, fall_lags = calc_indv_edge_lag_profile(
                signal1=signal1,
                signal2=signal2,
                period=period_frames,
                peak_prominence_fraction=peak_prominence_fraction,
            )
            rise_per_bin.append(rise_lags * frame_interval)
            fall_per_bin.append(fall_lags * frame_interval)

        combo_label = f'Ch{combo[0] + 1}-Ch{combo[1] + 1}'
        combo_display = get_channel_combo_name(img_props.get('channel_names'), combo)
        lag_profile_figs[f'{combo_label} Lag Profile'] = _return_lag_profile_figure(
            fractions=fractions,
            rise_per_bin=np.array(rise_per_bin),
            fall_per_bin=np.array(fall_per_bin),
            combo_label=combo_display,
            ch1_name=get_channel_name(img_props.get('channel_names'), combo[0]),
            ch2_name=get_channel_name(img_props.get('channel_names'), combo[1]),
            dark_plots=dark_plots,
        )

    return lag_profile_figs

def _return_lag_profile_figure(
    fractions: np.ndarray,
    rise_per_bin: np.ndarray,
    fall_per_bin: np.ndarray,
    combo_label: str,
    ch1_name: str = 'Ch1',
    ch2_name: str = 'Ch2',
    dark_plots: bool = False
) -> plt.Figure:
    '''
    Space saving function to return the lag-vs-threshold profile figure.
    '''
    rise_mean = np.nanmean(rise_per_bin, axis=0)
    rise_std = np.nanstd(rise_per_bin, axis=0)
    fall_mean = np.nanmean(fall_per_bin, axis=0)
    fall_std = np.nanstd(fall_per_bin, axis=0)
    pct = fractions * 100

    style = 'dark_background' if dark_plots else 'default'
    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=(8, 5))

        if dark_plots:
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')

        rise_color = 'lightblue' if dark_plots else 'tab:blue'
        fall_color = 'lightcoral' if dark_plots else 'tab:orange'

        ax.plot(pct, rise_mean, marker='^', color=rise_color, label='rising edge lag')
        ax.fill_between(pct, rise_mean - rise_std, rise_mean + rise_std, color=rise_color, alpha=0.2)
        ax.plot(pct, fall_mean, marker='v', color=fall_color, label='falling edge lag')
        ax.fill_between(pct, fall_mean - fall_std, fall_mean + fall_std, color=fall_color, alpha=0.2)

        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        ax.set_xlabel('Edge height (% of peak amplitude; 100% = apex)')
        ax.set_ylabel('Landmark lag between channels (seconds)')
        ax.set_title(f'{combo_label}: lag across rising and falling edges')
        _annotate_lead_direction(ax, ch1_name, ch2_name, axis='y', dark_plots=dark_plots)
        ax.legend(loc='best', fontsize='small')

        _add_figure_note(fig, f'{EDGE_LAG_NOTE}', dark_plots, bottom=0.16)
        plt.close(fig)

    return fig

def return_mean_wave_speeds_figure(
    wave_speeds: list[float],
    dark_plots: bool = False
) -> plt.Figure:
    """
    Returns a matplotlib Figure object that contains a histogram and boxplot of wave speeds.

    Parameters:
        wave_speeds (list[float]): A list of wave speeds in µm/s.

    Returns:
        plt.Figure: A matplotlib Figure object containing the histogram and boxplot.
    """
    style = 'dark_background' if dark_plots else 'default'

    with plt.style.context(style):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

        # Force backgrounds to black in dark mode
        if dark_plots:
            fig.patch.set_facecolor('black')
            ax1.set_facecolor('black')
            ax2.set_facecolor('black')

        # Histogram of wave speeds
        ax1.hist(wave_speeds, bins=10, color='blue' if not dark_plots else 'lightblue', alpha=0.75)
        ax1.set_xlabel('Wave speed (µm/s)')
        ax1.set_ylabel('Occurrences')
        ax1.set_title('Wave speeds histogram')

        # Boxplot of wave speeds
        boxes = ax2.boxplot(wave_speeds, vert=True, patch_artist=True)
        # Optional: color the box to show up on dark background
        for box in boxes['boxes']:
            box.set_facecolor('dimgray')
            box.set_edgecolor('white' if dark_plots else 'black')

        ax2.set_xlabel('Wave speeds')
        ax2.set_ylabel('Wave speed (µm/s)')
        ax2.set_title('Wave speeds boxplot')

        fig.tight_layout()
        plt.close(fig)

    return fig
