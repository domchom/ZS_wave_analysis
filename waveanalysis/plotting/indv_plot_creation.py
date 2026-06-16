import numpy as np
from tqdm import tqdm
import scipy.signal as sig
import matplotlib.pyplot as plt
from waveanalysis.signal_processing.correlation_functions import normalize_signal, _peak_landmarks
from waveanalysis.housekeeping.housekeeping_functions import get_channel_name, get_channel_combo_name

def plot_indv_peak_workflow(
	raw_bin_values: np.ndarray,
	img_props: dict,
	indv_peak_props: dict,
	num_frames: int,
 	dark_plots: bool = False
) -> dict:
	"""
	Generates individual peak plots for each channel and bin.

	Args:
		bin_values (np.ndarray): Array of bin values.
		img_props (dict): Dictionary containing image properties.
		indv_peak_props (dict): Dictionary containing individual peak properties.
		num_frames (int): Number of frames.

	Returns:
		dict: Dictionary containing the generated individual peak plots.
	"""
	# Extract image properties from the dictionary
	num_channels = img_props['num_channels']
	num_bins = img_props['num_bins']
	analysis_type = img_props['analysis_type']
	frame_interval = img_props['frame_interval']

	# Initialize dictionary to store the individual peak plots
	indv_peak_figs = {}
	
	# Loop through each channel and bin
	its = num_channels*num_bins
	with tqdm(total=its, miniters=its/100) as pbar:
		pbar.set_description('ind peaks')
		for channel in range(num_channels):
			for bin in range(num_bins):
				pbar.update(1)
				# Extract the bin values for the current channel and bin
				to_plot = raw_bin_values[:,channel, bin] if analysis_type == 'standard' else raw_bin_values[channel, bin]
				# Generate and store the figure for the current channel and bin
				indv_peak_figs[f'Ch {channel + 1} Bin {bin + 1} Peak Props'] = _return_indv_peak_prop_figure(
					bin_signal=to_plot,
					prop_dict=indv_peak_props[f'Ch {channel} Bin {bin}'],
					channel_name=f'{get_channel_name(img_props.get("channel_names"), channel)} Bin {bin + 1}',
					frame_interval=frame_interval,
					num_frames=num_frames,
					dark_plots=dark_plots
					)
	
	return indv_peak_figs

def _return_indv_peak_prop_figure(
    bin_signal: np.ndarray, 
    prop_dict: dict, 
    channel_name: str,
    frame_interval: float,
    num_frames: int,
    dark_plots: bool = False
) -> plt.Figure:
    """
    Space saving function to return individual peak property figures
    """
    # Extract peak properties from the dictionary
    signal = prop_dict['signal']
    peaks = prop_dict['peaks']
    proms = prop_dict['proms']
    heights = prop_dict['heights']
    leftWidthIndex = prop_dict['leftWidthIndex']
    rightWidthIndex = prop_dict['rightWidthIndex']
    midpoints = prop_dict['midpoints']
    left_bases = prop_dict['left_bases']
    right_bases = prop_dict['right_bases']

    style = 'dark_background' if dark_plots else 'default'

    with plt.style.context(style):
        # Create the figure and plot raw and smoothed signals
        fig, ax = plt.subplots()

        # Force black backgrounds in dark mode for consistency
        if dark_plots:
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')

        x_axis = np.arange(0, num_frames) * frame_interval
        ax.plot(x_axis, bin_signal, color='gray', label='raw intensity')
        ax.plot(x_axis, signal, color='blue' if not dark_plots else 'lightblue', label='smoothed intensity')
    
        # Plot each peak width and amplitude
        if not np.isnan(peaks).any():
            for i in range(peaks.shape[0]):

                left = left_bases[i]
                right = right_bases[i]
                if not np.isnan(left) and not np.isnan(right):
                    left = int(np.floor(left))
                    right = int(np.ceil(right))

                    # Local baseline (trough) for the peak
                    baseline = np.min(signal[left:right+1])

                    # Shade area under the peak relative to baseline
                    ax.fill_between(
                        x_axis[left:right+1], 
                        baseline, 
                        signal[left:right+1], 
                        color='yellow', alpha=0.3
                    )

                # Plot the peak width
                ax.hlines(
                    heights[i], 
                    leftWidthIndex[i] * frame_interval, 
                    rightWidthIndex[i] * frame_interval, 
                    color='olive' if not dark_plots else 'lightgreen', 
                    linestyle='-'
                )
                # Plot the peak amplitude
                ax.vlines(
                    peaks[i] * frame_interval, 
                    signal[peaks[i]] - proms[i],
                    signal[peaks[i]], 
                    color='purple' if not dark_plots else 'magenta', 
                    linestyle='-'
                )
                # Plot the peak offset
                ax.hlines(
                    heights[i] - 5, 
                    peaks[i] * frame_interval, 
                    midpoints[i] * frame_interval, 
                    color='orange' if not dark_plots else 'lightcoral', 
                    linestyle='-'
                )

            # Legend entries from the first peak
            ax.hlines(
                heights[0], 
                leftWidthIndex[0] * frame_interval, 
                rightWidthIndex[0] * frame_interval, 
                color='olive' if not dark_plots else 'lightgreen', 
                linestyle='-',
                label='peak width (FWHM)'
            )
            ax.vlines(
                peaks[0] * frame_interval, 
                signal[peaks[0]] - proms[0],
                signal[peaks[0]], 
                color='purple' if not dark_plots else 'magenta', 
                linestyle='-',
                label='peak amplitude'
            )
            ax.hlines(
                heights[0] - 5, 
                peaks[0] * frame_interval, 
                midpoints[0] * frame_interval, 
                color='orange' if not dark_plots else 'lightcoral', 
                linestyle='-',
                label='apex offset from midpoint'
            )
        
            ax.legend(loc='upper right', fontsize='small', ncol=1)

        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Mean bin intensity (AU)')
        ax.set_title(f'{channel_name}: detected peak measurements')

        plt.close(fig)

    return fig

def plot_indv_acf_workflow(
    raw_bin_values: np.ndarray,
	bin_values: np.ndarray,
	indv_acfs: np.ndarray,
	img_metrics: dict,
	img_props: dict,
 	dark_plots: bool = False
) -> dict:
	"""
	Generates individual ACF plots for each channel and bin.

	Args:
		bin_values (np.ndarray): Array of bin values.
		indv_acfs (np.ndarray): Array of individual ACF values.
		img_metrics (dict): Dictionary of image parameters.
		img_props (dict): Dictionary of image properties.

	Returns:
		dict: Dictionary containing individual ACF plots.
	"""
	# Extract image properties from the dictionary
	num_channels = img_props['num_channels']
	num_bins = img_props['num_bins']
	num_frames = img_props['num_frames']
	indv_periods = img_metrics['Period']
	analysis_type = img_props['analysis_type']
	frame_interval = img_props['frame_interval']

	# Initialize dictionary to store the individual ACF plots
	indv_acf_plots = {}

	# Loop through each channel and bin
	its = num_channels*num_bins
	with tqdm(total=its, miniters=its/100) as pbar:
		pbar.set_description('ind acfs')
		for channel in range(num_channels):
			for bin in range(num_bins):
				pbar.update(1) 
				# Extract the bin values for the current channel and bin
				if raw_bin_values is not None:
					raw_to_plot = raw_bin_values[:,channel, bin] if analysis_type == 'standard' else raw_bin_values[channel, bin]
				to_plot = bin_values[:,channel, bin] if analysis_type == 'standard' else bin_values[channel, bin]
				# Generate and store the figure for the current channel and bin
				indv_acf_plots[f'Ch {channel + 1} Bin {bin + 1} ACF'] = _return_indv_acf_figure(
					raw_to_plot = raw_to_plot if raw_bin_values is not None else None,
					signal=to_plot,
					acf_curve=indv_acfs[channel, bin],
					channel_name=get_channel_name(img_props.get('channel_names'), channel),
					period=indv_periods[channel, bin],
					num_frames=num_frames,
					frame_interval=frame_interval,
					dark_plots=dark_plots
					)
				
	return indv_acf_plots

def _return_indv_acf_figure(
    raw_to_plot: np.ndarray,
    signal: np.ndarray, 
    acf_curve: np.ndarray, 
    channel_name: str, 
    period: float,
    num_frames: int,
    frame_interval: float,
    dark_plots: bool = False
) -> plt.Figure:
    """
    Space saving function to return individual ACF figures
    """
    style = 'dark_background' if dark_plots else 'default'

    with plt.style.context(style):
        # Create subplots for raw signal and autocorrelation curve
        fig, (ax1, ax2) = plt.subplots(2, 1)

        # Force black backgrounds in dark mode
        if dark_plots:
            fig.patch.set_facecolor('black')
            ax1.set_facecolor('black')
            ax2.set_facecolor('black')

        x_axis = np.arange(0, num_frames) * frame_interval

        # Plot the signal(s)
        if raw_to_plot is not None:
            ax1.plot(x_axis, raw_to_plot, color='gray', label='raw intensity')

        ax1.plot(x_axis, signal, color='blue' if not dark_plots else 'lightblue', label='smoothed intensity')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Mean bin intensity (AU)')
        ax1.set_title(f'{channel_name}: signal and autocorrelation')
        ax1.legend(loc='upper right', fontsize='small', ncol=1)

        # Plot the autocorrelation curve
        lags = np.arange(-num_frames + 1, num_frames) * frame_interval
        ax2.plot(lags, acf_curve, color='lightcoral' if dark_plots else 'blue')
        ax2.set_ylabel('Autocorrelation')

        # Annotate the first peak identified as the period if available
        if not np.isnan(period):
            color = 'red'
            ax2.axvline(x=period, alpha=0.5, c=color, linestyle='--')
            ax2.axvline(x=-period, alpha=0.5, c=color, linestyle='--')
            ax2.set_xlabel(f'Detected period: {abs(round(period, 2))} seconds')
        else:
            ax2.set_xlabel('No period identified')

        fig.subplots_adjust(hspace=0.75)
        plt.close(fig)

    return fig


def plot_indv_ccf_workflow(
	bin_values: np.ndarray,
	indv_ccfs: np.ndarray,
	img_metrics: dict,
	img_props: dict,
	dark_plots: bool = False
) -> dict:
	"""
	Plot individual cross-correlation function (CCF) workflow.

	Parameters:
	- bin_values (np.ndarray): Array of bin values.
	- indv_ccfs (np.ndarray): Array of individual CCFs.
	- img_metrics (dict): Dictionary of image parameters.
	- img_props (dict): Dictionary of image properties.

	Returns:
	- indv_ccf_plots (dict): Dictionary of individual CCF plots.
	"""
	# Extract image properties from the dictionary
	channel_combos = img_props['channel_combos']
	num_bins = img_props['num_bins']
	num_frames = img_props['num_frames']
	indv_shifts = img_metrics['Shift']
	analysis_type = img_props['analysis_type']
	frame_interval = img_props['frame_interval']

	# Initialize dictionary to store the individual CCF plots
	indv_ccf_plots = {}

	# Loop through each channel and bin
	its = len(channel_combos)*num_bins
	with tqdm(total=its, miniters=its/100) as pbar:
		pbar.set_description('ind ccfs')
		for combo_number, combo in enumerate(channel_combos):
			for bin in range(num_bins):
				pbar.update(1)
				# Extract the bin values for the current channel and bin
				if analysis_type == 'standard':
					to_plot1 = bin_values[:, combo[0], bin] 
					to_plot2 = bin_values[:, combo[1], bin] 
				else:
					to_plot1 = bin_values[combo[0], bin]
					to_plot2 = bin_values[combo[1], bin]
				# Generate and store the figure for the current channel and bin
				indv_ccf_plots[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} Bin {bin + 1} CCF'] = _return_indv_ccf_figure(
					ch1 = normalize_signal(to_plot1),
					ch2 = normalize_signal(to_plot2),
					ccf_curve = indv_ccfs[combo_number, bin],
					ch1_name = get_channel_name(img_props.get('channel_names'), combo[0]),
					ch2_name = get_channel_name(img_props.get('channel_names'), combo[1]),
					shift = indv_shifts[combo_number, bin],
					num_frames = num_frames,
					frame_interval = frame_interval,
					dark_plots = dark_plots
     			)
				
	return indv_ccf_plots

def _return_indv_ccf_figure(
    ch1: np.ndarray, 
    ch2: np.ndarray, 
    ccf_curve: np.ndarray, 
    ch1_name: str, 
    ch2_name: str, 
    shift: float,
    num_frames: int,
    frame_interval: float,
    dark_plots: bool = False
) -> plt.Figure:
    """
    Space saving function to return individual CCF figures
    """
    style = 'dark_background' if dark_plots else 'default'

    with plt.style.context(style):
        fig, (ax1, ax2) = plt.subplots(2, 1)

        # Force black backgrounds for dark mode
        if dark_plots:
            fig.patch.set_facecolor('black')
            ax1.set_facecolor('black')
            ax2.set_facecolor('black')

        x_axis = np.arange(0, num_frames) * frame_interval 

        # Plot the raw signals
        ax1.plot(x_axis, ch1, color='blue' if not dark_plots else 'lightblue', label=ch1_name)
        ax1.plot(x_axis, ch2, color='orange' if not dark_plots else 'lightcoral', label=ch2_name)
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Normalized mean bin intensity')
        ax1.set_title(f'{ch1_name} and {ch2_name}: normalized signals')
        ax1.legend(loc='upper right', fontsize='small', ncol=1)

        # Plot the cross-correlation curve
        lags = np.arange(-num_frames + 1, num_frames) * frame_interval
        ax2.plot(lags, ccf_curve, color='lightcoral' if dark_plots else 'blue')
        ax2.set_ylabel('Cross-correlation')

        # Annotate the first peak identified as the shift if available
        if not np.isnan(shift):
            color = 'red'
            ax2.axvline(x=shift, alpha=0.5, c=color, linestyle='--')

            # Interpret shift in seconds (already scaled)
            if shift < -frame_interval:
                ax2.set_xlabel(f'CCF shift: {ch1_name} leads by {abs(round(shift, 2))} seconds')
            elif shift > frame_interval:
                ax2.set_xlabel(f'CCF shift: {ch2_name} leads by {abs(round(shift, 2))} seconds')
            else:
                ax2.set_xlabel('CCF shift: no lag detected')
        else:
            ax2.set_xlabel('No CCF peak identified')

        fig.subplots_adjust(hspace=0.5)
        plt.close(fig)

    return fig


def plot_indv_landmark_shift_workflow(
	bin_values: np.ndarray,
	img_metrics: dict,
	img_props: dict,
	dark_plots: bool = False
) -> dict:
	"""
	Plot the per-bin landmark-shift diagnostic for every channel combination.

	For each combo/bin both channels are normalized and overlaid, with the peak
	apex, selected-height rising-edge and selected-height falling-edge crossings marked on each
	signal. This is the visual companion to the Peak/Rise/Fall shift metrics: it
	shows directly where the two channels diverge at each part of the wave.

	Parameters:
	- bin_values (np.ndarray): Array of bin values.
	- img_metrics (dict): Dictionary of image parameters (uses the landmark shifts).
	- img_props (dict): Dictionary of image properties.

	Returns:
	- plots (dict): Dictionary of individual landmark-shift plots.
	"""
	channel_combos = img_props['channel_combos']
	num_bins = img_props['num_bins']
	num_frames = img_props['num_frames']
	analysis_type = img_props['analysis_type']
	frame_interval = img_props['frame_interval']
	peak_prominence_fraction = img_props.get('peak_prominence_fraction', 0.1)
	edge_height_fraction = img_props.get('edge_height_fraction', 0.5)

	plots = {}
	its = len(channel_combos) * num_bins
	with tqdm(total=its, miniters=its / 100) as pbar:
		pbar.set_description('ind landmark shifts')
		for combo_number, combo in enumerate(channel_combos):
			for bin in range(num_bins):
				pbar.update(1)
				if analysis_type == 'standard':
					to_plot1 = bin_values[:, combo[0], bin]
					to_plot2 = bin_values[:, combo[1], bin]
				else:
					to_plot1 = bin_values[combo[0], bin]
					to_plot2 = bin_values[combo[1], bin]
				shifts = {
					name: img_metrics[name][combo_number, bin]
					for name in ('Peak Shift', 'Rise Shift', 'Fall Shift')
					if name in img_metrics
				}
				plots[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} Bin {bin + 1} Landmark Shift'] = _return_indv_landmark_shift_figure(
					ch1=normalize_signal(to_plot1),
					ch2=normalize_signal(to_plot2),
					ch1_name=get_channel_name(img_props.get('channel_names'), combo[0]),
					ch2_name=get_channel_name(img_props.get('channel_names'), combo[1]),
					combo_name=get_channel_combo_name(img_props.get('channel_names'), combo),
					shifts=shifts,
					num_frames=num_frames,
					frame_interval=frame_interval,
					peak_prominence_fraction=peak_prominence_fraction,
					edge_height_fraction=edge_height_fraction,
					dark_plots=dark_plots,
				)

	return plots

def _mark_landmarks(
    ax: plt.Axes,
    signal: np.ndarray,
    frame_interval: float,
    peak_prominence_fraction: float,
    edge_height_fraction: float,
    color: str
) -> None:
    '''
    Space saving helper: mark each peak's apex, selected-height rise, and selected-height fall on the axes.
    '''
    apexes, rise_x, fall_x = _peak_landmarks(signal, peak_prominence_fraction, edge_height_fraction)
    frames = np.arange(len(signal))
    marker_kw = dict(color=color, zorder=5, edgecolor='black', linewidths=0.5)
    if len(apexes):
        ax.scatter(apexes * frame_interval, signal[apexes.astype(int)], marker='o', **marker_kw)
    if len(rise_x):
        ax.scatter(rise_x * frame_interval, np.interp(rise_x, frames, signal), marker='^', **marker_kw)
    if len(fall_x):
        ax.scatter(fall_x * frame_interval, np.interp(fall_x, frames, signal), marker='v', **marker_kw)

def _return_indv_landmark_shift_figure(
    ch1: np.ndarray,
    ch2: np.ndarray,
    ch1_name: str,
    ch2_name: str,
    combo_name: str,
    shifts: dict,
    num_frames: int,
    frame_interval: float,
    peak_prominence_fraction: float,
    edge_height_fraction: float,
    dark_plots: bool = False
) -> plt.Figure:
    '''
    Space saving function to return individual landmark-shift figures.
    '''
    style = 'dark_background' if dark_plots else 'default'
    x_axis = np.arange(0, num_frames) * frame_interval

    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=(10, 4))

        if dark_plots:
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')

        c1 = 'lightblue' if dark_plots else 'blue'
        c2 = 'lightcoral' if dark_plots else 'orange'
        ax.plot(x_axis, ch1, color=c1, label=ch1_name)
        ax.plot(x_axis, ch2, color=c2, label=ch2_name)

        _mark_landmarks(ax, ch1, frame_interval, peak_prominence_fraction, edge_height_fraction, c1)
        _mark_landmarks(ax, ch2, frame_interval, peak_prominence_fraction, edge_height_fraction, c2)

        # Legend entries explaining the marker shapes (drawn off-canvas)
        edge_pct = int(round(edge_height_fraction * 100))
        ax.scatter([], [], marker='o', color='gray', edgecolor='black', linewidths=0.5, label='peak apex')
        ax.scatter([], [], marker='^', color='gray', edgecolor='black', linewidths=0.5, label=f'{edge_pct}% rising edge')
        ax.scatter([], [], marker='v', color='gray', edgecolor='black', linewidths=0.5, label=f'{edge_pct}% falling edge')

        labels = {
            'Peak Shift': 'apex',
            'Rise Shift': f'rise {edge_pct}%',
            'Fall Shift': f'fall {edge_pct}%',
        }
        title = ' | '.join(f'{labels.get(name, name)} shift {value:.2f}s' for name, value in shifts.items())
        title = f'{combo_name}: landmark timing ({title})' if title else f'{combo_name}: landmark timing'
        ax.set_title(title)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Normalized mean bin intensity')
        ax.legend(loc='upper right', fontsize='small', ncol=2)

        fig.tight_layout()
        plt.close(fig)

    return fig
