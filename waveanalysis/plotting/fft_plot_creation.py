import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Optional
from waveanalysis.plotting.style import apply_dark

def plot_fft_workflow(
    bin_values: np.ndarray,
    img_props: dict,
    indv_peak_props: dict,
    num_frames: int,
    indv_periods: np.ndarray = None,
    dark_plots: bool = False
) -> dict:
    """
    Generates FFT power plots for each channel and bin.

    Args:
        bin_values (np.ndarray): Array of bin values.
        img_prop_dict (dict): Dictionary containing image properties.
        indv_peak_props (dict): Dictionary containing individual peak properties.
        num_frames (int): Number of frames.
        indv_periods (np.ndarray): ACF-detected periods in seconds.

    Returns:
        dict: Dictionary containing the generated FFT plots.
    """
    # Extract image properties from the dictionary
    num_channels = img_props['num_channels']
    num_bins = img_props['num_bins']
    analysis_type = img_props['analysis_type']
    frame_interval = img_props['frame_interval']

    # Initialize dictionary to store the FFT plots
    fft_figs = {}

    # Loop through each channel and bin
    its = num_channels * num_bins
    with tqdm(total=its, miniters=its/100) as pbar:
        pbar.set_description('fft plots')
        for channel in range(num_channels):
            for bin in range(num_bins):
                pbar.update(1)

                # Extract the bin values for the current channel and bin
                to_plot = bin_values[:, channel, bin] if analysis_type == 'standard' else bin_values[channel, bin]
                acf_period = indv_periods[channel, bin] if indv_periods is not None else np.nan

                # Generate and store the figure for the current channel and bin
                fft_figs[f'Ch{channel + 1} Bin {bin + 1} FFT Plot'] = return_fft_figure(
                    bin_signal=to_plot,
                    prop_dict=indv_peak_props[f'Ch {channel} Bin {bin}'],
                    Ch_name=f'Ch{channel + 1} Bin {bin + 1}',
                    frame_interval=frame_interval,
                    num_frames=num_frames,
                    acf_period=acf_period,
                    dark_plots=dark_plots
                )

    return fft_figs

def return_fft_figure(
    bin_signal: np.ndarray,
    prop_dict: dict,
    Ch_name: str,
    frame_interval: float,
    num_frames: int,
    acf_period: float = np.nan,
    dark_plots: bool = False
) -> plt.Figure:
    '''
    Helper function to return FFT power figures
    '''
    # Use smoothed signal if available, otherwise raw signal
    signal = prop_dict['smoothed'] if 'smoothed' in prop_dict else bin_signal

    # Remove mean so DC component does not dominate
    signal_centered = signal - np.mean(signal)

    # Fourier transform 
    fft_result = np.fft.fft(signal_centered)
    fft_freqs = np.fft.fftfreq(len(signal_centered), d=frame_interval)

    # Keep only positive frequencies
    pos_mask = fft_freqs > 0
    fft_freqs = fft_freqs[pos_mask]
    fft_power = np.abs(fft_result[pos_mask]) ** 2

    # Convert frequency to period
    periods = 1 / fft_freqs

    # Sort by period so x-axis increases left to right
    sort_idx = np.argsort(periods)
    periods = periods[sort_idx]
    fft_power = fft_power[sort_idx]

    # Find dominant peak among finite power values (a bin whose signal contains
    # NaNs yields NaN power; ignore those so a real peak is still found).
    finite = np.isfinite(fft_power)
    if finite.any():
        peak_idx = np.argmax(np.where(finite, fft_power, -np.inf))
        peak_period = periods[peak_idx]
        peak_power = fft_power[peak_idx]
    else:
        peak_period = np.nan
        peak_power = np.nan

    top_peak_idxs = _top_fft_peak_indices(fft_power, n_peaks=3)
    acf_guided_idx = _acf_guided_fft_peak_index(periods, fft_power, acf_period)

    # Create the figure. This plot uses explicit colors (not the dark_background
    # style), so set the black backgrounds via apply_dark and pick white-on-black
    # vs black-on-white for the line/text.
    fig, ax = plt.subplots(constrained_layout=True)
    apply_dark(fig, ax, dark_plots)

    text_color = 'white' if dark_plots else 'black'
    line_color = text_color
    peak_color = text_color
    acf_color = '#4cc9f0' if dark_plots else '#006d77'
    guided_color = '#ffb703' if dark_plots else '#d97706'

    ax.plot(periods, fft_power, color=line_color)

    # Plot top FFT peaks. Guard on finite values: a bin whose signal
    # contains NaNs produces NaN power, and a NaN/Inf y-limit raises.
    if np.isfinite(peak_period) and np.isfinite(peak_power):
        for rank, idx in enumerate(top_peak_idxs, start=1):
            ax.scatter(periods[idx], fft_power[idx], color=peak_color, zorder=3)
            ax.annotate(
                f'#{rank}: {periods[idx]:.1f}s',
                xy=(periods[idx], fft_power[idx]),
                xytext=(0, 6 + (rank - 1) * 10),
                textcoords='offset points',
                ha='center',
                va='bottom',
                color=text_color,
                fontsize=8,
            )
        # Add headroom above the tallest point so the label sits inside the
        # axes instead of colliding with the title.
        if peak_power > 0:
            ax.set_ylim(top=peak_power * 1.30)

    if np.isfinite(acf_period) and acf_period > 0:
        ax.axvline(acf_period, color=acf_color, linestyle=':', linewidth=1.5, label=f'ACF period {acf_period:.1f}s')
        if acf_guided_idx is not None:
            guided_period = periods[acf_guided_idx]
            guided_power = fft_power[acf_guided_idx]
            ax.scatter(guided_period, guided_power, color=guided_color, marker='D', zorder=4,
                       label=f'FFT near ACF {guided_period:.1f}s')

    ax.set_xlim(0, 300)

    ax.set_xlabel('1/f (s)', color=text_color)
    ax.set_ylabel('Power', color=text_color)
    ax.set_title(f'{Ch_name} Fast Fourier Transform', color=text_color)

    ax.tick_params(axis='x', colors=text_color)
    ax.tick_params(axis='y', colors=text_color)

    for spine in ax.spines.values():
        spine.set_color(text_color)

    if np.isfinite(acf_period) and acf_period > 0:
        ax.legend(loc='upper right', fontsize=8)

    plt.close(fig)

    return fig


def _top_fft_peak_indices(fft_power: np.ndarray, n_peaks: int = 3) -> list:
    finite_idxs = np.where(np.isfinite(fft_power))[0]
    if finite_idxs.size == 0:
        return []
    ranked = finite_idxs[np.argsort(fft_power[finite_idxs])[::-1]]
    return ranked[:n_peaks].tolist()


def _acf_guided_fft_peak_index(
    periods: np.ndarray,
    fft_power: np.ndarray,
    acf_period: float,
    tolerance_fraction: float = 0.25,
) -> Optional[int]:
    if not np.isfinite(acf_period) or acf_period <= 0:
        return None
    finite = np.isfinite(periods) & np.isfinite(fft_power)
    near_acf = finite & (periods >= acf_period * (1 - tolerance_fraction)) & (periods <= acf_period * (1 + tolerance_fraction))
    if not near_acf.any():
        return None
    candidate_idxs = np.where(near_acf)[0]
    return candidate_idxs[np.argmax(fft_power[candidate_idxs])]
