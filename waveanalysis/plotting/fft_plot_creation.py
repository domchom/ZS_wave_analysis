import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def plot_fft_workflow(
    bin_values: np.ndarray,
    img_props: dict,
    indv_peak_props: dict,
    num_frames: int,
    dark_plots: bool = False
) -> dict:
    """
    Generates FFT power plots for each channel and bin.

    Args:
        bin_values (np.ndarray): Array of bin values.
        img_prop_dict (dict): Dictionary containing image properties.
        indv_peak_props (dict): Dictionary containing individual peak properties.
        num_frames (int): Number of frames.

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

                # Generate and store the figure for the current channel and bin
                fft_figs[f'Ch{channel + 1} Bin {bin + 1} FFT Plot'] = return_fft_figure(
                    bin_signal=to_plot,
                    prop_dict=indv_peak_props[f'Ch {channel} Bin {bin}'],
                    Ch_name=f'Ch{channel + 1} Bin {bin + 1}',
                    frame_interval=frame_interval,
                    num_frames=num_frames,
                    dark_plots=dark_plots
                )

    return fft_figs

def return_fft_figure(
    bin_signal: np.ndarray,
    prop_dict: dict,
    Ch_name: str,
    frame_interval: float,
    num_frames: int,
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

    # Find dominant peak
    if len(fft_power) > 0:
        peak_idx = np.argmax(fft_power)
        peak_period = periods[peak_idx]
        peak_power = fft_power[peak_idx]
    else:
        peak_period = np.nan
        peak_power = np.nan

    # Create the figure
    fig, ax = plt.subplots()

    if dark_plots:
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        text_color = 'white'
        line_color = 'white'
        peak_color = 'white'
    else:
        text_color = 'black'
        line_color = 'black'
        peak_color = 'black'

    ax.plot(periods, fft_power, color=line_color)

    # Plot dominant peak
    if not np.isnan(peak_period):
        ax.scatter(peak_period, peak_power, color=peak_color, zorder=3)
        ax.text(
            peak_period,
            peak_power * 1.08,
            f'{peak_period:.1f} s',
            ha='center',
            va='bottom'
        )

    ax.set_xlim(0, 300)    

    ax.set_xlabel('1/f (s)', color=text_color)
    ax.set_ylabel('Power', color=text_color)
    ax.set_title(f'{Ch_name} Fast Fourier Transform', color=text_color)

    ax.tick_params(axis='x', colors=text_color)
    ax.tick_params(axis='y', colors=text_color)

    for spine in ax.spines.values():
        spine.set_color(text_color)

    plt.close(fig)

    return fig
