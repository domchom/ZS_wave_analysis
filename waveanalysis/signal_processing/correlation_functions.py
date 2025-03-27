import numpy as np
from scipy import signal as sig
import scipy.optimize as opt
import matplotlib.pyplot as plt

def calc_indv_ACF_workflow(
    bin_values: np.ndarray,
    img_props: dict
) -> np.ndarray: 
    '''
    Calculate individual Auto-Correlation Function (ACF) workflow.

    Args:
        bin_values (np.ndarray): The input bin values.
        img_props (dict): A dictionary containing image properties.

    Returns:
        np.ndarray: The calculated individual ACFs.
    '''
    # Extract image properties from the dictionary
    num_channels = img_props['num_channels']
    num_bins = img_props['num_bins']
    num_frames = img_props['num_frames']
    acf_peak_thresh = img_props['peak_thresh']
    analysis_type = img_props['analysis_type']

    # Initialize array to store the individual ACFs
    indv_acfs = np.zeros(shape=(num_channels, num_bins, num_frames * 2 - 1))

    # Loop through each channel and bin
    for channel in range(num_channels):
        for bin in range(num_bins):
            # Extract the bin values for the current channel and bin
            signal = bin_values[:, channel, bin] if analysis_type == 'standard' else bin_values[channel, bin]
            # Calculate and store the individual ACF for the current channel and bin
            acf_curve = calc_indv_ACF(signal=signal, num_frames=num_frames, peak_thresh=acf_peak_thresh)
            indv_acfs[channel, bin] = acf_curve

    return indv_acfs

def calc_indv_ACF(
    signal: np.ndarray,
    num_frames: int,
    peak_thresh: float,
) -> np.ndarray:
    '''
    Space saving function to calculate individual Auto-Correlation Function (ACF).
    '''
    # calc autocorrelation and normalize
    corr_signal = signal - np.mean(signal)
    acf_curve = np.correlate(corr_signal, corr_signal, mode='full')
    acf_curve = acf_curve / (num_frames * np.std(signal) ** 2)

    # Find peaks in the autocorrelation curve. If less than two peaks found, return NaNs
    peaks, _ = sig.find_peaks(acf_curve, prominence=peak_thresh)
    if len(peaks) < 2:
        acf_curve = np.full((num_frames * 2 - 1), np.nan)

    return acf_curve

def calc_indv_period_workflow(
    acf_curve: np.ndarray,
    img_props: dict
) -> np.ndarray: 
    '''
    Calculate individual periods for each channel and bin based on the autocorrelation function curve.

    Parameters:
        acf_curve (np.ndarray): The autocorrelation function curve for each channel and bin.
        img_props (dict): A dictionary containing image properties, including the number of channels, number of bins, and peak threshold for the autocorrelation function.

    Returns:
        np.ndarray: An array of individual periods for each channel and bin.
    '''
    # Extract image properties from the dictionary
    num_channels = img_props['num_channels']
    num_bins = img_props['num_bins']
    acf_peak_thresh = img_props['peak_thresh']

    # Initialize array to store the individual periods
    indv_periods = np.zeros(shape=(num_channels, num_bins))

    # Loop through each channel and bin
    for channel in range(num_channels):
        for bin in range(num_bins):
            # Calculate and store the individual period for the current channel and bin
            period = calc_indv_period(acf_curve=acf_curve[channel, bin], peak_thresh=acf_peak_thresh)
            indv_periods[channel, bin] = period

    return indv_periods

def calc_indv_period(
    acf_curve: np.ndarray,
    peak_thresh: float,
) -> float:
    '''
    Space saving function to calculate individual periods for each channel and bin based on the autocorrelation function curve.
    '''
    # Find peaks in the autocorrelation curve, Calculate absolute differences between peaks and center
    peaks, _ = sig.find_peaks(acf_curve, prominence=peak_thresh)
    peaks_abs = np.abs(peaks - acf_curve.shape[0] // 2)

    # If peaks are identified, pick the closest one to the center as the period
    period = np.min(peaks_abs[np.nonzero(peaks_abs)]) if len(peaks) > 1 else np.nan
        
    return period

def calc_indv_CCF_workflow(
    bin_values: np.ndarray,
    img_props: dict
) -> np.ndarray:
    '''
    Calculate individual cross-correlation functions (CCFs) for each combination of channels and bins.

    Args:
        bin_values (np.ndarray): Array of bin values.
        img_props (dict): Dictionary containing image properties.

    Returns:
        np.ndarray: Array of individual CCFs.
    '''
    # Extract image properties from the dictionary
    num_combos = img_props['num_combos']
    num_bins = img_props['num_bins']
    num_frames = img_props['num_frames']
    channel_combos = img_props['channel_combos']
    analysis_type = img_props['analysis_type']

    # Initialize array to store the individual CCFs
    indv_ccfs = np.zeros(shape=(num_combos, num_bins, num_frames*2-1))
    
    # Loop through each combination of channels and bin
    for combo_number, combo in enumerate(channel_combos):
        for bin in range(num_bins):
            # Extract the bin values for the current channel and bin
            if analysis_type == 'standard':
                signal1 = sig.savgol_filter(bin_values[:, combo[0], bin], window_length=11, polyorder=3)
                signal2 = sig.savgol_filter(bin_values[:, combo[1], bin], window_length=11, polyorder=3)
            else:
                signal1 = sig.savgol_filter(bin_values[combo[0], bin], window_length=11, polyorder=3)
                signal2 = sig.savgol_filter(bin_values[combo[1], bin], window_length=11, polyorder=3)
            # Calculate and store the individual CCF for the current combination of channels and bin
            ccf = calc_indv_CCF(signal1=signal1, signal2=signal2, num_frames=num_frames)
            indv_ccfs[combo_number, bin] = ccf

    return indv_ccfs

def calc_indv_CCF(
    signal1: np.ndarray,
    signal2: np.ndarray,
    num_frames: int,
) -> np.ndarray:
    '''
    Space saving function to calculate individual cross-correlation functions (CCFs) for each combination of channels and bins.
    '''
    # Find peaks in the signals
    peaks1, _ = sig.find_peaks(signal1, prominence=(np.max(signal1)-np.min(signal1))*0.25)
    peaks2, _ = sig.find_peaks(signal2, prominence=(np.max(signal2)-np.min(signal2))*0.25)

    # If peaks are found in both signals
    if len(peaks1) > 0 and len(peaks2) > 0:
        # Subtract the mean from the signals
        corr_signal1 = signal1 - signal1.mean()
        corr_signal2 = signal2 - signal2.mean()
        # Calculate cross-correlation curve
        cc_curve = np.correlate(corr_signal1, corr_signal2, mode='full')

        # Normalize the cross-correlation curve
        cc_curve = sig.savgol_filter(cc_curve, window_length=11, polyorder=3)
        cc_curve = cc_curve / (num_frames * signal1.std() * signal2.std())
        # Find peaks in the cross-correlation curve
        peaks, _ = sig.find_peaks(cc_curve, prominence=0.1)

        # If less than two peaks found, return NaNs
        if len(peaks) < 2:
            cc_curve = np.full((num_frames * 2 - 1), np.nan)

    else:
        # If no peaks found, return NaNs
        cc_curve = np.full((num_frames * 2 - 1), np.nan)

    return cc_curve

def calc_indv_shift_workflow(
    indv_ccfs: np.ndarray,
    indv_periods: np.ndarray,
    img_props: dict
) -> np.ndarray:
    '''
    Calculate individual shifts for each channel combination and bin.

    Parameters:
        indv_ccfs (np.ndarray): Array of cross-correlation functions for each channel combination and bin.
        indv_periods (np.ndarray): Array of periods for each channel combination and bin.
        img_props (dict): Dictionary containing image properties.

    Returns:
        np.ndarray: Array of individual shifts for each channel combination and bin.
    '''
    # Extract image properties from the dictionary
    num_combos = img_props['num_combos']
    num_bins = img_props['num_bins']
    channel_combos = img_props['channel_combos']
    
    # Initialize array to store the individual shifts
    indv_shifts = np.zeros(shape=(num_combos, num_bins))

    # Loop through each combination of channels and bin
    for combo_number, combo in enumerate(channel_combos):
        for bin in range(num_bins):
            # Calculate and store the individual shift for the current combination of channels and bin
            shift = calc_indv_shift(cc_curve=indv_ccfs[combo_number, bin])
            average_period = np.mean(indv_periods[:, bin]) # If the shift is too small, correct it
            shift = small_shifts_correction(delay_frames=shift, average_period=average_period)
            indv_shifts[combo_number, bin] = shift

    return indv_shifts

def calc_indv_shift(cc_curve: np.ndarray) -> np.ndarray:
    '''
    Space saving function to calculate individual shifts for each channel combination and bin.
    '''
    # Find peaks in the cross-correlation curve
    peaks, _ = sig.find_peaks(cc_curve, prominence=0.1)
    peaks_abs = abs(peaks - cc_curve.shape[0] // 2)

    # If multiple peaks found, select the one closest to the center
    if len(peaks) > 1:
        delay = np.argmin(peaks_abs[np.nonzero(peaks_abs)])
        delayIndex = peaks[delay]
        delay_frames = delayIndex - cc_curve.shape[0] // 2
    # Otherwise, return NaNs
    else:
        delay_frames = np.nan
    
    return delay_frames

def small_shifts_correction(
    delay_frames: float, 
    average_period: float
) -> float:
    '''
    Correct small shifts in the cross-correlation curve.
    '''
    # If the shift is larger than 60% of the average period, correct it by subtracting the average period
    if abs(delay_frames) > abs(average_period * .6):
        if delay_frames < 0:
            delay_frames = delay_frames + average_period
        elif delay_frames > 0:
            delay_frames = delay_frames - average_period

    return delay_frames

def calc_indv_coherence_workflow(
    acf_curves: np.ndarray,
    img_props: dict
) -> np.ndarray: 
    '''
    Calculate individual periods for each channel and bin based on the autocorrelation function curve.

    Parameters:
        acf_curves (np.ndarray): The autocorrelation function curves for each channel and bin.
        img_props (dict): A dictionary containing image properties, including the number of channels, 
                          number of bins, and peak threshold for the autocorrelation function.

    Returns:
        np.ndarray: An array of individual periods for each channel and bin.
    '''
    # Extract image properties from the dictionary
    num_channels = img_props['num_channels']
    num_bins = img_props['num_bins']
    acf_peak_thresh = 0.05

    # Initialize array to store the individual periods
    indv_coherences = np.zeros(shape=(num_channels, num_bins))  

    # Loop through each channel and bin
    for channel in range(num_channels):
        for bin_idx in range(num_bins):  # Renamed 'bin' to 'bin_idx' to avoid conflict with built-in function
            # Get the length of the ACF curve
            acf_length = acf_curves[channel, bin_idx].shape[0]
            half_length = (acf_length // 2) # Ensure symmetry

            # Extract the left and right halves of the ACF curve
            acf_right = acf_curves[channel, bin_idx][half_length - 10:]
            acf_left = acf_curves[channel, bin_idx][:half_length + 10][::-1]

            # Calculate coherence values for both halves
            coherence_right = calc_indv_coherence(acf_curve=acf_right, peak_thresh=acf_peak_thresh, plot=False)
            coherence_left = calc_indv_coherence(acf_curve=acf_left, peak_thresh=acf_peak_thresh, plot=False)

            # Average coherence from both sides
            coherence = (coherence_right + coherence_left) / 2
            indv_coherences[channel, bin_idx] = coherence

    return indv_coherences

def calc_indv_coherence(acf_curve: np.ndarray, peak_thresh: float, plot: bool = False) -> float:
    """
    Calculate coherence time (τ) from an autocorrelation function curve and plot the results.

    Parameters:
    - acf_curve (np.ndarray): Autocorrelation function curve.
    - peak_thresh (float): Minimum prominence for peak detection.
    - plot (bool): Whether to display a plot of the ACF curve and fitted decay.

    Returns:
    - tau (float): Estimated coherence time or NaN if an error occurs.
    """
    try:
        # Find peaks in the ACF curve
        peaks, _ = sig.find_peaks(acf_curve, prominence=peak_thresh)
        if len(peaks) < 2:
            print("No peaks found")
            return np.nan

        # Extract peak positions and values
        t_values = np.arange(len(acf_curve), dtype=np.float64)  # Ensure float64 for fitting
        peak_times = t_values[peaks].astype(np.float64)
        peak_values = acf_curve[peaks].astype(np.float64)

        # Define the stretched exponential decay function
        def stretched_exp_decay(t, A, tau, beta):
            return A * np.exp(- (t / tau) ** beta)

        # Fit stretched exponential decay to the peak envelope
        popt, _ = opt.curve_fit(stretched_exp_decay, peak_times, peak_values, p0=[np.max(peak_values), 50, 0.8])
        A_fit, tau_fit, beta_fit = popt

        # Plot the ACF curve, detected peaks, and fitted decay if requested
        if plot:
            plt.figure(figsize=(8, 5))
            plt.plot(t_values, acf_curve, label='Autocorrelation Function', color='blue')
            plt.scatter(peak_times, peak_values, color='red', marker='o', label='Detected Peaks')
            plt.plot(t_values, stretched_exp_decay(t_values, *popt), '--', color='green', 
                     label=f'Stretched Decay (τ={tau_fit:.2f}, β={beta_fit:.2f})')
            plt.xlabel("Time Lag")
            plt.ylabel("Autocorrelation")
            plt.title("Autocorrelation Function and Stretched Exponential Fit")
            plt.legend()
            plt.grid(True)
            plt.show()

        return tau_fit
    
    except Exception as e:
        print(f"Error in fitting: {e}")
        return np.nan