import numpy as np
from scipy import signal as sig

# Fraction of signal amplitude required as prominence to confirm both signals
# are oscillatory before computing CCF. Filters out flat or noisy signals.
# Not sure how much this specific value actually matters.
_CCF_SIGNAL_PEAK_PROMINENCE = 0.25

# Minimum prominence in the normalized CCF curve for a peak to be considered valid.
# This is not the user-configurable threshold for calculating the shift, 
# but rather a threshold to filter out noisy or flat CCF curves that do not have clear peaks.
_CCF_PEAK_PROMINENCE = 0.1

# If a detected shift exceeds this fraction of the period it is assumed to be a
# phase-aliased measurement and is corrected by ±1 period. 
# We chose 0.6 as the threshold to allow for some variability in the period while 
# still correcting for small shifts that are likely due to noise or phase aliasing. 
# This means that if the detected shift is greater than 60% of the average period, 
# it will be corrected by adding or subtracting the average period, depending on 
# the direction of the shift. This helps to improve the accuracy of the shift 
# measurements by reducing the impact of small, spurious shifts that may arise 
# from noise or other factors.
_SMALL_SHIFT_CORRECTION_THRESHOLD = 0.6

def calc_indv_ACF_workflow(
    bin_values: np.ndarray,
    img_props: dict,
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
    img_props: dict,
    ccf_smoothing: dict = None,
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
                signal1 = bin_values[:, combo[0], bin] #sig.savgol_filter(bin_values[:, combo[0], bin], window_length=11, polyorder=3)
                signal2 = bin_values[:, combo[1], bin] #sig.savgol_filter(bin_values[:, combo[1], bin], window_length=11, polyorder=3)
            else:
                signal1 = bin_values[combo[0], bin] #signal1 = sig.savgol_filter(bin_values[combo[0], bin], window_length=11, polyorder=3)
                signal2 = bin_values[combo[1], bin] #signal2 = sig.savgol_filter(bin_values[combo[1], bin], window_length=11, polyorder=3)
            # Calculate and store the individual CCF for the current combination of channels and bin
            ccf = calc_indv_CCF(signal1=signal1, signal2=signal2, num_frames=num_frames, ccf_smoothing=ccf_smoothing)
            indv_ccfs[combo_number, bin] = ccf

    return indv_ccfs

def calc_indv_CCF(
    signal1: np.ndarray,
    signal2: np.ndarray,
    num_frames: int,
    ccf_smoothing: dict = None,
) -> np.ndarray:
    '''
    Space saving function to calculate individual cross-correlation functions (CCFs) for each combination of channels and bins.
    '''
    # Find peaks in the signals
    peaks1, _ = sig.find_peaks(signal1, prominence=(np.max(signal1)-np.min(signal1))*_CCF_SIGNAL_PEAK_PROMINENCE)
    peaks2, _ = sig.find_peaks(signal2, prominence=(np.max(signal2)-np.min(signal2))*_CCF_SIGNAL_PEAK_PROMINENCE)

    # If peaks are found in both signals
    if len(peaks1) > 0 and len(peaks2) > 0:
        # Subtract the mean from the signals
        corr_signal1 = signal1 - signal1.mean()
        corr_signal2 = signal2 - signal2.mean()
        # Calculate cross-correlation curve
        cc_curve = np.correlate(corr_signal1, corr_signal2, mode='full')

        # Normalize the cross-correlation curve
        if ccf_smoothing is not None:
            cc_curve = sig.savgol_filter(cc_curve, window_length=ccf_smoothing["window"], polyorder=ccf_smoothing["poly_order"])
        cc_curve = cc_curve / (num_frames * signal1.std() * signal2.std())
        # Find peaks in the cross-correlation curve
        peaks, _ = sig.find_peaks(cc_curve, prominence=_CCF_PEAK_PROMINENCE)

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
    img_props: dict,
    small_shifts_correction: bool,
    ccf_peak_thresh: float
) -> np.ndarray:
    '''
    Calculate individual shifts for each channel combination and bin.

    Parameters:
        indv_ccfs (np.ndarray): Array of cross-correlation functions for each channel combination and bin.
        indv_periods (np.ndarray): Array of periods for each channel combination and bin.
        img_props (dict): Dictionary containing image properties.
        small_shifts_correction (bool): Flag to indicate if small shifts should be corrected.
        ccf_peak_thresh (float): Threshold for peak detection in the cross-correlation function.

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
            shift = calc_indv_shift(cc_curve=indv_ccfs[combo_number, bin], ccf_peak_thresh=ccf_peak_thresh)
            if small_shifts_correction == True:
                average_period = np.mean(indv_periods[:, bin]) # If the shift is too small, correct it
                shift = correct_small_shifts(delay_frames=shift, average_period=average_period)
            indv_shifts[combo_number, bin] = shift

    return indv_shifts

def calc_indv_shift(cc_curve: np.ndarray,
                    ccf_peak_thresh: float) -> np.ndarray:
    '''
    Space saving function to calculate individual shifts for each channel combination and bin.
    '''
    # Find peaks in the cross-correlation curve
    peaks, _ = sig.find_peaks(cc_curve, prominence=ccf_peak_thresh)
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

def correct_small_shifts(
    delay_frames: float, 
    average_period: float
) -> float:
    '''
    Correct small shifts in the cross-correlation curve.
    '''
    # If the shift is larger than 60% of the average period, correct it by subtracting the average period
    if abs(delay_frames) > abs(average_period * _SMALL_SHIFT_CORRECTION_THRESHOLD):
        if delay_frames < 0:
            delay_frames = delay_frames + average_period
        elif delay_frames > 0:
            delay_frames = delay_frames - average_period

    return delay_frames
