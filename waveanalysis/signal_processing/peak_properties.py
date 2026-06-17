import warnings
import numpy as np
import scipy.signal as sig
from .correlation_functions import _get_signal

warnings.filterwarnings("ignore") # Ignore warnings

# Default fraction of signal amplitude (max - min) required as peak prominence.
# Overridden by the user-supplied value passed through img_props.
_DEFAULT_PEAK_PROMINENCE_FRACTION = 0.1

# rel_height used when computing left/right peak bases — 0.99 gives the
# full-width near the base of the peak rather than at half-height.
# Used to determine if one peak entirely encompasses another, 
# which would indicate that the encompassed peak is not fully resolved 
# and should be excluded from offset calculations.
_PEAK_BASE_REL_HEIGHT = 0.99

def calc_indv_peak_props_workflow(
    bin_values:np.ndarray,
    img_props:dict
) -> tuple:
    '''
    Calculate individual peak properties for each channel and bin.

    Args:
        bin_values (np.ndarray): The input array of bin values.
        img_props (dict): A dictionary containing image properties.

    Returns:
        tuple: A tuple containing the calculated individual peak properties, including:
            - indv_peak_widths (np.ndarray): Array of mean peak widths for each channel and bin.
            - indv_peak_maxs (np.ndarray): Array of mean peak maximums for each channel and bin.
            - indv_peak_mins (np.ndarray): Array of mean peak minimums for each channel and bin.
            - indv_peak_offsets (np.ndarray): Array of mean peak offsets for each channel and bin.
            - indv_peak_props (dict): A dictionary containing individual peak properties for each channel and bin.
            - indv_peak_areas (np.ndarray): Array of mean area under the curve for each channel and bin.
            - indv_rising_slopes (np.ndarray): Array of mean rate of signal increase from left base to peak for each channel and bin.
            - indv_falling_slopes (np.ndarray): Array of mean rate of signal decrease from peak to right base for each channel and bin.
            - indv_max_rising_slopes (np.ndarray): Array of maximum first derivatives (d(signal)/dt) for each channel and bin.
            - indv_max_falling_slopes (np.ndarray): Array of minimum first derivatives (d(signal)/dt) for each channel and bin.
    '''
    # Extract image properties from the dictionary
    num_channels = img_props['num_channels']
    num_bins = img_props['num_bins']
    analysis_type = img_props['analysis_type']
    peak_prominence_fraction = img_props.get('peak_prominence_fraction', _DEFAULT_PEAK_PROMINENCE_FRACTION)

    # Initialize arrays to store the individual peak properties
    indv_peak_widths = np.zeros(shape=(num_channels, num_bins))
    indv_peak_maxs = np.zeros(shape=(num_channels, num_bins))
    indv_peak_mins = np.zeros(shape=(num_channels, num_bins))
    indv_peak_offsets = np.zeros(shape=(num_channels, num_bins))
    indv_peak_areas = np.zeros(shape=(num_channels, num_bins))
    indv_rising_slopes = np.zeros(shape=(num_channels, num_bins))
    indv_falling_slopes = np.zeros(shape=(num_channels, num_bins))
    indv_max_rising_slopes = np.zeros(shape=(num_channels, num_bins))
    indv_max_falling_slopes = np.zeros(shape=(num_channels, num_bins))
    indv_peak_props = {}

    # Loop through each channel and bin
    for channel in range(num_channels):
        for bin in range(num_bins):
            # Extract the bin values for the current channel and bin
            signal = _get_signal(bin_values, channel, bin, analysis_type)
            peaks, _ = sig.find_peaks(signal, prominence=(np.max(signal)-np.min(signal))*peak_prominence_fraction)

            # Find the first derivatve for this signal function
            signal_derivative = np.gradient(signal)
            signal_derivative = sig.savgol_filter(signal_derivative, window_length = 11, polyorder = 2)    

            # Add a constant (mean signal) to the derivate values to put it on a similar y-axis value for plotting
            avg_sig = np.mean(signal)
            signal_derivative_display = signal_derivative * 2 + avg_sig


            # If peaks detected, calculate properties, otherwise return NaNs
            if len(peaks) > 0:
                # Calculate the peak properties
                widths, heights, leftWidthIndex, rightWidthIndex = sig.peak_widths(signal, peaks, rel_height=0.5)
                proms, leftTrough, rightTrough = sig.peak_prominences(signal, peaks)

                # calculate the left and right bases of the peaks, then midpoints and peak offsets
                _, _, left_bases, right_bases = sig.peak_widths(signal, peaks, rel_height=_PEAK_BASE_REL_HEIGHT)
                midpoints = (leftWidthIndex + rightWidthIndex) / 2
                peak_offsets = peaks - midpoints

                # Helper method that calculates the extrema for the derivative of the signal plot
                # (ddx is used as a shorthand to refer to the derivative plot)
                max_rising_slopes, max_falling_slopes = compute_derivative_props(signal_derivative, peaks, leftTrough, rightTrough)
                
                # For each peak, calculate the differences in signal and time between the left trough, peak, and right trough
                left_signal_difference = signal[peaks] - signal[leftTrough]
                right_signal_difference = signal[peaks] - signal[rightTrough]
                left_trough_to_peak_distances = peaks - leftTrough
                peaks_to_right_trough_distances = peaks - rightTrough

                # Calculate the average rate of signal change by dividing the "rise" (change in signal) by the "run" (change in time)
                rising_slopes = left_signal_difference / left_trough_to_peak_distances
                falling_slopes = right_signal_difference / peaks_to_right_trough_distances

                # Check if one peak entirely encompasses another. If so, the encompassed peak is not fully resolved and should be excluded from offset calculations.
                for i in range(len(peaks)):
                    for j in range(len(peaks)):
                        if i != j:  # Avoid self-comparison
                            if left_bases[j] >= left_bases[i] and right_bases[j] <= right_bases[i]:
                                # Peak j is entirely encompassed by peak i
                                left_bases[i] = np.nan
                                right_bases[i] = np.nan
                                peak_offsets[i] = np.nan
                                midpoints[i] = np.nan

                # Exclude clipped peaks: if the signal is decreasing at the start or
                # increasing at the end, the nearest peak is not fully resolved
                if np.mean(signal[:2]) < np.mean(signal[2:3]):  # decreasing at start
                    leftmost = np.argmin(peaks)
                    left_bases[leftmost] = np.nan
                    right_bases[leftmost] = np.nan
                    peak_offsets[leftmost] = np.nan
                    midpoints[leftmost] = np.nan
                    widths[leftmost] = np.nan
                    heights[leftmost] = np.nan
                    leftWidthIndex[leftmost] = np.nan
                    rightWidthIndex[leftmost] = np.nan
                    proms[leftmost] = np.nan

                if np.mean(signal[-2:]) < np.mean(signal[-3:-2]):  # increasing at end
                    rightmost = np.argmax(peaks)
                    left_bases[rightmost] = np.nan
                    right_bases[rightmost] = np.nan
                    peak_offsets[rightmost] = np.nan
                    midpoints[rightmost] = np.nan
                    widths[rightmost] = np.nan
                    heights[rightmost] = np.nan
                    leftWidthIndex[rightmost] = np.nan
                    rightWidthIndex[rightmost] = np.nan
                    proms[rightmost] = np.nan

                # Calculate the mean of the peak widths, maximums, and minimums
                mean_width = np.nanmean(widths, axis=0 )
                mean_max = np.nanmean(signal[peaks], axis = 0)
                mean_min = np.nanmean(signal[peaks] - proms, axis = 0)

                # Drop NaN values because it will mess up the mean calculation
                valid_indices = ~np.isnan(peak_offsets)
                valid_offsets = peak_offsets[valid_indices]
                valid_indices = ~np.isnan(max_rising_slopes)
                valid_max_rising_slopes = max_rising_slopes[valid_indices]
                valid_indices = ~np.isnan(max_falling_slopes)
                valid_max_falling_slopes = max_falling_slopes[valid_indices]

                # Calculate the mean of valid peak offsets
                mean_offset = np.nanmean(valid_offsets)

                # Calculate the mean of valid peak to base values
                mean_rising_slope = np.nanmean(rising_slopes)
                mean_falling_slope = np.nanmean(falling_slopes)

                # Calculate the mean of valid rate extrema
                mean_max_rising_slope = np.nanmean(valid_max_rising_slopes)
                mean_max_falling_slope = np.nanmean(valid_max_falling_slopes)
                
                # --- Calculate peak areas relative to local baseline (trough) ---
                peak_areas = []
                for i in range(len(peaks)):
                    left = left_bases[i]
                    right = right_bases[i]
                    if np.isnan(left) or np.isnan(right):
                        peak_areas.append(np.nan)
                        continue
                    left = int(np.floor(left))
                    right = int(np.ceil(right))
                    
                    # Determine local baseline (trough) under the peak
                    baseline = np.min(signal[left:right+1])
                    
                    # Subtract baseline from signal to get area relative to it
                    auc_peak = np.trapz(signal[left:right+1] - baseline)
                    peak_areas.append(auc_peak)

                mean_area = np.nanmean(peak_areas)
                
            else:
                # If no peaks detected, return NaNs
                mean_width = np.nan
                mean_max = np.nan
                mean_min = np.nan
                mean_offset = np.nan
                mean_rising_slope = np.nan
                mean_falling_slope = np.nan
                mean_max_rising_slope = np.nan
                mean_max_falling_slope = np.nan
                peaks = np.nan
                proms = np.nan 
                heights = np.nan
                leftWidthIndex = np.nan
                rightWidthIndex = np.nan
                midpoints = np.nan
                peak_offsets = np.nan
                left_bases = np.nan
                right_bases = np.nan
                peak_areas = np.nan
                mean_area = np.nan

            # Store the mean peak properties in the arrays
            indv_peak_widths[channel, bin] = mean_width
            indv_peak_maxs[channel, bin] = mean_max
            indv_peak_mins[channel, bin] = mean_min
            indv_peak_offsets[channel, bin] = mean_offset
            indv_peak_areas[channel, bin] = mean_area
            indv_rising_slopes[channel, bin] = mean_rising_slope
            indv_falling_slopes[channel, bin] = mean_falling_slope
            indv_max_rising_slopes[channel, bin] = mean_max_rising_slope
            indv_max_falling_slopes[channel, bin] = mean_max_falling_slope

            # Store the individual peak properties in the dictionary
            indv_peak_props[f'Ch {channel} Bin {bin}'] = {'signal': signal,
                                                                'peaks': peaks,
                                                                'proms': proms,
                                                                'heights': heights,
                                                                'leftWidthIndex': leftWidthIndex,
                                                                'rightWidthIndex': rightWidthIndex,
                                                                'midpoints': midpoints,
                                                                'peak_offsets': peak_offsets,
                                                                'left_bases': left_bases,
                                                                'right_bases': right_bases,
                                                                'peak_areas': peak_areas,
                                                                'derivative': signal_derivative_display,
                                                                'average_signal': avg_sig
                                                                }
                        
                        # TODO: rename the keys to be more descriptive
    
    return indv_peak_widths, indv_peak_maxs, indv_peak_mins, indv_peak_offsets, indv_peak_props, indv_peak_areas, indv_rising_slopes, indv_falling_slopes, indv_max_rising_slopes, indv_max_falling_slopes

def calc_indv_peak_props_rolling(signal: np.ndarray, peak_prominence_fraction: float = _DEFAULT_PEAK_PROMINENCE_FRACTION) -> tuple:
    '''
    Calculate the individual peak properties of a signal using rolling window.

    Parameters:
        signal (np.ndarray): The input signal.

    Returns:
        tuple: A tuple containing the mean width, mean maximum, mean minimum, and mean offset of the peaks. If no peaks are detected, NaN values are returned.
    '''
    # Find peaks in the (already-smoothed) signal
    peaks, _ = sig.find_peaks(signal, prominence=(np.max(signal)-np.min(signal))*peak_prominence_fraction)

    # If peaks detected, calculate properties, otherwise return NaNs
    if len(peaks) > 0:
        # Calculate the peak properties
        widths, heights, leftWidthIndex, rightWidthIndex = sig.peak_widths(signal, peaks, rel_height=0.5)
        proms, _, _ = sig.peak_prominences(signal, peaks)
        # Calculate the mean of the peak widths, maximums, and minimums
        mean_width = np.mean(widths, axis=0)
        mean_max = np.mean(signal[peaks], axis = 0)
        mean_min = np.mean(signal[peaks]-proms, axis = 0)

        # calculate the left and right bases of the peaks, then midpoints and peak offsets
        _, _, left_bases, right_bases = sig.peak_widths(signal, peaks, rel_height=_PEAK_BASE_REL_HEIGHT)
        midpoints = (leftWidthIndex + rightWidthIndex) / 2
        peak_offsets = peaks - midpoints
        # Check if one peak entirely encompasses another
        for i in range(len(peaks)):
            for j in range(len(peaks)):
                if i != j:  # Avoid self-comparison
                    if left_bases[j] >= left_bases[i] and right_bases[j] <= right_bases[i]:
                        # Peak j is entirely encompassed by peak i
                        left_bases[i] = np.nan
                        right_bases[i] = np.nan
                        peak_offsets[i] = np.nan
                        midpoints[i] = np.nan
        
        # Drop NaN values because it will mess up the mean calculation
        valid_indices = ~np.isnan(peak_offsets)
        valid_offsets = peak_offsets[valid_indices]
        # Calculate the mean of valid peak offsets
        mean_offset = np.nanmean(valid_offsets)
        
        # --- Calculate peak areas relative to local baseline (trough) ---
        peak_areas = []
        for i in range(len(peaks)):
            left = left_bases[i]
            right = right_bases[i]
            if np.isnan(left) or np.isnan(right):
                peak_areas.append(np.nan)
                continue
            left = int(np.floor(left))
            right = int(np.ceil(right))
            
            # Determine local baseline (trough) under the peak
            baseline = np.min(signal[left:right+1])
            
            # Subtract baseline from signal to get area relative to it
            auc_peak = np.trapz(signal[left:right+1] - baseline)
            peak_areas.append(auc_peak)

        mean_area = np.nanmean(peak_areas)
        
    else:
        # If no peaks detected, return NaNs
        mean_width = np.nan
        mean_max = np.nan
        mean_min = np.nan
        mean_offset = np.nan
        mean_area = np.nan

    return mean_width, mean_max, mean_min, mean_offset, mean_area

def compute_derivative_props(signal_derivative, peaks, left_troughs, right_troughs):
    """
    Compute peak derivative extrema for all peaks.

    Returns:
        max_rising_slopes (np.ndarray): Max d(signal)/dt from left trough to peak
        max_falling_slopes (np.ndarray): Min d(signal)/dt from peak to right trough
    """

    max_falling_slopes = np.full(peaks.shape, np.nan)
    max_rising_slopes = np.full(peaks.shape, np.nan)

    len_of_peaks = len(peaks)

    for i in range(1, len_of_peaks - 1):
        left_segment = signal_derivative[int(left_troughs[i]):int(peaks[i])]
        right_segment = signal_derivative[int(peaks[i]):int(right_troughs[i])]
        left_segment = left_segment[~np.isnan(left_segment)]
        right_segment = right_segment[~np.isnan(right_segment)]

        if left_segment.size > 0 and right_segment.size > 0:
            max_rising_slopes[i] = np.max(left_segment)
            max_falling_slopes[i] = np.min(right_segment)

    return max_rising_slopes, max_falling_slopes