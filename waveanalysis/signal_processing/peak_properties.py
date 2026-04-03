import warnings
import numpy as np
import scipy.signal as sig
from .correlation_functions import _get_signal

warnings.filterwarnings("ignore") # Ignore warnings

# Fraction of signal amplitude (max - min) required as peak prominence for detection
# for calculating peak properties.
# TODO: make this user-configurable in the future, but for now we want to be able to
# detect smaller peaks that may be present in the data, so we set a relatively low threshold.
_PEAK_PROMINENCE_FRACTION = 0.1

# rel_height used when computing left/right peak bases — 0.99 gives the
# full-width near the base of the peak rather than at half-height.
# Used to determine if one peak entirely encompasses another, 
# which would indicate that the encompassed peak is not fully resolved 
# and should be excluded from offset calculations.
_PEAK_BASE_REL_HEIGHT = 0.99

# Savitzky-Golay pre-smoothing applied inside rolling peak detection.
# This is separate from the user-configurable per-channel smoothing.
_ROLLING_SMOOTH_WINDOW = 11
_ROLLING_SMOOTH_POLY = 2

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
    '''
    # Extract image properties from the dictionary
    num_channels = img_props['num_channels']
    num_bins = img_props['num_bins']
    analysis_type = img_props['analysis_type']

    # Initialize arrays to store the individual peak properties
    indv_peak_widths = np.zeros(shape=(num_channels, num_bins))
    indv_peak_maxs = np.zeros(shape=(num_channels, num_bins))
    indv_peak_mins = np.zeros(shape=(num_channels, num_bins))
    indv_peak_offsets = np.zeros(shape=(num_channels, num_bins))
    indv_peak_areas = np.zeros(shape=(num_channels, num_bins))
    indv_peak_props = {}

    # Loop through each channel and bin
    for channel in range(num_channels):
        for bin in range(num_bins):
            # Extract the bin values for the current channel and bin
            signal = _get_signal(bin_values, channel, bin, analysis_type)
            peaks, _ = sig.find_peaks(signal, prominence=(np.max(signal)-np.min(signal))*_PEAK_PROMINENCE_FRACTION)

            # If peaks detected, calculate properties, otherwise return NaNs
            if len(peaks) > 0:
                # Calculate the peak properties
                widths, heights, leftWidthIndex, rightWidthIndex = sig.peak_widths(signal, peaks, rel_height=0.5)
                proms, _, _ = sig.peak_prominences(signal, peaks)

                # calculate the left and right bases of the peaks, then midpoints and peak offsets
                _, _, left_bases, right_bases = sig.peak_widths(signal, peaks, rel_height=_PEAK_BASE_REL_HEIGHT)
                midpoints = (leftWidthIndex + rightWidthIndex) / 2
                peak_offsets = peaks - midpoints

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

            # Store the individual peak properties in the dictionary
            indv_peak_props[f'Ch {channel} Bin {bin}'] = {'signal': signal, 
                                                                'peaks': peaks,
                                                                'proms': proms, 
                                                                'heights': heights, 
                                                                'leftWidthIndex': leftWidthIndex, 
                                                                'rightWidthIndex': rightWidthIndex,
                                                                'midpoints': midpoints,
                                                                'peak_offsets': peak_offsets,
                                                                'left_base': left_bases,
                                                                'right_base': right_bases,
                                                                'peak_areas': peak_areas
                                                                }
                        
                        # TODO: rename the keys to be more descriptive
    
    return indv_peak_widths, indv_peak_maxs, indv_peak_mins, indv_peak_offsets, indv_peak_props, indv_peak_areas

def calc_indv_peak_props_rolling(signal: np.ndarray) -> tuple:
    '''
    Calculate the individual peak properties of a signal using rolling window.

    Parameters:
        signal (np.ndarray): The input signal.

    Returns:
        tuple: A tuple containing the mean width, mean maximum, mean minimum, and mean offset of the peaks. If no peaks are detected, NaN values are returned.
    '''
    # Calculate the peak properties
    signal = sig.savgol_filter(signal, window_length=_ROLLING_SMOOTH_WINDOW, polyorder=_ROLLING_SMOOTH_POLY)
    peaks, _ = sig.find_peaks(signal, prominence=(np.max(signal)-np.min(signal))*_PEAK_PROMINENCE_FRACTION)

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
