from typing import Optional
import numpy as np
import pandas as pd

# Landmark-based shift metrics added on top of the original CCF 'Shift'. Handled
# generically per combo (no "Pcnt No Shifts" column, unlike the CCF shift).
_LANDMARK_SHIFT_METRICS = ('Peak Shift', 'Rise Shift', 'Fall Shift', 'Rise-Peak Diff', 'Fall-Peak Diff')

# Metrics that are reported per channel-combination rather than per channel.
_COMBO_METRICS = ('Shift', '% Phase Shift') + _LANDMARK_SHIFT_METRICS

def summarize_image(
    img_metrics: dict,
    img_props: dict
) -> pd.DataFrame:
    '''
    Summarizes the image parameters and properties of a standard kymograph.

    Args:
        img_metrics (dict): A dictionary containing the image parameters.
        img_props (dict): A dictionary containing the image properties.

    Returns:
        pd.DataFrame: A dataframe summarizing the bin results.

    '''
    # Extract image properties from the dictionary
    num_bins = img_props['num_bins']
    num_channels = img_props['num_channels']
    channel_combos = img_props['channel_combos']

    # column names for the dataframe summarizing the bin results
    col_names = ["Parameter", "Mean", "Median", "StdDev", "SEM"]
    col_names.extend([f'Bin {i}' for i in range(1, num_bins + 1)])

    # combine all the statified measurements into a single list
    im_measurements = []
    stats_by_parameter = {}

    if 'num_submovies' in img_props:
        num_submovies = img_props['num_submovies']

        # insert Mean, Median, StdDev, and SEM into the beginning of each list
        for submovie in range(num_submovies):
            stats_rows = []
            for parameter, parameter_measurements in img_metrics.items():
                parameter_with_stats = _add_stats_for_parameter(parameter_measurements[submovie], parameter, num_channels, channel_combos)
                for channel_combo_stat in parameter_with_stats:
                    stats_rows.append(channel_combo_stat)

            # create a dataframe from the statified measurements
            submovie_meas_df = pd.DataFrame(stats_rows, columns = col_names)
            im_measurements.append(submovie_meas_df)
    else:
        # insert Mean, Median, StdDev, and SEM into the beginning of each list
        for parameter, parameter_measurements in img_metrics.items():
            parameter_with_stats = _add_stats_for_parameter(parameter_measurements, parameter, num_channels, channel_combos)
            stats_by_parameter[parameter] = parameter_with_stats
            for channel_combo_stat in parameter_with_stats:
                im_measurements.append(channel_combo_stat)

        # create a dataframe from the statified measurements
        im_measurements = pd.DataFrame(im_measurements, columns = col_names)

    return im_measurements, stats_by_parameter

def _add_stats_for_parameter(
    measurements: np.ndarray,
    measurement_name: str,
    num_channels: int,
    channel_combos: list = None
) -> list:
    '''
    Calculate statistics for a given measurement parameter.

    Parameters:
        measurements (np.ndarray): Array of measurements.
        measurement_name (str): Name of the measurement parameter.
        num_channels (int): Number of channels.
        channel_combos (list, optional): List of channel combinations. Defaults to None.

    Returns:
        list: List of statistics for the measurement parameter.
    '''
    stats_rows = []

    def calculate_statistics(measurements_subset, channel_label):
        meas_mean = np.nanmean(measurements_subset)
        meas_median = np.nanmedian(measurements_subset)
        meas_std = np.nanstd(measurements_subset)
        n_valid = int(np.sum(np.isfinite(measurements_subset)))
        meas_sem = meas_std / np.sqrt(n_valid) if n_valid > 0 else np.nan
        if isinstance(measurements_subset, np.ndarray):
            measurements_subset = measurements_subset.tolist()
        return [channel_label, meas_mean, meas_median, meas_std, meas_sem] + measurements_subset

    if measurement_name not in ['Wave Speed']:
        for index, item in enumerate(channel_combos if measurement_name in _COMBO_METRICS else range(num_channels)):
            if measurement_name in _COMBO_METRICS:
                measurements_subset = measurements[index]
                channel_label = f'Ch{channel_combos[index][0]+1}-Ch{channel_combos[index][1]+1} {measurement_name}'
            else:
                measurements_subset = measurements[item]
                channel_label = f'Ch {item + 1} {measurement_name}'
            
            stats_rows.append(calculate_statistics(measurements_subset, channel_label))

    else:
        measurements = calculate_statistics(measurements, measurement_name)
        stats_rows.append(measurements)
        
    return stats_rows

def combine_stats_for_image_kymo_standard(
    file_name: str, 
    group_name: str,
    img_props: dict,
    img_metrics: dict,
    stats_by_parameter: dict
) -> dict:
    '''
    Combine the statistics for an image in a kymograph or standard analysis.

    Args:
        file_name (str): The name of the file.
        group_name (str): The name of the group.
        img_props (dict): A dictionary containing image properties.
        img_metrics (dict): A dictionary containing image parameters.
        stats_by_parameter (dict): A dictionary containing per-parameter statistics.

    Returns:
        dict: A dictionary containing the summarized measurements for each image.
    '''
    # Extract image properties from the dictionary
    num_bins = img_props['num_bins']
    num_channels = img_props['num_channels']
    channel_combos = img_props['channel_combos']

    # dictionary to store the summarized measurements for each image
    file_data_summary = {}
    file_data_summary['File Name'] = file_name if file_name else 'None'
    file_data_summary['Group Name'] = group_name if group_name else 'None'
    file_data_summary['Num Bins'] = num_bins

    # column names for the dataframe summarizing the bin results
    stats_location = ['Mean', 'Median', 'StdDev', 'SEM']

    # Add stats for each Shifts
    if num_channels > 1:
        for combo_number, combo in enumerate(channel_combos):
            shift_data = img_metrics['Shift'][combo_number]
            pcnt_no_shift = np.count_nonzero(np.isnan(shift_data)) / shift_data.shape[0] * 100
            file_data_summary[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} Pcnt No Shifts'] = pcnt_no_shift
            for ind, stat in enumerate(stats_location):
                file_data_summary[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} {stat} Shift'] = stats_by_parameter['Shift'][combo_number][ind + 1]
            # Unnecessary for loop to add stats for % Phase Shift after the Shifts
            for ind, stat in enumerate(stats_location):
                file_data_summary[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} {stat} % Phase Shift'] = stats_by_parameter['% Phase Shift'][combo_number][ind + 1]

            # Add stats for the landmark-based shifts (peak apex, rising edge, and their difference)
            for metric in _LANDMARK_SHIFT_METRICS:
                if metric not in stats_by_parameter:
                    continue
                for ind, stat in enumerate(stats_location):
                    file_data_summary[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} {stat} {metric}'] = stats_by_parameter[metric][combo_number][ind + 1]

    # Add stats for each parameter
    for name, measurement in img_metrics.items():
        # Skip the combo-based metrics since they are handled separately above
        if name in _COMBO_METRICS:
            continue
        # We calculate the number of bins without Period and Peak Amp 
        elif name in ['Period', 'Peak Amp']:
            for channel in range(num_channels):
                pcnt_no_parameter = np.count_nonzero(np.isnan(measurement[channel])) / measurement[channel].shape[0] * 100
                param = 'Peaks' if name == 'Peak Amp' else 'Periods'
                file_data_summary[f'Ch {channel + 1} Pcnt No {param}'] = pcnt_no_parameter
                for ind, stat in enumerate(stats_location):
                    file_data_summary[f'Ch {channel + 1} {stat} {name}'] = stats_by_parameter[name][channel][ind + 1]
        # All parameters that are not wave speed
        elif name not in ['Wave Speed']:
            for channel in range(num_channels):        
                for ind, stat in enumerate(stats_location):
                    file_data_summary[f'Ch {channel + 1} {stat} {name}'] = stats_by_parameter[name][channel][ind + 1]
        # Wave Speed is a single value, so it doesn't need to be separated by channel
        elif name in ['Wave Speed']:
            for ind, stat in enumerate(stats_location):
                print(stats_by_parameter[name])
                file_data_summary[f'{stat} {name}'] = stats_by_parameter[name][0][ind + 1]

    return file_data_summary

def combine_stats_rolling(
    bin_values: np.ndarray,
    img_props: dict,
    img_metrics: dict,
    indv_ccfs: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    '''
    Combine statistics for rolling analysis.

    Args:
        bin_values (np.ndarray): Bin values for each bin.
        img_props (dict): A dictionary containing image properties.
        img_metrics (dict): A dictionary containing image parameters.
        indv_ccfs (np.ndarray): An array containing individual cross-correlation functions.

    Returns:
        pd.DataFrame: A DataFrame containing the combined statistics.

    '''
    # Extract image properties from the dictionary
    num_channels = img_props['num_channels']
    num_bins = img_props['num_bins']
    num_submovies = img_props['num_submovies']
    channel_combos = img_props['channel_combos']
    subframe_size = img_props['subframe_size']
    subframe_roll = img_props['subframe_roll']

    # Extract image parameters from the dictionary
    indv_periods = img_metrics['Period']
    indv_peak_widths = img_metrics['Peak Width']

    # Define the statistics to calculate
    stat_name_and_func = {'Mean' : np.nanmean,
                            'Median' : np.nanmedian,
                            'StdDev' : np.nanstd                            
                        }
    
    # Extract image properties from the dictionary
    all_submovie_summary = []
    
    # Loop through each submovie
    for submovie in range(num_submovies):
        # Initialize dictionary to store the summary for the current submovie
        submovie_summary = {'Submovie': submovie + 1}

        # Calculate percentage of no shifts for each channel combination
        if num_channels > 1:
            indv_shifts = img_metrics['Shift']
            indv_phase_shifts = img_metrics['% Phase Shift']
            for combo_number, combo in enumerate(channel_combos):
                pcnt_no_shift = np.count_nonzero(np.isnan(indv_ccfs[submovie, combo_number])) / num_bins * 100
                submovie_summary[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} Pcnt No Shifts'] = pcnt_no_shift
                for stat_name, func in stat_name_and_func.items():
                    submovie_summary[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} {stat_name} Shift'] = func(indv_shifts[submovie, combo_number])
                # Unnecessary for loop to add stats for % Phase Shift after the Shifts
                for stat_name, func in stat_name_and_func.items():
                    submovie_summary[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} {stat_name} % Phase Shift'] = func(indv_phase_shifts[submovie, combo_number])

        # Calculate statistics for each channel
        for channel in range(num_channels):

            # Mean signal intensity
            submovie_signal = bin_values[
                subframe_roll * submovie :
                subframe_roll * submovie + subframe_size,
                channel,
                :
            ]
            mean_signal = np.nanmean(submovie_signal)
            submovie_summary[f'Ch {channel + 1} Mean Signal'] = mean_signal

            # Calculate percentage of no periods for the current channel
            pcnt_no_period = (np.count_nonzero(np.isnan(indv_periods[submovie, channel])) / num_bins) * 100
            submovie_summary[f'Ch {channel + 1} Pcnt No Periods'] = pcnt_no_period
            
            # Calculate percentage of no peaks for the current channel
            pcnt_no_peaks = np.count_nonzero(np.isnan(indv_peak_widths[submovie, channel])) / num_bins * 100
            submovie_summary[f'Ch {channel + 1} Pcnt No Peaks'] = pcnt_no_peaks
            
            # Calculate statistics for other parameters excluding combo-based metrics
            for name, measurements in img_metrics.items():
                if name not in _COMBO_METRICS:
                    for stat_name, func in stat_name_and_func.items():
                        submovie_summary[f'Ch {channel + 1} {stat_name} {name}'] = func(measurements[submovie, channel])

        # Calculate statistics for landmark-based shift metrics (per combo)
        if num_channels > 1:
            for metric in _LANDMARK_SHIFT_METRICS:
                if metric not in img_metrics:
                    continue
                for combo_number, combo in enumerate(channel_combos):
                    for stat_name, func in stat_name_and_func.items():
                        submovie_summary[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} {stat_name} {metric}'] = func(img_metrics[metric][submovie, combo_number])

        all_submovie_summary.append(submovie_summary)
    
    col_names = [key for key in all_submovie_summary[0].keys()]
    full_movie_summary = pd.DataFrame(all_submovie_summary, columns = col_names)
            
    return full_movie_summary
