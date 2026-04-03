import os
import timeit
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Any
import scipy.signal as sig
import waveanalysis.plotting as pt
import waveanalysis.signal_processing as sp
import waveanalysis.housekeeping.housekeeping_functions as hf

from waveanalysis.data_workflows._helpers import _setup_workflow, _load_image_props, _smooth_bin_values_inplace
from waveanalysis.image_props.image_bin_calc import create_multi_frame_bin_array
from waveanalysis.image_props.image_to_np_arrays import tiff_to_np_array_multi_frame
from waveanalysis.summarize_save.summarize_images import summarize_image, combine_stats_rolling

def rolling_workflow(
    folder_path: str,
    log_params: dict[str, Any],
    box_size: int,
    box_shift: int,
    roll_size: int,
    roll_by: int,
    acf_peak_thresh: float,
    ccf_peak_thresh: float,
    small_shifts_correction: bool,
    test: bool = False, # for testing purposes
    smoothing_params: dict = None,
    smoothing: bool = True,
    dark_plots: bool = False
) -> pd.DataFrame:      
    '''
    This is the workflow for rolling analysis. It processes the image files in the specified folder 
    and saves the summary data and figures to a new folder in the same directory as the image files.

    It functions generally in this order (with some analysis specific steps):
        1. Convert a folder of tiff images to numpy arrays
        2. Iterate over every images in the folder
            a. Get the image properties
            b. Calculate the bin values based on the user provided box/line size and bin shift
            c. Calculate the ACF, period, peak properties, and CCFs/shifts (if specified)
            d. Save the summary data and figures to a new folder in the same directory as the image files
        3. Log the parameters and errors
        
    Parameters:
    - folder_path (str): The path to the folder containing the image files.
    - log_params (dict[str, Any]): The dictionary to store the log parameters.
    - acf_peak_thresh (float): The threshold for detecting peaks in the ACF curve.
    - box_size (int, optional): The size of the box for standard analysis. Defaults to None.
    - bin_shift (int, optional): The shift value for binning. Defaults to None.
    - roll_size (int, optional): The size of the submovies.
    - roll_by (int, optional): The amount to roll the submovies by.

    Returns:
    - pd.DataFrame: The summary data for each file.
    '''       
    file_names, start, now, main_save_path = _setup_workflow(folder_path, test)

    print('Processing files...')

    with tqdm(total = len(file_names)) as pbar:
        pbar.set_description('Files processed:')
        for file_name in file_names: 
            try:
                print('******'*10)
                print(f'Processing {file_name}...')

                ############################################
                ####### Image Convert and Properties #######
                ############################################

                image_path = f'{folder_path}/{file_name}'
                assert isinstance(roll_size, int) and isinstance(roll_by, int), 'Roll size and roll by must be integers'

                img_props = _load_image_props(
                    image_path, log_params, file_name,
                    bin_shift=box_shift, box_size=box_size, acf_peak_thresh=acf_peak_thresh,
                )
                if img_props is None:
                    log_params['Files Not Processed'].append(f'{file_name} has less than 11 frames')
                    continue

                num_frames = img_props['num_frames']
                num_channels = img_props['num_channels']
                num_submovies = (num_frames - roll_size) // roll_by
                img_props['num_submovies'] = num_submovies

                image_array = tiff_to_np_array_multi_frame(image_path)
                bin_values, num_bins, num_x_bins, num_y_bins = create_multi_frame_bin_array(image=image_array, img_props=img_props)
                raw_bin_values = bin_values.copy() if smoothing else None
                _smooth_bin_values_inplace(bin_values, num_bins, num_channels, smoothing_params)

                img_props['num_bins'] = num_bins
                img_props['num_x_bins'] = num_x_bins
                img_props['num_y_bins'] = num_y_bins

                file_stem = file_name.rsplit(".", 1)[0]

                ############################################
                ############## Signal Processing ###########
                ############################################

                # Calculate the individual periods for each channel
                indv_periods = np.zeros(shape=(num_submovies, num_channels, num_bins))
                its = num_submovies*num_channels*num_x_bins*num_y_bins
                with tqdm(total = its, miniters=its/100) as pbar:
                    pbar.set_description( 'Periods: ')
                    for submovie in range(num_submovies):
                        for channel in range(num_channels):
                            for bin in range(num_bins):
                                pbar.update(1)
                                signal = bin_values[roll_by * submovie: roll_size + roll_by * submovie, channel, bin]
                                acf_curve = sp.calc_indv_ACF(signal=signal, num_frames=roll_size, peak_thresh=acf_peak_thresh)
                                period = sp.calc_indv_period(acf_curve=acf_curve, peak_thresh=acf_peak_thresh)

                                indv_periods[submovie, channel, bin] = period
                    
                # Calculate the individual peak properties for each channel
                indv_peak_widths = np.zeros(shape=(num_submovies, num_channels, num_bins))
                indv_peak_maxs = np.zeros(shape=(num_submovies, num_channels, num_bins))
                indv_peak_mins = np.zeros(shape=(num_submovies, num_channels, num_bins))
                indv_peak_offsets = np.zeros(shape=(num_submovies, num_channels, num_bins))
                indv_peak_areas = np.zeros(shape=(num_submovies, num_channels, num_bins))

                its = num_submovies*num_channels*num_x_bins*num_y_bins
                with tqdm(total = its, miniters=its/100) as pbar:
                    pbar.set_description('Peak Props: ')
                    for submovie in range(num_submovies):
                        for channel in range(num_channels):
                            for bin in range(num_bins):
                                pbar.update(1)
                                signal = sig.savgol_filter(bin_values[roll_by*submovie : roll_size + roll_by*submovie, channel, bin], window_length=11, polyorder=2)

                                mean_width, mean_max, mean_min, mean_offset, mean_area = sp.calc_indv_peak_props_rolling(signal=signal)

                                # Store peak measurements for each bin in each channel
                                indv_peak_widths[submovie, channel, bin] = mean_width
                                indv_peak_maxs[submovie, channel, bin] = mean_max
                                indv_peak_mins[submovie, channel, bin] = mean_min
                                indv_peak_offsets[submovie, channel, bin] = mean_offset
                                indv_peak_amps = indv_peak_maxs - indv_peak_mins
                                indv_peak_rel_amps = indv_peak_amps / indv_peak_mins
                                indv_peak_areas[submovie, channel, bin] = mean_area

                channel_combos = hf.get_channel_combos(num_channels=num_channels)
                num_combos = len(channel_combos)
                img_props['channel_combos'] = channel_combos
                img_props['num_combos'] = num_combos

                # Calculate the individual CCFs and shifts for each channel
                ccf_smoothing = (smoothing_params or {}).get("CCF")
                if num_channels > 1:
                    indv_shifts = np.zeros(shape=(num_submovies, num_combos, num_bins))
                    indv_ccfs = np.zeros(shape=(num_submovies, num_combos, num_bins, roll_size*2-1))
                    its = num_submovies*num_combos*num_bins
                    with tqdm(total = its, miniters=its/100) as pbar:
                        pbar.set_description( 'Shifts: ')
                        for submovie in range(num_submovies):
                            for combo_number, combo in enumerate(channel_combos):
                                for bin in range(num_bins):
                                    pbar.update(1)
                                    signal1 = bin_values[roll_by*submovie : roll_size + roll_by*submovie, combo[0], bin]
                                    signal2 = bin_values[roll_by*submovie : roll_size + roll_by*submovie, combo[1], bin]
                                    ccf = sp.calc_indv_CCF(signal1=signal1, signal2=signal2, num_frames=roll_size, ccf_smoothing=ccf_smoothing)
                                    indv_ccfs[submovie, combo_number, bin] = ccf
                                    
                                    shift = sp.calc_indv_shift(cc_curve=ccf, ccf_peak_thresh=ccf_peak_thresh)
                                    if small_shifts_correction:
                                        average_period = np.mean(indv_periods[:, :, bin])
                                        shift = sp.correct_small_shifts(delay_frames=shift, average_period=average_period)
                                    indv_shifts[submovie, combo_number, bin] = shift

                # create a subfolder within the main save path with the same name as the image file
                im_save_path = os.path.join(main_save_path, file_stem)
                os.makedirs(im_save_path, exist_ok=True) if not test else None

                # adjust the different waves properties to be the use the frame interval rather than the number of frames
                indv_periods = indv_periods * img_props['frame_interval']
                indv_peak_offsets = indv_peak_offsets * img_props['frame_interval']
                indv_peak_widths = indv_peak_widths * img_props['frame_interval']

                img_metrics = {
                                'Period': indv_periods,
                                'Peak Amp': indv_peak_amps,
                                'Peak Rel Amp': indv_peak_rel_amps,
                                'Peak Width': indv_peak_widths,
                                'Peak Max': indv_peak_maxs,
                                'Peak Min': indv_peak_mins,
                                'Peak Offset': indv_peak_offsets,
                                'Peak Area': indv_peak_areas
                }

                # add shifts to the dictionary if there are multiple channels
                if img_props['num_channels'] > 1:
                    indv_shifts = indv_shifts * img_props['frame_interval']
                    img_metrics['Shift'] = indv_shifts
                    img_metrics['% Phase Shift'] = indv_shifts / indv_periods

                # calculate the number of subframes used
                log_params['Submovies Used'].append(num_submovies)

                ############################################
                ############## Saving and Summary ##########
                ############################################

                # summarize the data for each subframe as individual dataframes, and save as .csv
                submovie_meas_list, _ = summarize_image(
                    img_props=img_props,
                    img_metrics=img_metrics
                )
                csv_save_path = os.path.join(im_save_path, 'rolling_measurements')
                os.makedirs(csv_save_path, exist_ok=True) if not test else None
                for measurement_index, submovie_meas_df in enumerate(submovie_meas_list):  # type: ignore
                    submovie_meas_df.to_csv(f'{csv_save_path}/{file_stem}_subframe{measurement_index}_measurements.csv', index = False) if not test else None
                
                # summarize the data for each subframe as a single dataframe, and save as .csv
                summary_df = combine_stats_rolling(
                    img_props=img_props,
                    img_metrics=img_metrics,
                    indv_ccfs=indv_ccfs if num_channels > 1 else None
                )
                if not test:
                    summary_df.to_csv(f'{im_save_path}/{file_stem}_summary.csv', index = False)

                # make and save the summary plot for rolling data
                summary_plots = pt.plot_rolling_summary(
                    num_channels=num_channels,
                    fullmovie_summary=summary_df,
                    channel_combos=channel_combos,
                    dark_plots=dark_plots
                )
                plot_save_path = os.path.join(im_save_path, 'summary_plots')
                os.makedirs(plot_save_path, exist_ok=True) if not test else None
                hf.save_plots(summary_plots, plot_save_path) if not test else None

                end = timeit.default_timer()
                log_params["Time Elapsed"] = f"{end - start:.2f} seconds"
                # log parameters and errors
                hf.make_log(main_save_path, log_params) if not test else None

                # log that the file was processed
                log_params['Files Processed'].append(f'{file_name}')

            except Exception as e:
                print(f"****** ERROR ******",
                        f"\nError processing {file_name}: {str(e)}",
                        "\n****** ERROR ******")
                log_params['Errors'].append(f'Error processing {file_name}: {str(e)}')

            pbar.update(1)

            if test and file_stem == '1_Group2':
                return summary_df