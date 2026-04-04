import os
import timeit
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Any
import waveanalysis.plotting as pt
import waveanalysis.signal_processing as sp
import waveanalysis.housekeeping.housekeeping_functions as hf

from waveanalysis.data_workflows._helpers import _setup_workflow, _load_image_props, _smooth_bin_values_inplace
from waveanalysis.image_props.image_bin_calc import create_multi_frame_bin_array, create_kymo_bin_array, smooth_signal
from waveanalysis.image_props.image_to_np_arrays import tiff_to_np_array_multi_frame, tiff_to_np_array_single_frame
from waveanalysis.summarize_save.save_stats import save_parameter_means_to_csv, get_mean_CCF_values, get_indv_CCF_values, save_ccf_values_to_csv
from waveanalysis.summarize_save.summarize_images import summarize_image, combine_stats_for_image_kymo_standard
#import pickle

def combined_workflow(
    folder_path: str,
    group_names: list[str],
    log_params: dict[str, Any],
    analysis_type: str,
    acf_peak_thresh: float,
    ccf_peak_thresh: float,
    small_shifts_correction: bool,
    plot_flags: dict[str, bool],
    calc_wave_speeds: bool = False,
    plot_wave_speeds: bool = False,
    box_size: int = None,
    bin_shift: int = None, 
    line_width: int = None,
    test: bool = False, # for testing purposes
    smoothing_params: dict = None,
    smoothing: bool = False,
) -> pd.DataFrame:
    '''
    This is the combined workflow for kymographs and standard analysis. It processes the image files in the 
    specified folder and saves the summary data and figures to a new folder in the same directory as the 
    image files.

    It functions generally in this order (with some analysis specific steps):
        1. Convert a folder of tiff images to numpy arrays
        2. Iterate over every images in the folder
            a. Get the image properties
            b. Calculate the bin values based on the user provided box/line size and bin shift
            c. Calculate the ACF, period, peak properties, and CCFs/shifts (if specified)
                i. For kymographs, the user will be prompted to define the wave tracks (if specified)
            d. Plot the mean ACF, peak properties, wave speed, and CCF figures (if specified)
            e. Plot the individual ACF, peak properties, and CCF figures (if specified)
            f. Save the summary data and figures to a new folder in the same directory as the image files
        3. Generate the summary data for the entire folder and save it to a csv file
        4. Generate the group comparison figures and save them to a new folder in the same directory as the image files (if group names are specified)
        5. Generate the mean parameter measurements for each group and save them to a new folder in the same directory as the image files (if group names are specified)
        6. Log the parameters and errors to a log file in the new folder

    Parameters:
    - folder_path (str): The path to the folder containing the image files.
    - group_names (list[str]): The list of group names to match with the image files.
    - log_params (dict[str, Any]): The dictionary to store the log parameters.
    - analysis_type (str): The type of analysis to perform ('standard' or 'kymograph').
    - acf_peak_thresh (float): The threshold for detecting peaks in the ACF curve.
    - plot_flags (dict[str, bool]): A dictionary containing flags for plotting different types of figures.
    - calc_wave_speeds (bool, optional): Whether to calculate wave speeds. Defaults to False.
    - plot_wave_speeds (bool, optional): Whether to plot the wave speeds. Defaults to False.
    - box_size (int, optional): The size of the box for standard analysis. Defaults to None.
    - bin_shift (int, optional): The shift value for binning. Defaults to None.
    - line_width (int, optional): The width of the line for kymograph analysis. Defaults to None.
    - test (bool, optional): Whether to run in test mode (does not save files). Defaults to False.
    - Ch1_window (int, optional): The window size for Savitzky-Golay filter for channel 1. Defaults to None.
    - Ch1_poly_order (int, optional): The polynomial order for Savitzky-Golay filter for channel 1. Defaults to None.
    - Ch2_window (int, optional): The window size for Savitzky-Golay filter for channel 2. Defaults to None.
    - Ch2_poly_order (int, optional): The polynomial order for Savitzky-Golay filter for channel 2. Defaults to None.
    - CCF_window (int, optional): The window size for Savitzky-Golay filter for CCF. Defaults to None.
    - CCF_poly_order (int, optional): The polynomial order for Savitzky-Golay filter for CCF. Defaults to None. 
    - smoothing (bool, optional): Whether to apply smoothing to the signals. Defaults to True.
    
    Returns:
    - pd.DataFrame: The summary data for each file.
    '''
    file_names, start, now, main_save_path = _setup_workflow(folder_path, test)
    hf.group_name_error_check(file_names=file_names, group_names=group_names, log_params=log_params)

    # empty list to fill with summary data for each file, and column headers list
    summary_list, col_headers = [], []

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

                img_props = _load_image_props(
                    image_path, log_params, file_name,
                    bin_shift=bin_shift,
                    box_size=box_size if analysis_type == 'standard' else None,
                    acf_peak_thresh=acf_peak_thresh,
                    analysis_type=analysis_type,
                    extra_props={
                        'line_width': line_width if analysis_type == 'kymograph' else None,
                        'analysis_type': analysis_type,
                    },
                )
                if img_props is None:
                    log_params['Files Not Processed'].append(file_name)
                    log_params['Errors'].append(f'{file_name} has less than 11 frames')
                    continue

                if analysis_type == 'standard':
                    image_array = tiff_to_np_array_multi_frame(image_path)
                    bin_values, num_bins, num_x_bins, num_y_bins = create_multi_frame_bin_array(image=image_array, img_props=img_props)
                    img_props['num_x_bins'] = num_x_bins
                    img_props['num_y_bins'] = num_y_bins
                    raw_bin_values = bin_values.copy() if smoothing else None
                    _smooth_bin_values_inplace(bin_values, num_bins, img_props['num_channels'], smoothing_params)
                else:  # kymograph
                    image_array = tiff_to_np_array_single_frame(image_path)
                    bin_values, num_bins = create_kymo_bin_array(image=image_array, img_props=img_props)
                    raw_bin_values = bin_values.copy() if smoothing else None
                    for channel in range(img_props['num_channels']):
                        ch_params = (smoothing_params or {}).get(f"Ch{channel + 1}")
                        if ch_params is not None:
                            for bin in range(num_bins):
                                bin_values[channel, bin] = smooth_signal(signal=bin_values[channel, bin], window=ch_params["window"], poly_order=ch_params["poly_order"])
                    
                # get the channel combinations
                channel_combos = hf.get_channel_combos(num_channels=img_props['num_channels'])
                num_combos = len(channel_combos)
                img_props['channel_combos'] = channel_combos
                img_props['num_combos'] = num_combos

                # store the number of bins and the bin values in the image properties dictionary
                img_props['num_bins'] = num_bins
                img_props['bin_values'] = bin_values
                
                # with open(f'/Users/domchom/Desktop/{file_name}_img_props.json', 'w') as f:
                #    json.dump(img_props, f, default=str)

                # if user entered group name(s) into GUI, match the group for this file. If no match, keep set to None
                file_stem = file_name.rsplit(".",1)[0]
                group_name = hf.match_group_to_file(name_wo_ext=file_stem, group_names=group_names)

                ############################################
                ############## Signal Processing ###########
                ############################################

                # Calculate the ACF
                indv_acfs = sp.calc_indv_ACF_workflow(bin_values=bin_values, img_props=img_props)

                # Calculate the period
                indv_periods = sp.calc_indv_period_workflow(acf_curve=indv_acfs, img_props=img_props)

                # Calculate the peak properties
                indv_peak_widths, indv_peak_maxs, indv_peak_mins, indv_peak_offsets, indv_peak_props, indv_peak_areas = sp.calc_indv_peak_props_workflow(bin_values=bin_values, img_props=img_props)
                
                # with open(f'/Users/domchom/Desktop/{file_name}_peak_props.pkl', 'wb') as f:
                #    pickle.dump(indv_peak_props, f)
                
                indv_peak_amps = indv_peak_maxs - indv_peak_mins
                indv_peak_rel_amps = indv_peak_amps / indv_peak_mins
                
                # Calculate the individual CCFs and shifts
                if img_props['num_channels'] > 1:
                    indv_ccfs = sp.calc_indv_CCF_workflow(bin_values=bin_values, img_props=img_props, ccf_smoothing=(smoothing_params or {}).get("CCF"))
                    indv_shifts = sp.calc_indv_shift_workflow(indv_ccfs=indv_ccfs, indv_periods=indv_periods, img_props=img_props, small_shifts_correction=small_shifts_correction, ccf_peak_thresh=ccf_peak_thresh)
                    
                    # Save individual CCFs to pickle file
                    # with open(f'/Users/domchom/Desktop/{file_name}_indv_ccfs.pkl', 'wb') as f:
                    #    pickle.dump(indv_ccfs, f)

                # adjust the different waves properties to be the use the frame interval rather than the number of frames
                indv_periods = indv_periods * img_props['frame_interval']
                indv_peak_offsets = indv_peak_offsets * img_props['frame_interval']
                indv_peak_widths = indv_peak_widths * img_props['frame_interval']

                # create dictionary of image parameters and their values for later use
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
                    channel_combos = img_props['channel_combos']
                    indv_phase_shifts = np.zeros_like(indv_shifts)
                    for combo_idx, (ch1, ch2) in enumerate(channel_combos):
                        combo_period = np.nanmean(indv_periods[[ch1, ch2], :], axis=0)
                        indv_phase_shifts[combo_idx] = (indv_shifts[combo_idx] / combo_period) * 100
                    img_metrics['% Phase Shift'] = indv_phase_shifts
                    
                # create the directory to save the figures and data for the image
                im_save_path = os.path.join(main_save_path, file_stem)
                os.makedirs(im_save_path, exist_ok=True) if not test else None

                ############################################
                ############## Plotting ####################
                ############################################

                # plot the mean ACF figures for the file
                if plot_flags["plot_summary_ACFs"]:
                    mean_acf_figs = pt.plot_mean_acf_workflow(
                        img_metrics=img_metrics,
                        img_props=img_props,
                        indv_acfs=indv_acfs,
                        dark_plots=plot_flags["dark_plots"]
                    )
                    hf.save_plots(mean_acf_figs, im_save_path)

                # plot the mean peak properties figures for the file
                if plot_flags["plot_summary_peaks"]:
                    mean_peak_figs = pt.plot_mean_peak_props_workflow(
                        img_metrics=img_metrics,
                        img_props=img_props,
                        dark_plots=plot_flags["dark_plots"]
                    )
                    hf.save_plots(mean_peak_figs, im_save_path)

                # plot spatial metric heatmaps (standard analysis only)
                if plot_flags["plot_heatmaps"] and analysis_type == 'standard':
                    heatmap_figs = pt.plot_metric_heatmaps_workflow(
                        img_metrics=img_metrics,
                        img_props=img_props,
                        image_array=image_array,
                        dark_plots=plot_flags["dark_plots"],
                    )
                    heatmap_path = os.path.join(im_save_path, 'Metric_Heatmaps')
                    os.makedirs(heatmap_path, exist_ok=True)
                    hf.save_plots(heatmap_figs, heatmap_path)

                # plot the mean CCF figures for the file
                if plot_flags["plot_summary_CCFs"] and img_props['num_channels'] > 1:
                    mean_ccf_figs = pt.plot_mean_ccf_workflow(
                        img_metrics=img_metrics,
                        img_props=img_props,
                        indv_ccfs=indv_ccfs,
                        dark_plots=plot_flags["dark_plots"]
                    )
                    hf.save_plots(mean_ccf_figs, im_save_path)
                    # save the mean CCF values for the file
                    mean_ccf_values = get_mean_CCF_values(channel_combos=channel_combos, indv_ccfs=indv_ccfs, frame_interval=img_props['frame_interval'])
                    save_ccf_values_to_csv(mean_ccf_values, im_save_path)

                # Error check for plotting individual CCFs
                elif plot_flags["plot_summary_CCFs"] and img_props['num_channels'] == 1:
                    log_params['Miscellaneous'] = f'CCF plots were not generated for {file_name} because the image only has one channel'

                # plot the individual ACF figures for the file
                if plot_flags["plot_indv_ACFs"]:
                    indv_acf_plots = pt.plot_indv_acf_workflow(
                        raw_bin_values=raw_bin_values,
                        bin_values=bin_values,
                        indv_acfs=indv_acfs,
                        img_metrics=img_metrics,
                        img_props=img_props,
                        dark_plots=plot_flags["dark_plots"]
                    )
                    indv_acf_path = os.path.join(im_save_path, 'Individual_ACF_plots')
                    os.makedirs(indv_acf_path, exist_ok=True)
                    hf.save_plots(indv_acf_plots, indv_acf_path)

                # plot the individual peak properties figures for the file
                if plot_flags["plot_indv_peaks"]:        
                    indv_peak_figs = pt.plot_indv_peak_workflow(
                        raw_bin_values=raw_bin_values if raw_bin_values is not None else bin_values,
                        img_props=img_props,
                        indv_peak_props=indv_peak_props,
                        num_frames=img_props['num_frames'],
                        dark_plots=plot_flags["dark_plots"]
                    )
                    indv_peak_path = os.path.join(im_save_path, 'Individual_peak_plots')
                    os.makedirs(indv_peak_path, exist_ok=True)
                    hf.save_plots(indv_peak_figs, indv_peak_path)
                    
                # plot the individual CCF figures for the file
                if plot_flags["plot_indv_CCFs"] and img_props['num_channels'] > 1:
                    if img_props['num_channels'] == 1:
                        log_params['Miscellaneous'] = f'CCF plots were not generated for {file_name} because the image only has one channel'
                    indv_ccf_plots = pt.plot_indv_ccf_workflow(
                        bin_values=bin_values,
                        indv_ccfs=indv_ccfs,
                        img_metrics=img_metrics,
                        img_props=img_props,
                        dark_plots=plot_flags["dark_plots"]
                    )
                    indv_ccf_plots_path = os.path.join(im_save_path, 'Individual_CCF_plots')
                    os.makedirs(indv_ccf_plots_path, exist_ok=True)
                    hf.save_plots(indv_ccf_plots, indv_ccf_plots_path)
                    # save the individual CCF values for the file
                    indv_ccf_values = get_indv_CCF_values(
                        indv_ccfs=indv_ccfs,
                        bin_values=bin_values,
                        img_props=img_props
                    )
                    indv_ccf_val_path = os.path.join(im_save_path, 'Individual_CCF_values')
                    os.makedirs(indv_ccf_val_path, exist_ok=True)
                    save_ccf_values_to_csv(indv_ccf_values, indv_ccf_val_path)                    

                ############################################
                ############## Saving ######################
                ############################################

                # Summarize the data for current image as dataframe, and save as .csv
                im_measurements_df, stats_by_parameter = summarize_image(
                    img_metrics=img_metrics,
                    img_props=img_props
                )
                im_measurements_df.to_csv(f'{im_save_path}/{file_stem}_measurements.csv', index = False) if not test else None
                
                # generate stats for the image such as mean, median, std, etc
                file_summary = combine_stats_for_image_kymo_standard(
                    file_name=file_name, 
                    group_name=group_name,
                    img_props=img_props,
                    img_metrics=img_metrics,
                    stats_by_parameter=stats_by_parameter
                )

                # populate column headers list with keys from the measurements dictionary
                for key in file_summary.keys(): 
                    if key not in col_headers: 
                        col_headers.append(key) 
            
                # append summary data to the summary list
                summary_list.append(file_summary)

                # log that the file was processed
                log_params['Files Processed'].append(f'{file_name}')
            
            except Exception as e:
                print(f"****** ERROR ******",
                        f"\nError processing {file_name}: {str(e)}",
                        "\n****** ERROR ******")
                log_params['Errors'].append(f'Error processing {file_name}: {str(e)}')
                log_params['Files Not Processed'].append(f'{file_name}')

            # useless progress bar to force completion of previous bars
            with tqdm(total = 10, miniters = 1) as dummy_pbar:
                dummy_pbar.set_description('cleanup:')
                for i in range(10):
                    dummy_pbar.update(1)
            pbar.update(1)

        ############################################
        ############## Summary #####################
        ############################################

        # create dataframe from summary list, then sort and save the summary to a csv file
        summary_df = pd.DataFrame(summary_list, columns=col_headers)
        summary_df = summary_df.sort_values('File Name', ascending=True)
        summary_df.to_csv(f"{main_save_path}/!{now.strftime('%Y%m%d%H%M')}_summary.csv", index = False) if not test else None

        if group_names != ['']:
            # generate comparisons between each group
            mean_parameter_figs = pt.generate_group_comparison(summary_df = summary_df, log_params = log_params, dark_plots = plot_flags["dark_plots"])
            group_plots_save_path = os.path.join(main_save_path, "group_comparison_graphs")
            os.makedirs(group_plots_save_path, exist_ok=True) if not test else None
            hf.save_plots(mean_parameter_figs, group_plots_save_path) if not test else None

            # save the means each parameter for the attributes to make them easier to work with 
            parameter_tables_dict = save_parameter_means_to_csv(summary_df=summary_df,group_names=group_names)
            mean_measurements_save_path = os.path.join(main_save_path, "mean_parameter_measurements")
            os.makedirs(mean_measurements_save_path, exist_ok=True) if not test else None
            for filename, table in parameter_tables_dict.items():
                table.to_csv(f"{mean_measurements_save_path}/{filename}", index = False) if not test else None

        # performance tracker end
        end = timeit.default_timer()

        # log parameters and errors
        log_params["Time Elapsed"] = f"{end - start:.2f} seconds"
        hf.make_log(main_save_path, log_params) if not test else None

        if log_params['Errors']:
            print('*' * 50)
            print('*' * 50)
            print('ERRORS WERE ENCOUNTERED DURING PROCESSING. PLEASE CHECK THE LOG FILE FOR MORE INFORMATION.')
            print('*' * 50)
            print('*' * 50)

        return summary_df # only here for testing