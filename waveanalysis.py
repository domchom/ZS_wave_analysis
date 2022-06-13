import os                                       
import sys 
import timeit
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from waveanalysismods.customgui import BaseGUI, RollingGUI
from waveanalysismods.processor import TotalSignalProcessor, RollingSignalProcessor

np.seterr(divide='ignore', invalid='ignore')

'''** GUI Window and sanity checks'''
# make GUI object and display the window
gui = BaseGUI()
gui.mainloop()
# get GUI parameters
rolling = False
box_size = gui.box_size
folder_path = gui.folder_path
group_names = gui.group_names
acf_peak_thresh = gui.acf_peak_thresh
plot_summary_CFs = gui.plot_summary_CFs
plot_summary_peaks = gui.plot_summary_peaks
# if rolling GUI specified, make rolling GUI object and display the window
if gui.roll:
    rolling = True
    gui = RollingGUI()
    gui.mainloop()
    # get GUI parameters
    box_size = gui.box_size
    folder_path = gui.folder_path
    acf_peak_thresh = gui.acf_peak_thresh
    plot_sf_ACFs = gui.plot_sf_ACFs
    plot_sf_CCFs = gui.plot_sf_CCFs
    plot_sf_peaks = gui.plot_sf_peaks
    subframe_size = gui.subframe_size
    subframe_roll = gui.subframe_roll

    group_names = ['']
    plot_summary_ACFs = False
    plot_summary_CCFs = False
    plot_summary_peaks = False

# identify and report errors in GUI input
errors = []
if gui.acf_peak_thresh > 1 :
    errors.append("The ACF peak prominence can not be greater than 1",
                  ", set 'ACF peak prominence threshold' to a value between 0 and 1.",
                  "More realistically, a value between 0 and 0.5")
if len(gui.folder_path) < 1 :
    errors.append("You didn't enter a directory to analyze")

if len(errors) >= 1 :
    print("Error Log:")
    for count, error in enumerate(errors):
        print(count,":", error)
    sys.exit("Please fix errors and try again.") 

#make dictionary of parameters for log file use
log_params = {  "Box Size(px)" : box_size,
                "Base Directory" : folder_path,
                "ACF Peak Prominence" : acf_peak_thresh,
                "Group Names" : group_names,
                "Plot Individual CFs" : plot_summary_CFs,
                "Plot Individual Peaks" : plot_summary_peaks,
                "Group Matching Errors" : [],
                "Files Processed" : [],
                "Files Not Processed" : [],
                'Plotting errors' : []
             } 
if rolling:
    log_params = {  "Box Size(px)" : box_size,
                    "Base Directory" : folder_path,
                    "ACF Peak Prominence" : acf_peak_thresh,
                    "Plot sub-movie ACFs" : plot_sf_ACFs,
                    "Plot movie CCFs" : plot_sf_CCFs,
                    "Plot movie Peaks" : plot_sf_peaks,
                    "Files Processed" : [],
                    "Files Not Processed" : [],
                    'Submovies Used' : []
             } 

''' ** housekeeping functions ** '''
def make_log(directory, logParams):
    '''
    Convert dictionary of parameters to a log file and save it in the directory
    '''
    now = datetime.datetime.now()
    logPath = os.path.join(directory, f"0_log-{now.strftime('%Y%m%d%H%M')}.txt")
    logFile = open(logPath, "w")                                    
    logFile.write("\n" + now.strftime("%Y-%m-%d %H:%M") + "\n")     
    for key, value in logParams.items():                            
        logFile.write('%s: %s\n' % (key, value))                    
    logFile.close()                                                 

def plotComparisons(dataFrame: pd.DataFrame, dependent: str, independent = 'Group Name'):
    '''
    This func accepts a dataframe, the name of a dependent variable, and the name of an
    independent variable (by default, set to Group Name). It returns a figure object showing
    a box and scatter plot of the dependent variable grouped by the independent variable.
    '''
    ax = sns.boxplot(x=independent, y=dependent, data=dataFrame, palette = "Set2", showfliers = False)
    ax = sns.swarmplot(x=independent, y=dependent, data=dataFrame, color=".25")	
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
    fig = ax.get_figure()
    return fig

''' ** error catching for group names ** '''
# list of file names in specified directory
file_names = filelist = [fname for fname in os.listdir(folder_path) if fname.endswith('.tif') and not fname.startswith('.')]

# list of groups that matched to file names
groups_found = np.unique([group for group in group_names for file in file_names if group in file]).tolist()

# dictionary of file names and their corresponding group names
uniqueDic = {file : [group for group in group_names if group in file] for file in file_names}

for file_name, matching_groups in uniqueDic.items():
    # if a file doesn't have a group name, log it but still run the script
    if len(matching_groups) == 0:
        log_params["Group Matching Errors"].append(f'{file_name} was not matched to a group')

    # if a file has multiple groups names, raise error and exit the script
    elif len(matching_groups) > 1:
        print('****** ERROR ******',
             f'\n{file_name} matched to multiple groups: {matching_groups}',
             '\nPlease fix errors and try again.',
             '\n****** ERROR ******')
        sys.exit()

# if a group was specified but not matched to a file name, raise error and exit the script
if len(groups_found) != len(group_names):
    print("****** ERROR ******",
          "\nOne or more groups were not matched to file names",
          f"\nGroups specified: {group_names}",
          f"\nGroups found: {groups_found}",
          "\n****** ERROR ******")
    sys.exit()

''' ** Main Workflow ** '''
# performance tracker
start = timeit.default_timer()
# create main save path
now = datetime.datetime.now()
main_save_path = os.path.join(folder_path, f"0_signalProcessing-{now.strftime('%Y%m%d%H%M')}")
# create directory if it doesn't exist
if not os.path.exists(main_save_path):
    os.makedirs(main_save_path)

# empty list to fill with summary data for each file
summary_list = []
# column headers to use with summary data during conversion to dataframe
col_headers = []

print('Processing files...')

if not rolling:
    with tqdm(total = len(file_names)) as pbar:
        for file_name in file_names: 
            print('******'*10)
            print(f'Processing {file_name}...')
            processor = TotalSignalProcessor(image_path = f'{folder_path}/{file_name}', kern = box_size, step = box_size)
            ''' WILL WANT TO CHANGE BOX TO KERN SIZE AND ADD STEP AS AN OPTION TO THE GUI'''

            # log error and skip image if frames < 2 
            if processor.num_frames < 2:
                print(f"****** ERROR ******",
                    f"\n{file_name} has less than 2 frames",
                    "\n****** ERROR ******")
                log_params['Files Not Processed'].append(f'{file_name} has less than 2 frames')
                continue

            # if file is not skipped, log it and continue
            log_params['Files Processed'].append(f'{file_name}')

            # name without the extension
            name_wo_ext = file_name.rsplit(".",1)[0]

            # if user entered group name(s) into GUI, match the group for this file. If no match, keep set to None
            group_name = None
            if group_names != ['']:
                try:
                    group_name = [group for group in group_names if group in name_wo_ext][0]
                except IndexError:
                    pass

            # calculate the number of boxes used for analysis
            num_meas = processor.xpix * processor.ypix

            # calculate the population signal properties
            processor.calc_ACF(peak_thresh = acf_peak_thresh)
            processor.calc_peak_props()
            if processor.num_channels > 1:
                processor.calc_CCF()
            
            # create a subfolder within the main save path with the same name as the image file
            im_save_path = os.path.join(main_save_path, name_wo_ext)
            if not os.path.exists(im_save_path):
                os.makedirs(im_save_path)

            # plot and save the population autocorrelation results for each channel
            if plot_summary_CFs:
                acf_plots = processor.plot_mean_CF()
                for plot_name, plot in acf_plots.items():
                    plot.savefig(f'{im_save_path}/{plot_name}.png')
                    
            # plot and save the population peak properties for each channel
            if plot_summary_peaks:
                peak_plots = processor.plot_peak_props()
                for plot_name, plot in peak_plots.items():
                    plot.savefig(f'{im_save_path}/{plot_name}.png')

            # Summarize the data for current image as dataframe, and save as .csv
            im_measurements_df = processor.organize_measurements()
            im_measurements_df.to_csv(f'{im_save_path}/{name_wo_ext}_measurements.csv', index = False)

            # generate summary data for current image
            im_summary_dict = processor.summarize_image(file_name = file_name, group_name = group_name)

            # populate column headers list with keys from the measurements dictionary
            for key in im_summary_dict.keys(): 
                if key not in col_headers: 
                    col_headers.append(key) 
        
            # append summary data to the summary list
            summary_list.append(im_summary_dict)

            pbar.update(1)

         # create dataframe from summary list    
        summary_df = pd.DataFrame(summary_list, columns = col_headers)
        summary_df.to_csv(f'{main_save_path}/summary.csv', index = False)

        # if group names were entered into the gui, generate comparisons between each group
        if group_names != ['']:
            print('Generating group comparisons...')
            # make a group comparisons save path in the main save directory
            group_save_path = os.path.join(main_save_path, "0_groupComparisons")
            if not os.path.exists(group_save_path):
                os.makedirs(group_save_path)
            
            # make a list of parameters to compare
            stats_to_compare = ['Mean']
            channels_to_compare = [f'Ch {i+1}' for i in range(processor.num_channels)]
            measurments_to_compare = ['Period', 'Shift', 'Peak Width', 'Peak Max', 'Peak Min', 'Peak Amp', 'Peak Rel Amp']
            params_to_compare = []
            for channel in channels_to_compare:
                for stat in stats_to_compare:
                    for measurment in measurments_to_compare:
                        params_to_compare.append(f'{channel} {stat} {measurment}')

            shifts_to_compare = [f'Ch{combo[0]+1}-Ch{combo[1]+1} Mean Shift' for combo in processor.channel_combos]
            params_to_compare.extend(shifts_to_compare)

            # generate and save figures for each parameter
            for param in params_to_compare:
                try:
                    fig = plotComparisons(summary_df, param)
                    fig.savefig(f'{group_save_path}/{param}.png')
                    plt.close(fig)
                except ValueError:
                    log_params['Plotting errors'].append(f'No data to compare for {param}')

        end = timeit.default_timer()
        log_params["Time Elapsed"] = f"{end - start:.2f} seconds"
        # log parameters and errors
        make_log(main_save_path, log_params)
        print('Done!')


if rolling:
    with tqdm(total = len(file_names)) as pbar:
        for file_name in file_names: 
            print('******'*10)
            print(f'Processing {file_name}...')
            processor = RollingSignalProcessor(image_path = f'{folder_path}/{file_name}', kern = box_size, step = box_size, roll_size = subframe_size, roll_by = subframe_roll)

            # log error and skip image if frames < 2 
            if processor.num_frames < 2:
                print(f"****** ERROR ******",
                    f"\n{file_name} has less than 2 frames",
                    "\n****** ERROR ******")
                log_params['Files Not Processed'].append(f'{file_name} has less than 2 frames')
                continue
        
            # if file is not skipped, log it and continue
            log_params['Files Processed'].append(f'{file_name}')

            # name without the extension
            name_wo_ext = file_name.rsplit(".",1)[0]
        
            # calculate the number of boxes used for analysis
            num_meas = processor.xpix * processor.ypix

            # calculate the numbe of subframes used
            num_submovies = processor.num_submovies
            log_params['Submovies Used'].append(num_submovies)

            # calculate the population signal properties
            processor.calc_ACF(peak_thresh = acf_peak_thresh)
            processor.calc_peak_props()
            if processor.num_channels > 1:
                processor.calc_CCF()
        
            # create a subfolder within the main save path with the same name as the image file
            im_save_path = os.path.join(main_save_path, name_wo_ext)
            if not os.path.exists(im_save_path):
                os.makedirs(im_save_path)

            # summarize the data for each subframe as individual dataframes, and save as .csv
            submovie_meas_list = processor.get_submovie_measurements()
            csv_save_path = os.path.join(im_save_path, 'rolling_measurements')
            if not os.path.exists(csv_save_path):
                os.makedirs(csv_save_path)
            for measurement_index, submovie_meas_df in enumerate(submovie_meas_list):
                submovie_meas_df.to_csv(f'{csv_save_path}/{name_wo_ext}_subframe{measurement_index}_measurements.csv', index = False)
            
            # summarize the data for each subframe as a single dataframe, and save as .csv
            summary_df = processor.summarize_file()
            summary_df.to_csv(f'{im_save_path}/{name_wo_ext}_summary.csv', index = False)

            # make and save the summary plot for rolling data
            summary_plots = processor.plot_rolling_summary()
            plot_save_path = os.path.join(im_save_path, 'summary_plots')
            if not os.path.exists(plot_save_path):
                os.makedirs(plot_save_path)
            for title, plot in summary_plots.items():
                plot.savefig(f'{plot_save_path}/{name_wo_ext}_{title}.png')
            pbar.update(1)