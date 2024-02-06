import scipy
import numpy as np
import pandas as pd
import scipy.signal as sig
from tqdm import tqdm
import matplotlib.pyplot as plt
from tifffile import imread, TiffFile
import scipy.ndimage as nd
np.seterr(divide='ignore', invalid='ignore')

class TotalSignalProcessor:
    
    def __init__(self, analysis_type, image_path, kern = None, step=None, roll_size = None, roll_by = None, line_width = None):
        self.analysis_type = analysis_type
        if self.analysis_type != "kymograph":
            self.image_path = image_path
            self.kernel_size = kern
            self.image = imread(self.image_path)
            self.step  = step

            # standardize image dimensions
            with TiffFile(self.image_path) as tif_file:
                metadata = tif_file.imagej_metadata
            self.num_channels = metadata.get('channels', 1)
            self.num_slices = metadata.get('slices', 1)
            self.num_frames_or_rows = metadata.get('frames', 1)
            self.image = self.image.reshape(self.num_frames_or_rows, 
                                            self.num_slices, 
                                            self.num_channels, 
                                            self.image.shape[-2], 
                                            self.image.shape[-1])

            # max project image stack if num_slices > 1
            if self.num_slices > 1:
                print(f'Max projecting image stack')
                self.image = np.max(self.image, axis = 1)
                self.num_slices = 1
                self.image = self.image.reshape(self.num_frames_or_rows, 
                                                self.num_slices, 
                                                self.num_channels, 
                                                self.image.shape[-2], 
                                                self.image.shape[-1])
                
            if analysis_type == "rolling":
                self.roll_size = roll_size
                self.roll_by = roll_by

                # sanity checks
                assert type(self.roll_size) == int and type(self.roll_by) == int, 'Roll size and roll by must be integers'
                # specify the number of submovies to analyze
                self.num_submovies = (self.num_frames_or_rows - roll_size) // roll_by

            # return the time-axis means for each channel
            ind = kern // 2
            self.means = nd.uniform_filter(self.image[:,0,:,:,:], size = (1,1,kern,kern))[:,:,ind:-ind:step, ind:-ind:step]
            self.xpix = self.means.shape[-2]
            self.ypix = self.means.shape[-1]
            self.num_boxes_or_cols = self.xpix*self.ypix
            self.means = self.means.reshape(self.means.shape[0], self.means.shape[1], self.num_boxes_or_cols)
        
        else:
            self.image_path = image_path
            self.line_width = line_width
            self.image = imread(self.image_path)

            # standardize image dimensions
            with TiffFile(self.image_path) as tif_file:
                metadata = tif_file.imagej_metadata
            self.num_channels = metadata.get('channels', 1)
            self.image = self.image.reshape(self.num_channels, 
                                        self.image.shape[-2],  # rows
                                        self.image.shape[-1])  # cols
            
            self.num_boxes_or_cols = self.image.shape[-1]
            self.num_frames_or_rows = self.image.shape[-2]
            
            smooth_length = 25

            self.indv_line_values = np.zeros(shape=(self.num_channels, self.num_boxes_or_cols, self.num_frames_or_rows))
            for channel in range(self.num_channels):
                for col_num in range(self.num_boxes_or_cols):
                    if self.line_width == 1:
                        signal = sig.savgol_filter(self.image[channel, :, col_num], window_length = smooth_length, polyorder = 2)
                        self.indv_line_values[channel, col_num] = signal
                    elif self.line_width % 2 != 0:
                        line_width_extra = int((self.line_width - 1) / 2)
                        if col_num + line_width_extra < self.num_boxes_or_cols and col_num - line_width_extra > -1:
                            signal = np.mean(self.image[channel, :, col_num-line_width_extra:col_num+line_width_extra], axis=1)
                            signal = sig.savgol_filter(signal, window_length = smooth_length, polyorder=2)
                            self.indv_line_values[channel, col_num] = signal

##############################################################################################################################################################################
# INDIVIDUAL CALCULATION #####################################################################################################################################################
##############################################################################################################################################################################

    # function to return the autocorrelation of each box in the image stack for each channel
    def calc_indv_ACFs(self, peak_thresh=0.1):
        def norm_and_calc_shifts(signal, num_frames_or_rows_or_rollsize = None):
            corr_signal = signal - signal.mean()
            acf_curve = np.correlate(corr_signal, corr_signal, mode='full')
            # normalize the curve
            acf_curve = acf_curve / (num_frames_or_rows_or_rollsize * signal.std() ** 2)
            peaks, _ = sig.find_peaks(acf_curve, prominence=peak_thresh)
            # absolute difference between each peak and zero
            peaks_abs = abs(peaks - acf_curve.shape[0]//2)
            # if peaks were identified, pick the one closest to the center
            if len(peaks) > 1:
                delay = np.min(peaks_abs[np.nonzero(peaks_abs)])
            # otherwise, return nans for both period and autocorrelation curve
            else:
                delay = np.nan
                acf_curve = np.full((num_frames_or_rows_or_rollsize*2-1), np.nan)

            return delay, acf_curve

        # make empty arrays to populate with 1) period measurements and 2) acf curves
        self.periods = np.zeros(shape=(self.num_channels, self.num_boxes_or_cols))
        self.acfs = np.zeros(shape=(self.num_channels, self.num_boxes_or_cols, self.num_frames_or_rows*2-1))

        if self.analysis_type == "standard":
            for channel in range(self.num_channels):
                for box in range(self.num_boxes_or_cols):
                    # calculate full autocorrelation
                    signal = self.means[:,channel, box]
                    delay, acf_curve = norm_and_calc_shifts(signal, num_frames_or_rows_or_rollsize = self.num_frames_or_rows)
                    self.periods[channel, box] = delay
                    self.acfs[channel, box] = acf_curve
                    
        elif self.analysis_type == "rolling":
            self.periods = np.zeros(shape=(self.num_submovies, self.num_channels, self.num_boxes_or_cols))
            self.acfs = np.zeros(shape = (self.num_submovies, self.num_channels, self.num_boxes_or_cols, self.roll_size*2-1))

            for submovie in range(self.num_submovies):
                for channel in range(self.num_channels):
                    for box in range(self.num_boxes_or_cols):
                        # calculate full autocorrelation
                        signal = self.means[self.roll_by*submovie : self.roll_size + self.roll_by*submovie, channel, box]
                        delay, acf_curve = norm_and_calc_shifts(signal, num_frames_or_rows_or_rollsize = self.roll_size)
                        self.periods[channel, box] = delay
                        self.acfs[channel, box] = acf_curve

        elif self.analysis_type == "kymograph":
            for channel in range(self.num_channels):
                for col_num in range(self.num_boxes_or_cols):
                    # calculate full autocorrelation
                    signal = self.indv_line_values[channel, col_num, :]
                    delay, acf_curve = norm_and_calc_shifts(signal, num_frames_or_rows_or_rollsize = self.num_frames_or_rows)
                    self.periods[channel, col_num] = delay
                    self.acfs[channel, col_num] = acf_curve

        return self.acfs, self.periods


    def calc_indv_CCFs(self):
        def calc_shifts(signal1, signal2, prominence=0.1, rolling = False):
            #smooth signals and find peaks in the signals. Sanity check to continue on with calculating the individual shifts
            signal1 = scipy.signal.savgol_filter(signal1, window_length=11, polyorder=3)
            signal2 = scipy.signal.savgol_filter(signal2, window_length=11, polyorder=3)
            peaks1, _ = scipy.signal.find_peaks(signal1, prominence=(np.max(signal1)-np.min(signal1))*0.25)
            peaks2, _ = scipy.signal.find_peaks(signal2, prominence=(np.max(signal2)-np.min(signal2))*0.25)

            if len(peaks1) > 0 and len(peaks2) > 0:
                corr_signal1 = signal1 - signal1.mean()
                corr_signal2 = signal2 - signal2.mean()
                if rolling == True:
                    cc_curve = cc_curve / (self.roll_size * signal1.std() * signal2.std())
                else:
                    cc_curve = np.correlate(corr_signal1, corr_signal2, mode='full')
                    # smooth the curve
                    cc_curve = sig.savgol_filter(cc_curve, window_length=11, polyorder=3)
                    # normalize the curve
                    cc_curve = cc_curve / (self.num_frames_or_rows * signal1.std() * signal2.std())
                # find peaks
                peaks, _ = sig.find_peaks(cc_curve, prominence=prominence)
                # absolute difference between each peak and zero
                peaks_abs = abs(peaks - cc_curve.shape[0] // 2)
                # if peaks were identified, pick the one closest to the center
                if len(peaks) > 1:
                    delay = np.argmin(peaks_abs[np.nonzero(peaks_abs)])
                    delayIndex = peaks[delay]
                    delay_frames = delayIndex - cc_curve.shape[0] // 2
                # otherwise, return NaNs for both period and autocorrelation curve
                else:
                    delay_frames = np.nan
                    cc_curve = np.full((self.num_frames_or_rows * 2 - 1), np.nan)
            else:
                delay_frames = np.nan
                if rolling == True:
                    cc_curve = np.full((self.roll_size*2-1), np.nan)
                else:
                    cc_curve = np.full((self.num_frames_or_rows * 2 - 1), np.nan)

            return delay_frames, cc_curve
        
        # make a list of unique channel combinations to calculate CCF for
        channels = list(range(self.num_channels))
        self.channel_combos = []
        for i in range(self.num_channels):
            for j in channels[i+1:]:
                self.channel_combos.append([channels[i],j])
        num_combos = len(self.channel_combos)

        # make empty arrays to populate with 1) period measurements and 2) acf curves   
        self.indv_shifts = np.zeros(shape=(num_combos, self.num_boxes_or_cols))
        self.indv_ccfs = np.zeros(shape=(num_combos, self.num_boxes_or_cols, self.num_frames_or_rows*2-1))

        if self.analysis_type == "standard":
            for combo_number, combo in enumerate(self.channel_combos):
                for box in range(self.num_boxes_or_cols):
                    signal1 = self.means[:, combo[0], box]
                    signal2 = self.means[:, combo[1], box]

                    delay_frames, cc_curve = calc_shifts(signal1, signal2, prominence=0.1)

                    self.indv_shifts[combo_number, box] = delay_frames
                    self.indv_ccfs[combo_number, box] = cc_curve

        elif self.analysis_type == "rolling":
            # make empty arrays to populate with 1) period measurements and 2) acf curves   
            self.indv_shifts = np.zeros(shape=(self.num_submovies, num_combos, self.num_boxes_or_cols))
            self.indv_ccfs = np.zeros(shape=(self.num_submovies, num_combos, self.num_boxes_or_cols, self.roll_size*2-1))

            for submovie in range(self.num_submovies):
                for combo_number, combo in enumerate(self.channel_combos):
                    for box in range(self.num_boxes_or_cols):
                        signal1 = self.means[self.roll_by*submovie : self.roll_size + self.roll_by*submovie, combo[0], box]
                        signal2 = self.means[self.roll_by*submovie : self.roll_size + self.roll_by*submovie, combo[1], box]

                        self.indv_shifts[submovie, combo_number, box] = delay_frames
                        self.indv_ccfs[submovie, combo_number, box] = cc_curve
        
        elif self.analysis_type == "kymograph":
            for combo_number, combo in enumerate(self.channel_combos):
                for col_num in range(self.num_boxes_or_cols):
                    signal1 = self.indv_line_values[combo[0], col_num]
                    signal2 = self.indv_line_values[combo[1], col_num]

                    delay_frames, cc_curve = calc_shifts(signal1, signal2, prominence=0.1)

                    self.indv_shifts[combo_number, col_num] = delay_frames
                    self.indv_ccfs[combo_number, col_num] = cc_curve
        
        
        return self.indv_shifts, self.indv_ccfs, self.channel_combos

    def calc_indv_peak_props(self):

        def indv_props(signal, box_or_col_num, submovie = False):
            peaks, _ = sig.find_peaks(signal, prominence=(np.max(signal)-np.min(signal))*0.1)

            # if peaks detected, calculate properties and return property averages. Otherwise return nans
            if len(peaks) > 0:
                proms, _, _ = sig.peak_prominences(signal, peaks)
                widths, heights, leftIndex, rightIndex = sig.peak_widths(signal, peaks, rel_height=0.5)
                mean_width = np.mean(widths, axis=0)
                mean_max = np.mean(signal[peaks], axis = 0)
                mean_min = np.mean(signal[peaks]-proms, axis = 0)
            else:
                mean_width = np.nan
                mean_max = np.nan
                mean_min = np.nan
                peaks = np.nan
                proms = np.nan 
                heights = np.nan
                leftIndex = np.nan
                rightIndex = np.nan

            if submovie == False:
                self.ind_peak_widths[channel, box_or_col_num] = mean_width
                self.ind_peak_maxs[channel, box_or_col_num] = mean_max
                self.ind_peak_mins[channel, box_or_col_num] = mean_min
            
                # store the smoothed signal, peak locations, maxs, mins, and widths for each box in each channel
                self.ind_peak_props[f'Ch {channel} Box {box_or_col_num}'] = {'smoothed': signal, 
                                                        'peaks': peaks,
                                                        'proms': proms, 
                                                        'heights': heights, 
                                                        'leftIndex': leftIndex, 
                                                        'rightIndex': rightIndex}
            
            else:
                self.ind_peak_widths[submovie, channel, box_or_col_num] = mean_width
                self.ind_peak_maxs[submovie, channel, box_or_col_num] = mean_max
                self.ind_peak_mins[submovie, channel, box_or_col_num] = mean_min
        

        # make empty arrays to fill with peak measurements for each channel
        self.ind_peak_widths = np.zeros(shape=(self.num_channels, self.num_boxes_or_cols))
        self.ind_peak_maxs = np.zeros(shape=(self.num_channels, self.num_boxes_or_cols))
        self.ind_peak_mins = np.zeros(shape=(self.num_channels, self.num_boxes_or_cols))
        # make a dictionary to store the arrays and measurments generated by this function so they don't have to be re-calculated later
        self.ind_peak_props = {}

        if self.analysis_type == "standard":
            for channel in range(self.num_channels):
                for box in range(self.num_boxes_or_cols):

                    signal = sig.savgol_filter(self.means[:,channel, box], window_length = 11, polyorder = 2)
                    indv_props(signal, box)

        elif self.analysis_type == "rolling":
            self.ind_peak_widths = np.zeros(shape=(self.num_submovies, self.num_channels, self.num_boxes_or_cols))
            self.ind_peak_maxs = np.zeros(shape=(self.num_submovies, self.num_channels, self.num_boxes_or_cols))
            self.ind_peak_mins = np.zeros(shape=(self.num_submovies, self.num_channels, self.num_boxes_or_cols))

            for submovie in range(self.num_submovies):
                for channel in range(self.num_channels):
                    for box in range(self.num_boxes_or_cols):
                        signal = sig.savgol_filter(self.means[self.roll_by*submovie : self.roll_size + self.roll_by*submovie, channel, box], window_length=11, polyorder=2)
                        indv_props(signal, box, submovie)

        elif self.analysis_type == "kymograph":
            for channel in range(self.num_channels):
                for col_num in range(self.num_boxes_or_cols):

                    signal = sig.savgol_filter(self.indv_line_values[channel, col_num], window_length = 11, polyorder = 2)
                    indv_props(signal, col_num)

        self.ind_peak_amps = self.ind_peak_maxs - self.ind_peak_mins
        self.ind_peak_rel_amps = self.ind_peak_amps / self.ind_peak_mins

        return self.ind_peak_widths, self.ind_peak_maxs, self.ind_peak_mins, self.ind_peak_amps, self.ind_peak_rel_amps, self.ind_peak_props

##############################################################################################################################################################################
# Indv plotting ###########################################################################################################################################################
##############################################################################################################################################################################

    def plot_indv_acfs(self):
        '''
        Plot the raw signal and individual autocorrelation curve for each box in each channel. Annotates the first peak
        identified to estimate the period. 
        '''
        def return_figure(raw_signal: np.ndarray, acf_curve: np.ndarray, Ch_name: str, period: int):
            '''
            space saving function to generate individual plots with variable input
            '''
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.plot(raw_signal)
            ax1.set_xlabel(f'{Ch_name} Raw Signal')
            ax1.set_ylabel('Mean box px value')
            ax2.plot(np.arange(-self.num_frames_or_rows + 1, self.num_frames_or_rows), acf_curve)
            ax2.set_ylabel('Autocorrelation')
            
            if not period == np.nan:
                color = 'red'
                ax2.axvline(x = period, alpha = 0.5, c = color, linestyle = '--')
                ax2.axvline(x = -period, alpha = 0.5, c = color, linestyle = '--')
                ax2.set_xlabel(f'Period is {period} frames')
            else:
                ax2.set_xlabel(f'No period identified')

            fig.subplots_adjust(hspace=0.5)
            plt.close(fig)
            return(fig)

        # empty dictionary to fill with figures, in the event that we make more than one
        self.indv_acf_plots = {}

        its = self.num_channels*self.num_boxes_or_cols
        with tqdm(total=its, miniters=its/100) as pbar:
            pbar.set_description('ind acfs')
            for channel in range(self.num_channels):
                for box in range(self.num_boxes_or_cols):
                    pbar.update(1)
                    if self.analysis_type == "standard":
                        to_plot = self.means[:,channel, box]
                    elif self.analysis_type == "kymograph":
                        to_plot = self.indv_line_values[channel, box, :]
                    self.indv_acf_plots[f'Ch{channel + 1} Box{box + 1} ACF'] = return_figure(to_plot, 
                                                                                            self.acfs[channel, box], 
                                                                                            f'Ch{channel + 1}', 
                                                                                            self.periods[channel, box])
        return self.indv_acf_plots

    def plot_indv_ccfs(self):
        '''
        Plot the raw signals and corresponding crosscurve for each box in each unique channel combo. 
        Annotates the first peak identified to estimate the temporal shift between signals. 
        '''
        def return_figure(ch1: np.ndarray, ch2: np.ndarray, ccf_curve: np.ndarray, ch1_name: str, ch2_name: str, shift: int):
            '''
            Space saving function to generate individual plots with variable input. returns a figure object.
            '''
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.plot(ch1, color = 'tab:blue', label = ch1_name)
            ax1.plot(ch2, color = 'tab:orange', label = ch2_name)
            ax1.set_xlabel('time (frames)')
            ax1.set_ylabel('Mean box px value')
            ax1.legend(loc='upper right', fontsize = 'small', ncol = 1)
            ax2.plot(np.arange(-self.num_frames_or_rows + 1, self.num_frames_or_rows), ccf_curve)
            ax2.set_ylabel('Crosscorrelation')
            
            if not shift == np.nan:
                color = 'red'
                ax2.axvline(x = shift, alpha = 0.5, c = color, linestyle = '--')
                if shift < 1:
                    ax2.set_xlabel(f'{ch1_name} leads by {int(abs(shift))} frames')
                elif shift > 1:
                    ax2.set_xlabel(f'{ch2_name} leads by {int(abs(shift))} frames')
                else:
                    ax2.set_xlabel('no shift detected')
            else:
                ax2.set_xlabel(f'No peaks identified')
            
            fig.subplots_adjust(hspace=0.5)
            plt.close(fig)
            return(fig)
        
        def normalize(signal: np.ndarray):
            '''
            Normalize between 0 and 1
            '''
            return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
        
        # empty dictionary to fill with figures, in the event that we make more than one
        self.indv_ccf_plots = {}

        if self.num_channels > 1:
            its = len(self.channel_combos)*self.num_boxes_or_cols
            with tqdm(total=its, miniters=its/100) as pbar:
                pbar.set_description('ind ccfs')
                for combo_number, combo in enumerate(self.channel_combos):
                    for box in range(self.num_boxes_or_cols):
                        pbar.update(1)
                        if self.analysis_type == "standard":
                            Ch1 = normalize(self.means[:, combo[0], box])
                            Ch2 = normalize(self.means[:, combo[1], box])
                        elif self.analysis_type == "kymograph":
                            Ch1 = normalize(self.indv_line_values[combo[0], box, :])
                            Ch2 = normalize(self.indv_line_values[combo[1], box, :])
                        self.indv_ccf_plots[f'Ch{combo[0]}-Ch{combo[1]} Box{box + 1} CCF'] = return_figure(ch1 = Ch1,
                                                                                                        ch2 = Ch2,
                                                                                                        ccf_curve = self.indv_ccfs[combo_number, box],
                                                                                                        ch1_name = f'Ch{combo[0] + 1}',
                                                                                                        ch2_name = f'Ch{combo[1] + 1}',
                                                                                                        shift = self.indv_shifts[combo_number, box])
        
        return self.indv_ccf_plots

    def plot_indv_peak_props(self):
        '''
        Plots the individual peaks measured with annotated peak properties for each box in each channel.

        Returns:
        self.peak_figs is a dictionary object containing the plot names as keys and the figure objects as values. These can
        be easily visualized by or saved to a file using the key value as a file name.
        '''
        def return_figure(box_signal: np.ndarray, prop_dict: dict, Ch_name: str):

            smoothed_signal = prop_dict['smoothed']
            peaks = prop_dict['peaks']
            proms = prop_dict['proms']
            heights = prop_dict['heights']
            leftIndex = prop_dict['leftIndex']
            rightIndex = prop_dict['rightIndex']

            fig, ax = plt.subplots()
            ax.plot(box_signal, color = 'tab:gray', label = 'raw signal')
            ax.plot(smoothed_signal, color = 'tab:cyan', label = 'smoothed signal')

            # plot all of the peak widths and amps in a loop
            if not np.isnan(peaks).any():
                for i in range(peaks.shape[0]):
                    ax.hlines(heights[i], 
                            leftIndex[i], 
                            rightIndex[i], 
                            color='tab:olive', 
                            linestyle = '-')
                    ax.vlines(peaks[i], 
                            smoothed_signal[peaks[i]]-proms[i],
                            smoothed_signal[peaks[i]], 
                            color='tab:purple', 
                            linestyle = '-')
                # plot the first peak width and amp again so we can add it to the legend
                ax.hlines(heights[0], 
                        leftIndex[0], 
                        rightIndex[0], 
                        color='tab:olive', 
                        linestyle = '-',
                        label='FWHM')
                ax.vlines(peaks[0], 
                        smoothed_signal[peaks[0]]-proms[0],
                        smoothed_signal[peaks[0]], 
                        color='tab:purple', 
                        linestyle = '-',
                        label = 'Peak amplitude')

                ax.legend(loc='upper right', fontsize='small', ncol=1)
                ax.set_xlabel('Time (frames)')
                ax.set_ylabel('Signal (AU)')
                ax.set_title(f'{Ch_name} peak properties')
            plt.close(fig)
            return fig

        # empty dictionary to fill with figures, in the event that we make more than one
        self.indv_peak_figs = {}
        # fill the dictionary with plots for each channel
        if hasattr(self, 'ind_peak_widths'):
            its = self.num_channels*self.num_boxes_or_cols
            with tqdm(total=its, miniters=its/100) as pbar:
                pbar.set_description('ind peaks')
                for channel in range(self.num_channels):
                    for box in range(self.num_boxes_or_cols):
                        pbar.update(1)
                        if self.analysis_type == "standard":
                            to_plot = self.means[:,channel, box]
                        elif self.analysis_type == "kymograph":
                            to_plot = self.indv_line_values[channel, box, :]
                        self.indv_peak_figs[f'Ch{channel + 1} Box{box + 1} Peak Props'] = return_figure(to_plot,
                                                                                                    self.ind_peak_props[f'Ch {channel} Box {box}'],
                                                                                                    f'Ch{channel + 1} Box{box + 1}')

        return self.indv_peak_figs


##############################################################################################################################################################################
# MEAN plotting ###########################################################################################################################################################
##############################################################################################################################################################################
    
    def plot_mean_ACF(self):
        def return_figure(arr: np.ndarray, shifts_or_periods: np.ndarray, channel: str):
            fig, ax = plt.subplot_mosaic(mosaic = '''
                                                  AA
                                                  BC
                                                  ''')
            arr_mean = np.nanmean(arr, axis = 0)
            arr_std = np.nanstd(arr, axis = 0)
            x_axis = np.arange(-self.num_frames_or_rows + 1, self.num_frames_or_rows)
            ax['A'].plot(x_axis, arr_mean, color='blue')
            ax['A'].fill_between(x_axis, 
                                 arr_mean - arr_std, 
                                 arr_mean + arr_std, 
                                 color='blue', 
                                 alpha=0.2)
            ax['A'].set_title(f'{channel} Mean Autocorrelation Curve ± Standard Deviation') 
            ax['B'].hist(shifts_or_periods)
            shifts_or_periods = [val for val in shifts_or_periods if not np.isnan(val)]
            ax['B'].set_xlabel(f'Histogram of period values (frames)')
            ax['B'].set_ylabel('Occurances')
            ax['C'].boxplot(shifts_or_periods)
            ax['C'].set_xlabel(f'Boxplot of period values')
            ax['C'].set_ylabel(f'Measured period (frames)')
            fig.subplots_adjust(hspace=0.25, wspace=0.5)   
            plt.close(fig)
            return fig

        # empty dict to fill with figures, in the event that we make more than one
        self.acf_figs = {}
        
        if hasattr(self, 'acfs'):
            # make a separate plot for each channel
            for channel in range(self.num_channels):
                self.acf_figs[f'Ch{channel + 1} Mean ACF'] = return_figure(self.acfs[channel], 
                                                                         self.periods[channel], 
                                                                         f'Ch{channel + 1}')        

        return self.acf_figs
    
    # function to plot a summary of the period measurements
    def plot_mean_CCF(self):
        def return_figure(arr: np.ndarray, shifts_or_periods: np.ndarray, channel_combo: str):
            fig, ax = plt.subplot_mosaic(mosaic = '''
                                                  AA
                                                  BC
                                                  ''')
            arr_mean = np.nanmean(arr, axis = 0)
            arr_std = np.nanstd(arr, axis = 0)
            x_axis = np.arange(-self.num_frames_or_rows + 1, self.num_frames_or_rows)
            ax['A'].plot(x_axis, arr_mean, color='blue')
            ax['A'].fill_between(x_axis, 
                                 arr_mean - arr_std, 
                                 arr_mean + arr_std, 
                                 color='blue', 
                                 alpha=0.2)
            ax['A'].set_title(f'{channel_combo} Mean Crosscorrelation Curve ± Standard Deviation') 
            ax['B'].hist(shifts_or_periods)
            shifts_or_periods = [val for val in shifts_or_periods if not np.isnan(val)]
            ax['B'].set_xlabel(f'Histogram of shift values (frames)')
            ax['B'].set_ylabel('Occurances')
            ax['C'].boxplot(shifts_or_periods)
            ax['C'].set_xlabel(f'Boxplot of shift values')
            ax['C'].set_ylabel(f'Measured shift (frames)')
            fig.subplots_adjust(hspace=0.25, wspace=0.5)   
            plt.close(fig)
            return fig

        # empty dict to fill with figures, in the event that we make more than one
        self.ccf_figs = {}
               
        if hasattr(self, 'ccfs'):
            if self.num_channels > 1:
                for combo_number, combo in enumerate(self.channel_combos):
                    self.ccf_figs[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} Mean CCF'] = return_figure(self.indv_ccfs[combo_number], 
                                                                                                self.indv_shifts[combo_number], 
                                                                                                f'Ch{combo[0] + 1}-Ch{combo[1] + 1}')

        return self.ccf_figs

    def plot_mean_peak_props(self):
        def return_figure(min_array: np.ndarray, max_array: np.ndarray, amp_array: np.ndarray, width_array: np.ndarray, Ch_name: str):
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
            # filter nans out of arrays
            min_array = [val for val in min_array if not np.isnan(val)]
            max_array = [val for val in max_array if not np.isnan(val)]
            amp_array = [val for val in amp_array if not np.isnan(val)]
            width_array = [val for val in width_array if not np.isnan(val)]

            plot_params = { 'amp' : (amp_array, 'tab:blue'),
                            'min' : (min_array, 'tab:purple'),
                            'max' : (max_array, 'tab:orange')}
            for labels, (arr, arr_color) in plot_params.items():
                ax1.hist(arr, color = arr_color, label = labels, alpha = 0.75)
            boxes = ax2.boxplot([val[0] for val in plot_params.values()], patch_artist = True)
            ax2.set_xticklabels(plot_params.keys())
            for box, box_color in zip(boxes['boxes'], [val[1] for val in plot_params.values()]):
                box.set_color(box_color)

            ax1.legend(loc='upper right', fontsize = 'small', ncol = 1)
            ax1.set_xlabel(f'{Ch_name} histogram of peak values')
            ax1.set_ylabel('Occurances')

            ax2.set_xlabel(f'{Ch_name} boxplot of peak values')
            ax2.set_ylabel('Value (AU)')
            
            ax3.hist(width_array, color = 'dimgray', alpha = 0.75)
            ax3.set_xlabel(f'{Ch_name} histogram of peak widths')
            ax3.set_ylabel('Occurances')
            bp = ax4.boxplot(width_array, vert=True, patch_artist=True)
            bp['boxes'][0].set_facecolor('dimgray')
            ax4.set_xlabel(f'{Ch_name} boxplot of peak widths')
            ax4.set_ylabel('Peak width (frames)')
            fig.subplots_adjust(hspace=0.6, wspace=0.6)
            plt.close(fig)
            return fig

        # empty dictionary to fill with figures, in the event that we make more than one
        self.peak_figs = {}
        # fill the dictionary with plots for each channel
        if hasattr(self, 'peak_widths'):
            for channel in range(self.num_channels):
                self.peak_figs[f'Ch{channel + 1} Peak Props'] = return_figure(self.ind_peak_mins[channel], 
                                                                              self.ind_peak_maxs[channel], 
                                                                              self.ind_peak_amps[channel], 
                                                                              self.ind_peak_widths[channel], 
                                                                              f'Ch{channel + 1}')

        return self.peak_figs
    

    # function to summarize the results in the acf_results, ccf_results, and peak_results dictionaries as a dataframe
    def organize_measurements(self):
        '''
        Organizes the results of the ACF, CCF, and peak measurements into a dataframe. If any measurements were not
        performed, they will be excluded from the summary. Returns a dataframe with every measured parameter summarized
        by channel as well as the raw values measured for each box.

        Returns:
        self.im_measurements is a dataframe object containing the summarized results of the ACF, CCF, and peak measurements.
        '''
        
        # function to summarize measurments statistics by appending them to the beginning of the measurement list
        def add_stats(measurements: np.ndarray, measurement_name: str):
            '''
            Accepts a list of measurements. Calculates the mean, median, standard deviation, and SEM,
            and append them to the beginning of the list in that order. Finally, appends the name of
            the measurement of the beginning of the list.
            '''
            # shift measurements need special treatment to generate the correct measurements and names
            if measurement_name == 'Shift':
                statified = []
                for combo_number, combo in enumerate(self.channel_combos):
                    meas_mean = np.nanmean(measurements[combo_number])
                    meas_median = np.nanmedian(measurements[combo_number])
                    meas_std = np.nanstd(measurements[combo_number])
                    meas_sem = meas_std / np.sqrt(len(measurements[combo_number]))
                    meas_list = list(measurements[combo_number])
                    meas_list.insert(0, meas_mean)
                    meas_list.insert(1, meas_median)
                    meas_list.insert(2, meas_std)
                    meas_list.insert(3, meas_sem)
                    meas_list.insert(0, f'Ch{combo[0] + 1}-Ch{combo[1] + 1} {measurement_name}')
                    statified.append(meas_list)

            # acf and peak measurements are just iterated by channel
            else:
                statified = []
                for channel in range(self.num_channels):
                    meas_mean = np.nanmean(measurements[channel])
                    meas_median = np.nanmedian(measurements[channel])
                    meas_std = np.nanstd(measurements[channel])
                    meas_sem = meas_std / np.sqrt(len(measurements[channel]))
                    meas_list = list(measurements[channel])
                    meas_list.insert(0, meas_mean)
                    meas_list.insert(1, meas_median)
                    meas_list.insert(2, meas_std)
                    meas_list.insert(3, meas_sem)
                    meas_list.insert(0, f'Ch {channel +1} {measurement_name}')
                    statified.append(meas_list)
            return(statified)

        # column names for the dataframe summarizing the box results
        col_names = ["Parameter", "Mean", "Median", "StdDev", "SEM"]
        col_names.extend([f'Box{i}' for i in range(self.num_boxes_or_cols)])
        
        
        if self.analysis_type != "rolling":
            # combine all the statified measurements into a single list
            statified_measurements = []

            # insert Mean, Median, StdDev, and SEM into the beginning of each  list
            if hasattr(self, 'acfs'):
                self.periods_with_stats = add_stats(self.periods, 'Period')
                for channel in range(self.num_channels):
                    statified_measurements.append(self.periods_with_stats[channel])

            if hasattr(self, 'indv_ccfs'):
                self.shifts_with_stats = add_stats(self.indv_shifts, 'Shift')
                for combo_number, combo in enumerate(self.channel_combos):
                    statified_measurements.append(self.shifts_with_stats[combo_number])

            if hasattr(self, 'ind_peak_widths'):
                self.peak_widths_with_stats = add_stats(self.ind_peak_widths, 'Peak Width')
                self.peak_maxs_with_stats = add_stats(self.ind_peak_maxs, 'Peak Max')
                self.peak_mins_with_stats = add_stats(self.ind_peak_mins, 'Peak Min')
                self.peak_amps_with_stats = add_stats(self.ind_peak_amps, 'Peak Amp')
                self.peak_relamp_with_stats = add_stats(self.ind_peak_rel_amps, 'Peak Rel Amp')
                for channel in range(self.num_channels):
                    statified_measurements.append(self.peak_widths_with_stats[channel])
                    statified_measurements.append(self.peak_maxs_with_stats[channel])
                    statified_measurements.append(self.peak_mins_with_stats[channel])
                    statified_measurements.append(self.peak_amps_with_stats[channel])
                    statified_measurements.append(self.peak_relamp_with_stats[channel])

            self.im_measurements = pd.DataFrame(statified_measurements, columns = col_names)

        else: 

            self.submovie_measurements = []

            for submovie in range(self.num_submovies):
                statified_measurements = []

                if hasattr(self, 'acfs'):
                    submovie_periods_with_stats = add_stats(self.periods[submovie], 'Period')
                    for channel in range(self.num_channels):
                        statified_measurements.append(submovie_periods_with_stats[channel])
                
                if hasattr(self, 'ccfs'):
                    submovie_shifts_with_stats = add_stats(self.ccfs[submovie], 'Shift')
                    for combo_number, _ in enumerate(self.channel_combos):
                        statified_measurements.append(submovie_shifts_with_stats[combo_number])
                
                if hasattr(self, 'peak_widths'):
                    submovie_widths_with_stats = add_stats(self.peak_widths[submovie], 'Peak Width')
                    submovie_maxs_with_stats = add_stats(self.peak_maxs[submovie], 'Peak Max')
                    submovie_mins_with_stats = add_stats(self.peak_mins[submovie], 'Peak Min')
                    submovie_amps_with_stats = add_stats(self.peak_amps[submovie], 'Peak Amp')
                    submovie_rel_amps_with_stats = add_stats(self.peak_rel_amps[submovie], 'Peak Rel Amp')
                    for channel in range(self.num_channels):
                        statified_measurements.append(submovie_widths_with_stats[channel])
                        statified_measurements.append(submovie_maxs_with_stats[channel])
                        statified_measurements.append(submovie_mins_with_stats[channel])
                        statified_measurements.append(submovie_amps_with_stats[channel])
                        statified_measurements.append(submovie_rel_amps_with_stats[channel])

                submovie_meas_df = pd.DataFrame(statified_measurements, columns = col_names)
                
                self.submovie_measurements.append(submovie_meas_df)

                return self.submovie_measurements

        return self.im_measurements
    


    def summarize_image(self, file_name = None, group_name = None):
        '''
        Summarizes the results of all the measurements performed on the image.
        Returns dictionary object:
        self.file_data_summary contains the name of every summarized result for 
        each channel or channel combination as a key and the summarized results as a value.
        '''
        # dictionary to store the summarized measurements for each image
        self.file_data_summary = {}
        
        if file_name:
            self.file_data_summary['File Name'] = file_name
        if group_name:
            self.file_data_summary['Group Name'] = group_name
        self.file_data_summary['Num Boxes'] = self.num_boxes_or_cols

        stats_location = ['Mean', 'Median', 'StdDev', 'SEM']

        if hasattr(self, 'periods_with_stats'):
            pcnt_no_period = [np.count_nonzero(np.isnan(self.periods[channel])) / self.periods[channel].shape[0] * 100 for channel in range(self.num_channels)]
            for channel in range(self.num_channels):
                self.file_data_summary[f'Ch {channel + 1} Pcnt No Periods'] = pcnt_no_period[channel]
                for ind, stat in enumerate(stats_location):
                    self.file_data_summary[f'Ch {channel + 1} {stat} Period'] = self.periods_with_stats[channel][ind + 1]
        
        if hasattr(self, 'shifts_with_stats'):
            pcnt_no_shift = [np.count_nonzero(np.isnan(self.indv_shifts[combo_number])) / self.indv_shifts[combo_number].shape[0] * 100 for combo_number, combo in enumerate(self.channel_combos)]
            for combo_number, combo in enumerate(self.channel_combos):
                self.file_data_summary[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} Pcnt No Shifts'] = pcnt_no_shift[combo_number]
                for ind, stat in enumerate(stats_location):
                    self.file_data_summary[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} {stat} Shift'] = self.shifts_with_stats[combo_number][ind + 1]

        if hasattr(self, 'peak_widths_with_stats'):
            # using widths, but because these are all assigned together it applies to all peak properties
            pcnt_no_peaks = [np.count_nonzero(np.isnan(self.ind_peak_widths[channel])) / self.ind_peak_widths[channel].shape[0] * 100 for channel in range(self.num_channels)]
            for channel in range(self.num_channels):
                self.file_data_summary[f'Ch {channel + 1} Pcnt No Peaks'] = pcnt_no_peaks[channel]
                for ind, stat in enumerate(stats_location):
                    self.file_data_summary[f'Ch {channel + 1} {stat} Peak Width'] = self.peak_widths_with_stats[channel][ind + 1]
                    self.file_data_summary[f'Ch {channel + 1} {stat} Peak Max'] = self.peak_maxs_with_stats[channel][ind + 1]
                    self.file_data_summary[f'Ch {channel + 1} {stat} Peak Min'] = self.peak_mins_with_stats[channel][ind + 1]
                    self.file_data_summary[f'Ch {channel + 1} {stat} Peak Amp'] = self.peak_amps_with_stats[channel][ind + 1]
                    self.file_data_summary[f'Ch {channel + 1} {stat} Peak Rel Amp'] = self.peak_relamp_with_stats[channel][ind + 1]
            
        return self.file_data_summary


    def summarize_rolling_file(self):
        '''
        Summarizes the results of period, shift (if applicable) and peak analyses. Returns a
        SINGLE dataframe summarizing each of the relevant measurements for each submovie.

        Returns:
        self.full_movie_summary is a dataframe summarizing the results of period, shift, and peak analyses for each submovie
        '''
        all_submovie_summary = []

        stat_name_and_func = {'Mean' : np.nanmean,
                              'Median' : np.nanmedian,
                              'StdDev' : np.nanstd
                              }

        for submovie in range(self.num_submovies):
            submovie_summary = {}
            submovie_summary['Submovie'] = submovie + 1 
            if hasattr(self, 'periods'):
                for channel in range(self.num_channels):
                    pcnt_no_period = (np.count_nonzero(np.isnan(self.periods[submovie, channel])) / self.num_boxes_or_cols) * 100
                    submovie_summary[f'Ch {channel + 1} Pcnt No Periods'] = pcnt_no_period
                    for stat_name, func in stat_name_and_func.items():
                        submovie_summary[f'Ch {channel + 1} {stat_name} Period'] = func(self.periods[submovie, channel])

            if hasattr(self, 'shifts'):
                for combo_number, combo in enumerate(self.channel_combos):
                    pcnt_no_shift = np.count_nonzero(np.isnan(self.ccfs[submovie, combo_number])) / self.num_boxes_or_cols * 100
                    submovie_summary[f'Ch{combo[0]+1}-Ch{combo[1]+1} Pcnt No Shifts'] = pcnt_no_shift
                    for stat_name, func in stat_name_and_func.items():
                        submovie_summary[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} {stat_name} Shift'] = func(self.indv_shifts[submovie, combo_number])

            if hasattr(self, 'peak_widths'):
                for channel in range(self.num_channels):
                    # using widths, but because these are all assigned together it applies to all peak properties
                    pcnt_no_peaks = np.count_nonzero(np.isnan(self.ind_peak_widths[submovie, channel])) / self.num_boxes_or_cols * 100
                    submovie_summary[f'Ch {channel + 1} Pcnt No Peaks'] = pcnt_no_peaks
                    for stat_name, func in stat_name_and_func.items():
                        submovie_summary[f'Ch {channel + 1} {stat_name} Peak Width'] = func(self.ind_peak_widths[submovie, channel])
                        submovie_summary[f'Ch {channel + 1} {stat_name} Peak Max'] = func(self.ind_peak_maxs[submovie, channel])
                        submovie_summary[f'Ch {channel + 1} {stat_name} Peak Min'] = func(self.ind_peak_mins[submovie, channel])
                        submovie_summary[f'Ch {channel + 1} {stat_name} Peak Amp'] = func(self.ind_peak_amps[submovie, channel])
            all_submovie_summary.append(submovie_summary)
        
        col_names = [key for key in all_submovie_summary[0].keys()]
        self.full_movie_summary = pd.DataFrame(all_submovie_summary, columns = col_names)
                
        return self.full_movie_summary

    # function to plot the date from the self.file_data_summary dataframe
    def plot_rolling_summary(self):
        '''
        This function plots the data from the self.full_movie_summary dataframe.

        Returns:
        self.plot_list is a dictionary containing the names of the summary plots as keys and the fig object as values
        '''
        def return_plot(independent_variable, dependent_variable, dependent_error, y_label):
            '''
            This function returns plot objects to its parent function
            '''                
            fig, ax = plt.subplots()
            # plot the dataframe
            ax.plot(self.full_movie_summary[independent_variable], 
                         self.full_movie_summary[dependent_variable])
            # fill between the ± standard deviation of the dependent variable
            ax.fill_between(x = self.full_movie_summary[independent_variable],
                            y1 = self.full_movie_summary[dependent_variable] - self.full_movie_summary[dependent_error],
                            y2 = self.full_movie_summary[dependent_variable] + self.full_movie_summary[dependent_error],
                            color = 'blue',
                            alpha = 0.25)

            ax.set_xlabel('Frame Number')
            ax.set_ylabel(y_label)
            ax.set_title(f'{y_label} over time')
            plt.close(fig)
            return fig

        # empty list to fill with plots
        self.plot_list = {}
        if hasattr(self, 'periods'):
            for channel in range(self.num_channels):
                self.plot_list[f'Ch {channel + 1} Period'] = return_plot('Submovie',
                                                                          f'Ch {channel + 1} Mean Period',
                                                                          f'Ch {channel + 1} StdDev Period',
                                                                          f'Ch {channel + 1} Mean ± StdDev Period (frames)')
        if hasattr(self, 'shifts'):
            for combo_number, combo in enumerate(self.channel_combos):
                self.plot_list[f'Ch{combo[0]+1}-Ch{combo[1]+1} Shift'] = return_plot('Submovie',
                                                                                      f'Ch{combo[0]+1}-Ch{combo[1]+1} Mean Shift',
                                                                                      f'Ch{combo[0]+1}-Ch{combo[1]+1} StdDev Shift',
                                                                                      f'Ch{combo[0]+1}-Ch{combo[1]+1} Mean ± StdDev Shift (frames)')
        
        if hasattr(self, 'peak_widths'):
            for channel in range(self.num_channels):
                self.plot_list[f'Ch {channel + 1} Peak Width'] = return_plot('Submovie',
                                                                            f'Ch {channel + 1} Mean Peak Width',
                                                                            f'Ch {channel + 1} StdDev Peak Width',
                                                                            f'Ch {channel + 1} Mean ± StdDev Peak Width (frames)')
                self.plot_list[f'Ch {channel + 1} Peak Max'] = return_plot('Submovie',
                                                                            f'Ch {channel + 1} Mean Peak Max',
                                                                            f'Ch {channel + 1} StdDev Peak Max',
                                                                            f'Ch {channel + 1} Mean ± StdDev Peak Max (frames)')
                self.plot_list[f'Ch {channel + 1} Peak Min'] = return_plot('Submovie',
                                                                            f'Ch {channel + 1} Mean Peak Min',
                                                                            f'Ch {channel + 1} StdDev Peak Min',
                                                                            f'Ch {channel + 1} Mean ± StdDev Peak Min (frames)')
                self.plot_list[f'Ch {channel + 1} Peak Amp'] = return_plot('Submovie',
                                                                            f'Ch {channel + 1} Mean Peak Amp',
                                                                            f'Ch {channel + 1} StdDev Peak Amp',
                                                                            f'Ch {channel + 1} Mean ± StdDev Peak Amp (frames)')    

        return self.plot_list
    
    def save_means_to_csv(self, main_save_path, group_names, summary_df):
        """
        Save the mean values of certain metrics to separate CSV files for each group.

        Args:
            main_save_path (str): The path where the CSV files will be saved.
            group_names (list): A list of strings representing the names of the groups to be analyzed.
            summary_df (pandas DataFrame): The summary DataFrame containing the data to be analyzed.
        """
        for channel in range(self.num_channels):
            data_to_extract = [f"Ch {channel + 1} {data}" for data in ['Mean Period', 'Mean Peak Width', 'Mean Peak Max', 'Mean Peak Min', 'Mean Peak Amp', 'Mean Peak Rel Amp']]

            # Set up the output file paths
            output_file_paths = {}
            for data_name in data_to_extract:
                output_file_paths[f"{data_name}"] = f"{main_save_path}/{data_name.lower().replace(' ', '_')}_means.csv"
            
            # extract all the data (data_to_extract) from the summary df and store in a data frame
            result_df = pd.DataFrame(columns=['Data Type', 'Group Name', 'Value'])
            for data in data_to_extract:
                for group_name in group_names:
                    subset_df = summary_df.loc[summary_df['File Name'].str.contains(group_name)]
                    values = subset_df[data].tolist()
                    new_df = pd.DataFrame({'Data Type': data, 'Group Name': group_name, 'Value': values})
                    result_df = pd.concat([result_df, new_df], ignore_index=True)

            # extract, sort, and save individual tables for each data type in data_to_extract
            for data_type, output_path in output_file_paths.items():
                table = result_df[result_df['Data Type'] == data_type][['Group Name', 'Value']]
                table = pd.pivot_table(table, index=table.index, columns='Group Name', values='Value')
                for col in table.columns:
                    table[col] = sorted(table[col], key=lambda x: 1 if pd.isna(x) or x == '' else 0)
                table.to_csv(output_path, index=False)
