import os
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.signal as sig
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from itertools import zip_longest
from tifffile import imread, TiffFile

np.seterr(divide='ignore', invalid='ignore')

class TotalSignalProcessor:
    def __init__(self, analysis_type, image_path, kern=None, step=None, roll_size=None, roll_by=None, line_width=None):
        # Image import and extract common metadata
        self.analysis_type = analysis_type
        self.image_path = image_path
        self.image = imread(self.image_path)
        with TiffFile(self.image_path) as tif_file:
            metadata = tif_file.imagej_metadata
        self.num_channels = metadata.get('channels', 1)

        if analysis_type != "kymograph":
            self.kernel_size = kern
            self.step = step
            self.standardize_image_dimensions(metadata)
            self.max_project_image_stack()
            # Specific functions for rolling analysis
            if analysis_type == "rolling":
                self.roll_size = roll_size
                self.roll_by = roll_by
                self.check_and_set_rolling_parameters()
            self.calculate_box_values()
        # Specific functions for kymograph analysis
        else:
            self.line_width = line_width
            self.standardize_image_dimensions_for_kymograph()
            self.calculate_line_values()

    def standardize_image_dimensions(self, metadata):
        '''Extract more analysis-type specific metadata, and reshape the image'''
        self.num_frames = metadata.get('frames', 1)
        self.num_slices = metadata.get('slices', 1)
        self.image = self.image.reshape(self.num_frames, self.num_slices, self.num_channels, *self.image.shape[-2:])

    def max_project_image_stack(self):
        '''Max project the image if it is not already max_projected'''
        if self.num_slices > 1:
            print('Max projecting image stack')
            self.image = np.max(self.image, axis=1)
            self.num_slices = 1
            self.image = self.image.reshape(self.num_frames, self.num_slices, self.num_channels, *self.image.shape[-2:])

    def check_and_set_rolling_parameters(self):
        '''Specific parameters that are only set in the rolling analysis'''
        assert isinstance(self.roll_size, int) and isinstance(self.roll_by, int), 'Roll size and roll by must be integers'
        self.num_submovies = (self.num_frames - self.roll_size) // self.roll_by

    def calculate_box_values(self):
        '''Generate and calculate the mean signal for the specified box size over the standard and rolling images'''
        ind = self.kernel_size // 2
        self.means = nd.uniform_filter(self.image[:, 0, :, :, :], size=(1, 1, self.kernel_size, self.kernel_size))[:, :, ind::self.step, ind::self.step]
        self.xpix, self.ypix = self.means.shape[-2:]
        self.total_bins = self.xpix * self.ypix
        self.means = self.means.reshape(self.means.shape[0], self.means.shape[1], self.total_bins)

    def standardize_image_dimensions_for_kymograph(self):
        '''Reshape the kymograph image for future analysis'''
        self.image = self.image.reshape(self.num_channels, *self.image.shape[-2:])
        self.total_bins = self.image.shape[-1] # we are either binning the image into boxes (standard) or columns (kymographs), so just call bins for simplicity
        self.num_frames = self.image.shape[-2] # the number of rows in a kymograph is equal to the number to number of frames, so just call frames for simplicity
        
    def calculate_line_values(self):
        '''Generate and calculate the mean signal for the specified line width over the kymograph images'''
        self.indv_line_values = np.zeros(shape=(self.num_channels, self.total_bins, self.num_frames))
        for channel in range(self.num_channels):
            for col_num in range(self.total_bins):
                if self.line_width == 1:
                    signal = sig.savgol_filter(self.image[channel, :, col_num], window_length=25, polyorder=2)
                    self.indv_line_values[channel, col_num] = signal
                elif self.line_width % 2 != 0:
                    line_width_extra = int((self.line_width - 1) / 2)
                    if col_num + line_width_extra < self.total_bins and col_num - line_width_extra > -1:
                        signal = np.mean(self.image[channel, :, col_num - line_width_extra:col_num + line_width_extra], axis=1)
                        signal = sig.savgol_filter(signal, window_length=25, polyorder=2)
                        self.indv_line_values[channel, col_num] = signal

##############################################################################################################################################################################
# INDIVIDUAL CALCULATION #####################################################################################################################################################
##############################################################################################################################################################################

    def calc_indv_ACFs(self, peak_thresh=0.1):
        def norm_and_calc_shifts(signal, num_frames_or_rows_or_rollsize = None):
            corr_signal = signal - np.mean(signal)
            acf_curve = np.correlate(corr_signal, corr_signal, mode='full')
            # Normalize the autocorrelation curve
            acf_curve = acf_curve / (num_frames_or_rows_or_rollsize * np.std(signal) ** 2)
            # Find peaks in the autocorrelation curve
            peaks, _ = sig.find_peaks(acf_curve, prominence=peak_thresh)
            # Calculate absolute differences between peaks and center
            peaks_abs = np.abs(peaks - acf_curve.shape[0] // 2)
            # If peaks are identified, pick the closest one to the center
            if len(peaks) > 1:
                delay = np.min(peaks_abs[np.nonzero(peaks_abs)])
            else:
                # Otherwise, return NaNs for both delay and autocorrelation curve
                delay = np.nan
                acf_curve = np.full((num_frames_or_rows_or_rollsize * 2 - 1), np.nan)
            return delay, acf_curve
        
        # Initialize arrays to store period measurements and autocorrelation curves
        self.periods = np.zeros(shape=(self.num_channels, self.total_bins))
        self.acfs = np.zeros(shape=(self.num_channels, self.total_bins, self.num_frames * 2 - 1))

        # Loop through channels and bins for standard or kymograph analysis
        if self.analysis_type != "rolling":
            for channel in range(self.num_channels):
                for bin in range(self.total_bins):
                    signal = self.means[:, channel, bin] if self.analysis_type == "standard" else self.indv_line_values[channel, bin, :]
                    delay, acf_curve = norm_and_calc_shifts(signal, num_frames_or_rows_or_rollsize=self.num_frames)
                    self.periods[channel, bin] = delay
                    self.acfs[channel, bin] = acf_curve
        # If rolling analysis
        elif self.analysis_type == "rolling":
            self.periods = np.zeros(shape=(self.num_submovies, self.num_channels, self.total_bins))
            self.acfs = np.zeros(shape=(self.num_submovies, self.num_channels, self.total_bins, self.roll_size * 2 - 1))
            # Loop through submovies, channels, and bins
            for submovie in range(self.num_submovies):
                for channel in range(self.num_channels):
                    for bin in range(self.total_bins):
                        # Extract signal for rolling autocorrelation calculation
                        signal = self.means[self.roll_by * submovie: self.roll_size + self.roll_by * submovie, channel, bin]
                        delay, acf_curve = norm_and_calc_shifts(signal, num_frames_or_rows_or_rollsize=self.roll_size)
                        self.periods[submovie, channel, bin] = delay
                        self.acfs[submovie, channel, bin] = acf_curve
        return self.acfs, self.periods

    def calc_indv_CCFs(self):
        def calc_shifts(signal1, signal2, prominence=0.1, rolling = False):
            # Smoothing signals and finding peaks
            signal1 = sig.savgol_filter(signal1, window_length=11, polyorder=3)
            signal2 = sig.savgol_filter(signal2, window_length=11, polyorder=3)
            peaks1, _ = sig.find_peaks(signal1, prominence=(np.max(signal1)-np.min(signal1))*0.25)
            peaks2, _ = sig.find_peaks(signal2, prominence=(np.max(signal2)-np.min(signal2))*0.25)

            # If peaks are found in both signals
            if len(peaks1) > 0 and len(peaks2) > 0:
                corr_signal1 = signal1 - signal1.mean()
                corr_signal2 = signal2 - signal2.mean()
                # Calculate cross-correlation curve
                cc_curve = np.correlate(corr_signal1, corr_signal2, mode='full')
                if rolling:
                    cc_curve = cc_curve / (self.roll_size * signal1.std() * signal2.std())
                else:
                    cc_curve = sig.savgol_filter(cc_curve, window_length=11, polyorder=3)
                    cc_curve = cc_curve / (self.num_frames * signal1.std() * signal2.std())
                # Find peaks in the cross-correlation curve
                peaks, _ = sig.find_peaks(cc_curve, prominence=prominence)
                peaks_abs = abs(peaks - cc_curve.shape[0] // 2)
                # If multiple peaks found, select the one closest to the center
                if len(peaks) > 1:
                    delay = np.argmin(peaks_abs[np.nonzero(peaks_abs)])
                    delayIndex = peaks[delay]
                    delay_frames = delayIndex - cc_curve.shape[0] // 2
                # Otherwise, return NaNs
                else:
                    delay_frames = np.nan
                    cc_curve = np.full((self.roll_size*2-1 if rolling else self.num_frames * 2 - 1), np.nan)
            else:
                # If no peaks found, return NaNs
                delay_frames = np.nan
                cc_curve = np.full((self.roll_size*2-1 if rolling else self.num_frames * 2 - 1), np.nan)

            return delay_frames, cc_curve
        
        # Initialize arrays to store shifts and cross-correlation curves
        channels = list(range(self.num_channels))
        self.channel_combos = []
        for i in range(self.num_channels):
            for j in channels[i+1:]:
                self.channel_combos.append([channels[i],j])
        num_combos = len(self.channel_combos)

        # Initialize arrays to store shifts and cross-correlation curves
        self.indv_shifts = np.zeros(shape=(num_combos, self.total_bins))
        self.indv_ccfs = np.zeros(shape=(num_combos, self.total_bins, self.num_frames*2-1))

        # Loop through combos for standard or kymograph analysis
        if self.analysis_type != "rolling":
            for combo_number, combo in enumerate(self.channel_combos):
                for bin in range(self.total_bins):
                    if self.analysis_type == "standard":
                        signal1 = self.means[:, combo[0], bin]
                        signal2 = self.means[:, combo[1], bin]
                    elif self.analysis_type == "kymograph":
                        signal1 = self.indv_line_values[combo[0], bin]
                        signal2 = self.indv_line_values[combo[1], bin]
     
                    delay_frames, cc_curve = calc_shifts(signal1, signal2, prominence=0.1)

                    self.indv_shifts[combo_number, bin] = delay_frames
                    self.indv_ccfs[combo_number, bin] = cc_curve

        # If rolling analysis
        elif self.analysis_type == "rolling":
            # Initialize arrays to store shifts and cross-correlation curves
            self.indv_shifts = np.zeros(shape=(self.num_submovies, num_combos, self.total_bins))
            self.indv_ccfs = np.zeros(shape=(self.num_submovies, num_combos, self.total_bins, self.roll_size*2-1))

            for submovie in range(self.num_submovies):
                for combo_number, combo in enumerate(self.channel_combos):
                    for bin in range(self.total_bins):
                        signal1 = self.means[self.roll_by*submovie : self.roll_size + self.roll_by*submovie, combo[0], bin]
                        signal2 = self.means[self.roll_by*submovie : self.roll_size + self.roll_by*submovie, combo[1], bin]

                        delay_frames, cc_curve = calc_shifts(signal1, signal2, prominence=0.1, rolling = True)

                        self.indv_shifts[submovie, combo_number, bin] = delay_frames
                        self.indv_ccfs[submovie, combo_number, bin] = cc_curve

        return self.indv_shifts, self.indv_ccfs, self.channel_combos

    def calc_indv_peak_props(self):
        def indv_props(signal, bin, submovie = None):
            peaks, _ = sig.find_peaks(signal, prominence=(np.max(signal)-np.min(signal))*0.1)

            # If peaks detected, calculate properties, otherwise return NaNs
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

            # If rolling analysis
            if submovie != None:
                # Store peak measurements for each bin in each channel of a submovie
                self.ind_peak_widths[submovie, channel, bin] = mean_width
                self.ind_peak_maxs[submovie, channel, bin] = mean_max
                self.ind_peak_mins[submovie, channel, bin] = mean_min
            
            else:
                # Store peak measurements for each bin in each channel
                self.ind_peak_widths[channel, bin] = mean_width
                self.ind_peak_maxs[channel, bin] = mean_max
                self.ind_peak_mins[channel, bin] = mean_min
                self.ind_peak_props[f'Ch {channel} Bin {bin}'] = {'smoothed': signal, 
                                                        'peaks': peaks,
                                                        'proms': proms, 
                                                        'heights': heights, 
                                                        'leftIndex': leftIndex, 
                                                        'rightIndex': rightIndex}

        # Initialize arrays/dictionary to store peak measurements
        self.ind_peak_widths = np.zeros(shape=(self.num_channels, self.total_bins))
        self.ind_peak_maxs = np.zeros(shape=(self.num_channels, self.total_bins))
        self.ind_peak_mins = np.zeros(shape=(self.num_channels, self.total_bins))
        self.ind_peak_props = {}

        # Loop through channels and bins for standard or kymograph analysis
        if self.analysis_type != "rolling":
            for channel in range(self.num_channels):
                for bin in range(self.total_bins):
                    signal = sig.savgol_filter(self.means[:,channel, bin], window_length = 11, polyorder = 2) if self.analysis_type == "standard" else sig.savgol_filter(self.indv_line_values[channel, bin], window_length = 11, polyorder = 2)                        
                    indv_props(signal, bin)

        # If rolling analysis
        elif self.analysis_type == "rolling":
            self.ind_peak_widths = np.zeros(shape=(self.num_submovies, self.num_channels, self.total_bins))
            self.ind_peak_maxs = np.zeros(shape=(self.num_submovies, self.num_channels, self.total_bins))
            self.ind_peak_mins = np.zeros(shape=(self.num_submovies, self.num_channels, self.total_bins))

            for submovie in range(self.num_submovies):
                for channel in range(self.num_channels):
                    for bin in range(self.total_bins):
                        signal = sig.savgol_filter(self.means[self.roll_by*submovie : self.roll_size + self.roll_by*submovie, channel, bin], window_length=11, polyorder=2)
                        indv_props(signal, bin, submovie = submovie)

        # Calculate additional peak properties
        self.ind_peak_amps = self.ind_peak_maxs - self.ind_peak_mins
        self.ind_peak_rel_amps = self.ind_peak_amps / self.ind_peak_mins

        return self.ind_peak_widths, self.ind_peak_maxs, self.ind_peak_mins, self.ind_peak_amps, self.ind_peak_rel_amps, self.ind_peak_props

##############################################################################################################################################################################
# Indv plotting ###########################################################################################################################################################
##############################################################################################################################################################################

    def plot_indv_acfs(self):
        def return_figure(raw_signal: np.ndarray, acf_curve: np.ndarray, Ch_name: str, period: int):
            # Create subplots for raw signal and autocorrelation curve
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.plot(raw_signal)
            ax1.set_xlabel(f'{Ch_name} Raw Signal')
            ax1.set_ylabel('Mean bin px value')
            ax2.plot(np.arange(-self.num_frames + 1, self.num_frames), acf_curve)
            ax2.set_ylabel('Autocorrelation')
            
            # Annotate the first peak identified as the period if available
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

        # Empty dictionary to store generated figures
        self.indv_acf_plots = {}

        # Iterate through channels and bins to plot individual autocorrelation curves
        its = self.num_channels*self.total_bins
        with tqdm(total=its, miniters=its/100) as pbar:
            pbar.set_description('ind acfs')
            for channel in range(self.num_channels):
                for bin in range(self.total_bins):
                    pbar.update(1)
                    # Select the raw signal based on the analysis type
                    to_plot = self.means[:,channel, bin] if self.analysis_type == "standard" else self.indv_line_values[channel, bin, :]
                    # Generate and store the figure for the current channel and bin
                    self.indv_acf_plots[f'Ch{channel + 1} Bin {bin + 1} ACF'] = return_figure(to_plot, 
                                                                                            self.acfs[channel, bin], 
                                                                                            f'Ch{channel + 1}', 
                                                                                            self.periods[channel, bin])
        return self.indv_acf_plots

    def plot_indv_ccfs(self):
        # Create subplots for raw signals and cross-correlation curve
        def return_figure(ch1: np.ndarray, ch2: np.ndarray, ccf_curve: np.ndarray, ch1_name: str, ch2_name: str, shift: int):
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.plot(ch1, color = 'tab:blue', label = ch1_name)
            ax1.plot(ch2, color = 'tab:orange', label = ch2_name)
            ax1.set_xlabel('time (frames)')
            ax1.set_ylabel('Mean bin px value')
            ax1.legend(loc='upper right', fontsize = 'small', ncol = 1)
            ax2.plot(np.arange(-self.num_frames + 1, self.num_frames), ccf_curve)
            ax2.set_ylabel('Crosscorrelation')
            
            # Annotate the first peak identified as the shift if available
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
            # Normalize between 0 and 1
            return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
        
        # Empty dictionary to store generated figures
        self.indv_ccf_plots = {}

        # Iterate through channel combinations and bins to plot individual cross-correlation curves
        if self.num_channels > 1:
            its = len(self.channel_combos)*self.total_bins
            with tqdm(total=its, miniters=its/100) as pbar:
                pbar.set_description('ind ccfs')
                for combo_number, combo in enumerate(self.channel_combos):
                    for bin in range(self.total_bins):
                        pbar.update(1)
                        # Select the raw signals based on the analysis type
                        if self.analysis_type == "standard":
                            Ch1 = normalize(self.means[:, combo[0], bin])
                            Ch2 = normalize(self.means[:, combo[1], bin])
                        elif self.analysis_type == "kymograph":
                            Ch1 = normalize(self.indv_line_values[combo[0], bin, :])
                            Ch2 = normalize(self.indv_line_values[combo[1], bin, :])
                        # Generate and store the figure for the current channel combination and bin
                        self.indv_ccf_plots[f'Ch{combo[0]}-Ch{combo[1]} Bin {bin + 1} CCF'] = return_figure(ch1 = Ch1,
                                                                                                        ch2 = Ch2,
                                                                                                        ccf_curve = self.indv_ccfs[combo_number, bin],
                                                                                                        ch1_name = f'Ch{combo[0] + 1}',
                                                                                                        ch2_name = f'Ch{combo[1] + 1}',
                                                                                                        shift = self.indv_shifts[combo_number, bin])
        
        return self.indv_ccf_plots

    def save_ind_ccf_values(self, save_folder):
        def normalize(signal: np.ndarray):
            # Normalize between 0 and 1
            return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

        # Iterate through channel combinations and bins
        for combo_number, combo in enumerate(self.channel_combos):
            for bin in range(self.total_bins):
                # Normalize the signals
                if self.analysis_type == "standard":
                    ch1_normalized = normalize(self.means[:, combo[0], bin])
                    ch2_normalized = normalize(self.means[:, combo[1], bin])
                else:   
                    ch1_normalized = normalize(self.indv_line_values[combo[0], bin, :])
                    ch2_normalized = normalize(self.indv_line_values[combo[1], bin, :])
                # Retrieve the CCF curve
                ccf_curve = self.indv_ccfs[combo_number, bin]

                # Combine measurements
                measurements = list(zip_longest(range(1, len(ccf_curve) + 1), ch1_normalized, ch2_normalized, ccf_curve, fillvalue=None))
                
                # Define the filename for saving
                indv_ccfs_filename = os.path.join(save_folder, f'Bin {bin + 1}_CCF_values.csv')
            
                # Write measurements to CSV file
                with open(indv_ccfs_filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Time', 'Ch1_Value', 'Ch2_Value', 'CCF_Value'])
                    for time, ch1_val, ch2_val, ccf_val in measurements:
                        writer.writerow([time, ch1_val, ch2_val, ccf_val])

    def plot_indv_peak_props(self):
        def return_figure(bin_signal: np.ndarray, prop_dict: dict, Ch_name: str):
            # Extract peak properties from the dictionary
            smoothed_signal = prop_dict['smoothed']
            peaks = prop_dict['peaks']
            proms = prop_dict['proms']
            heights = prop_dict['heights']
            leftIndex = prop_dict['leftIndex']
            rightIndex = prop_dict['rightIndex']

            # Create the figure and plot raw and smoothed signals
            fig, ax = plt.subplots()
            ax.plot(bin_signal, color = 'tab:gray', label = 'raw signal')
            ax.plot(smoothed_signal, color = 'tab:cyan', label = 'smoothed signal')

            # Plot each peak width and amplitude
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
                # Plot the legend for the first peak
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

        # Dictionary to store generated figures
        self.indv_peak_figs = {}

        # Generate plots for each channel
        if hasattr(self, 'ind_peak_widths'):
            its = self.num_channels*self.total_bins
            with tqdm(total=its, miniters=its/100) as pbar:
                pbar.set_description('ind peaks')
                for channel in range(self.num_channels):
                    for bin in range(self.total_bins):
                        pbar.update(1)
                        # Select the signal to plot based on the analysis type
                        to_plot = self.means[:,channel, bin] if self.analysis_type == "standard" else self.indv_line_values[channel, bin, :]
                        # Generate and store the figure for the current channel and bin
                        self.indv_peak_figs[f'Ch{channel + 1} Bin {bin + 1} Peak Props'] = return_figure(to_plot,
                                                                                                    self.ind_peak_props[f'Ch {channel} Bin {bin}'],
                                                                                                    f'Ch{channel + 1} Bin {bin + 1}')

        return self.indv_peak_figs


##############################################################################################################################################################################
# MEAN plotting ###########################################################################################################################################################
##############################################################################################################################################################################
    
    def plot_mean_ACF(self):
        def return_figure(arr: np.ndarray, shifts_or_periods: np.ndarray, channel: str):
            # Plot mean autocorrelation curve with shaded area representing standard deviation
            arr_mean = np.nanmean(arr, axis = 0)
            arr_std = np.nanstd(arr, axis = 0)
            x_axis = np.arange(-self.num_frames + 1, self.num_frames)

            # Create the figure with subplots
            fig, ax = plt.subplot_mosaic(mosaic = '''
                                                  AA
                                                  BC
                                                  ''')
            
            # Plot mean autocorrelation curve with shaded area representing standard deviation
            ax['A'].plot(x_axis, arr_mean, color='blue')
            ax['A'].fill_between(x_axis, 
                                 arr_mean - arr_std, 
                                 arr_mean + arr_std, 
                                 color='blue', 
                                 alpha=0.2)
            ax['A'].set_title(f'{channel} Mean Autocorrelation Curve ± Standard Deviation') 

            # Plot histogram of period values
            ax['B'].hist(shifts_or_periods)
            shifts_or_periods = [val for val in shifts_or_periods if not np.isnan(val)]
            ax['B'].set_xlabel(f'Histogram of period values (frames)')
            ax['B'].set_ylabel('Occurances')

            # Plot boxplot of period values
            ax['C'].boxplot(shifts_or_periods)
            ax['C'].set_xlabel(f'Boxplot of period values')
            ax['C'].set_ylabel(f'Measured period (frames)')

            fig.subplots_adjust(hspace=0.25, wspace=0.5)  
            plt.close(fig)
            return fig

        # Dictionary to store generated figures
        self.acf_figs = {}
        
        if hasattr(self, 'acfs'):
            # Generate plots for each channel
            for channel in range(self.num_channels):
                self.acf_figs[f'Ch{channel + 1} Mean ACF'] = return_figure(self.acfs[channel], 
                                                                         self.periods[channel], 
                                                                         f'Ch{channel + 1}')        

        return self.acf_figs
    
    def plot_mean_CCF(self):
        def return_figure(arr: np.ndarray, shifts_or_periods: np.ndarray, channel_combo: str):
            # Plot mean cross-correlation curve with shaded area representing standard deviation
            arr_mean = np.nanmean(arr, axis = 0)
            arr_std = np.nanstd(arr, axis = 0)
            x_axis = np.arange(-self.num_frames + 1, self.num_frames)

            # Calculate mean and standard deviation of cross-correlation curves
            fig, ax = plt.subplot_mosaic(mosaic = '''
                                                  AA
                                                  BC
                                                  ''')
            
            # Plot mean cross-correlation curve with shaded area representing standard deviation
            ax['A'].plot(x_axis, arr_mean, color='blue')
            ax['A'].fill_between(x_axis, 
                                 arr_mean - arr_std, 
                                 arr_mean + arr_std, 
                                 color='blue', 
                                 alpha=0.2)
            ax['A'].set_title(f'{channel_combo} Mean Crosscorrelation Curve ± Standard Deviation') 

            # Plot histogram of period values
            ax['B'].hist(shifts_or_periods)
            shifts_or_periods = [val for val in shifts_or_periods if not np.isnan(val)]
            ax['B'].set_xlabel(f'Histogram of shift values (frames)')
            ax['B'].set_ylabel('Occurances')

            # Plot boxplot of period values
            ax['C'].boxplot(shifts_or_periods)
            ax['C'].set_xlabel(f'Boxplot of shift values')
            ax['C'].set_ylabel(f'Measured shift (frames)')

            fig.subplots_adjust(hspace=0.25, wspace=0.5)   
            plt.close(fig)
            return fig
        
        def return_mean_CCF_val(arr: np.ndarray):
            # Calculate mean and standard deviation of cross-correlation curve
            arr_mean = np.nanmean(arr, axis = 0)
            arr_std = np.nanstd(arr, axis = 0)

            # Combine mean and standard deviation into a list of tuples
            mean_CCF_values = list(zip_longest(range(1, len(arr_mean) + 1), arr_mean, arr_std, fillvalue=None))

            return mean_CCF_values

        # Dictionary to store generated figures and mean CCF values
        self.ccf_figs = {}
        self.mean_ccf_values = {}
                       
        if hasattr(self, 'indv_ccfs'):
            if self.num_channels > 1:
                # Iterate over each channel combination
                for combo_number, combo in enumerate(self.channel_combos):
                    # Generate figure for mean CCF
                    self.ccf_figs[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} Mean CCF'] = return_figure(self.indv_ccfs[combo_number], 
                                                                                                self.indv_shifts[combo_number], 
                                                                                                f'Ch{combo[0] + 1}-Ch{combo[1] + 1}')
                    # Calculate and store mean CCF values
                    self.mean_ccf_values[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} Mean CCF values.csv'] = return_mean_CCF_val(self.indv_ccfs[combo_number])

        return self.ccf_figs, self.mean_ccf_values

    def plot_mean_peak_props(self):
        def return_figure(min_array: np.ndarray, max_array: np.ndarray, amp_array: np.ndarray, width_array: np.ndarray, Ch_name: str):
            # Create subplots for histograms and boxplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

            # Filter out NaN values from arrays
            min_array = [val for val in min_array if not np.isnan(val)]
            max_array = [val for val in max_array if not np.isnan(val)]
            amp_array = [val for val in amp_array if not np.isnan(val)]
            width_array = [val for val in width_array if not np.isnan(val)]

            # Define plot parameters for histograms and boxplots
            plot_params = { 'amp' : (amp_array, 'tab:blue'),
                            'min' : (min_array, 'tab:purple'),
                            'max' : (max_array, 'tab:orange')}
            
            # Plot histograms for peak properties
            for labels, (arr, arr_color) in plot_params.items():
                ax1.hist(arr, color = arr_color, label = labels, alpha = 0.75)

            # Plot boxplots for peak properties
            boxes = ax2.boxplot([val[0] for val in plot_params.values()], patch_artist = True)
            ax2.set_xticklabels(plot_params.keys())
            for box, box_color in zip(boxes['boxes'], [val[1] for val in plot_params.values()]):
                box.set_color(box_color)

            # Set labels and legends for histograms and boxplots
            ax1.legend(loc='upper right', fontsize = 'small', ncol = 1)
            ax1.set_xlabel(f'{Ch_name} histogram of peak values')
            ax1.set_ylabel('Occurances')
            ax2.set_xlabel(f'{Ch_name} boxplot of peak values')
            ax2.set_ylabel('Value (AU)')
            
            # Plot histogram for peak widths
            ax3.hist(width_array, color = 'dimgray', alpha = 0.75)
            ax3.set_xlabel(f'{Ch_name} histogram of peak widths')
            ax3.set_ylabel('Occurances')

            # Plot boxplot for peak widths
            bp = ax4.boxplot(width_array, vert=True, patch_artist=True)
            bp['boxes'][0].set_facecolor('dimgray')
            ax4.set_xlabel(f'{Ch_name} boxplot of peak widths')
            ax4.set_ylabel('Peak width (frames)')

            fig.subplots_adjust(hspace=0.6, wspace=0.6)
            plt.close(fig)
            return fig

        # Empty dictionary to fill with figures for each channel
        self.peak_figs = {}
        if hasattr(self, 'ind_peak_widths'):
            for channel in range(self.num_channels):
                self.peak_figs[f'Ch{channel + 1} Peak Props'] = return_figure(self.ind_peak_mins[channel], 
                                                                              self.ind_peak_maxs[channel], 
                                                                              self.ind_peak_amps[channel], 
                                                                              self.ind_peak_widths[channel], 
                                                                              f'Ch{channel + 1}')

        return self.peak_figs

##############################################################################################################################################################################
# DATA ORGANIZATION ###########################################################################################################################################################
##############################################################################################################################################################################
   

    def organize_measurements(self):
        # function to summarize measurements statistics by appending them to the beginning of the measurement list
        def add_stats(measurements: np.ndarray, measurement_name: str):
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

        # column names for the dataframe summarizing the bin results
        col_names = ["Parameter", "Mean", "Median", "StdDev", "SEM"]
        col_names.extend([f'Bin {i}' for i in range(self.total_bins)])
        
        
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

            return self.im_measurements

        else: 

            self.submovie_measurements = []

            for submovie in range(self.num_submovies):
                statified_measurements = []

                if hasattr(self, 'acfs'):
                    submovie_periods_with_stats = add_stats(self.periods[submovie], 'Period')
                    for channel in range(self.num_channels):
                        statified_measurements.append(submovie_periods_with_stats[channel])
                
                if hasattr(self, 'indv_ccfs'):
                    submovie_shifts_with_stats = add_stats(self.indv_ccfs[submovie], 'Shift')
                    for combo_number, _ in enumerate(self.channel_combos):
                        statified_measurements.append(submovie_shifts_with_stats[combo_number])
                
                if hasattr(self, 'peak_widths'):
                    submovie_widths_with_stats = add_stats(self.ind_peak_widths[submovie], 'Peak Width')
                    submovie_maxs_with_stats = add_stats(self.ind_peak_maxs[submovie], 'Peak Max')
                    submovie_mins_with_stats = add_stats(self.ind_peak_mins[submovie], 'Peak Min')
                    submovie_amps_with_stats = add_stats(self.ind_peak_amps[submovie], 'Peak Amp')
                    submovie_rel_amps_with_stats = add_stats(self.ind_peak_rel_amps[submovie], 'Peak Rel Amp')
                    for channel in range(self.num_channels):
                        statified_measurements.append(submovie_widths_with_stats[channel])
                        statified_measurements.append(submovie_maxs_with_stats[channel])
                        statified_measurements.append(submovie_mins_with_stats[channel])
                        statified_measurements.append(submovie_amps_with_stats[channel])
                        statified_measurements.append(submovie_rel_amps_with_stats[channel])

                submovie_meas_df = pd.DataFrame(statified_measurements, columns = col_names)
                
                self.submovie_measurements.append(submovie_meas_df)

                return self.submovie_measurements       

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
        self.file_data_summary['Num Bins'] = self.total_bins

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
    
    # function to summarize the results in the acf_results, ccf_results, and peak_results dictionaries as a dataframe
    def get_submovie_measurements(self):
        '''
        Gathers period, shift, and peak properties measurements (if they exist), appends some simple statistics, 
        and returns a SEPARATE dataframe with raw and summarized measurements for each submovie in the dataset.
        returns:
        self.submovie_measurements is a list of dataframes, one for each submovie in the full sequence
        '''
        
        # function to summarize measurements statistics by appending them to the beginning of the measurement list
        def add_stats(measurements: np.ndarray, measurement_name: str):
            '''
            Accepts a list of measurements. Calculates the mean, median, standard deviation,
            and append them to the beginning of the list in that order. Finally, appends the name of
            the measurement of the beginning of the list.
            '''

            if measurement_name == 'Shift':
                statified = []
                for combo_number, combo in enumerate(self.channel_combos):
                    meas_mean = np.nanmean(measurements[combo_number])
                    meas_median = np.nanmedian(measurements[combo_number])
                    meas_std = np.nanstd(measurements[combo_number])
                    meas_list = list(measurements[combo_number])
                    meas_list.insert(0, meas_mean)
                    meas_list.insert(1, meas_median)
                    meas_list.insert(2, meas_std)
                    meas_list.insert(0, f'Ch{combo[0]+1}-Ch{combo[1]+1} {measurement_name}')
                    statified.append(meas_list)

            else:
                statified = []
                for channel in range(self.num_channels):
                    meas_mean = np.nanmean(measurements[channel])
                    meas_median = np.nanmedian(measurements[channel])
                    meas_std = np.nanstd(measurements[channel])
                    meas_list = list(measurements[channel])
                    meas_list.insert(0, meas_mean)
                    meas_list.insert(1, meas_median)
                    meas_list.insert(2, meas_std)
                    meas_list.insert(0, f'Ch {channel +1} {measurement_name}')
                    statified.append(meas_list)

            return(statified)

        # column names for the dataframe summarizing the box results
        col_names = ["Parameter", "Mean", "Median", "StdDev"]
        col_names.extend([f'Box{i}' for i in range(self.total_bins)])
        
        self.submovie_measurements = []

        for submovie in range(self.num_submovies):
            statified_measurements = []

            if hasattr(self, 'acfs'):
                submovie_periods_with_stats = add_stats(self.periods[submovie], 'Period')
                for channel in range(self.num_channels):
                    statified_measurements.append(submovie_periods_with_stats[channel])
            
            if hasattr(self, 'ccfs'):
                submovie_shifts_with_stats = add_stats(self.indv_ccfs[submovie], 'Shift')
                for combo_number, _ in enumerate(self.channel_combos):
                    statified_measurements.append(submovie_shifts_with_stats[combo_number])
            
            if hasattr(self, 'peak_widths'):
                submovie_widths_with_stats = add_stats(self.ind_peak_widths[submovie], 'Peak Width')
                submovie_maxs_with_stats = add_stats(self.ind_peak_maxs[submovie], 'Peak Max')
                submovie_mins_with_stats = add_stats(self.ind_peak_mins[submovie], 'Peak Min')
                submovie_amps_with_stats = add_stats(self.ind_peak_amps[submovie], 'Peak Amp')
                submovie_rel_amps_with_stats = add_stats(self.ind_peak_rel_amps[submovie], 'Peak Rel Amp')
                for channel in range(self.num_channels):
                    statified_measurements.append(submovie_widths_with_stats[channel])
                    statified_measurements.append(submovie_maxs_with_stats[channel])
                    statified_measurements.append(submovie_mins_with_stats[channel])
                    statified_measurements.append(submovie_amps_with_stats[channel])
                    statified_measurements.append(submovie_rel_amps_with_stats[channel])

            submovie_meas_df = pd.DataFrame(statified_measurements, columns = col_names)
            self.submovie_measurements.append(submovie_meas_df)

        return self.submovie_measurements

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
                    pcnt_no_period = (np.count_nonzero(np.isnan(self.periods[submovie, channel])) / self.total_bins) * 100
                    submovie_summary[f'Ch {channel + 1} Pcnt No Periods'] = pcnt_no_period
                    for stat_name, func in stat_name_and_func.items():
                        submovie_summary[f'Ch {channel + 1} {stat_name} Period'] = func(self.periods[submovie, channel])

            if hasattr(self, 'shifts'):
                for combo_number, combo in enumerate(self.channel_combos):
                    pcnt_no_shift = np.count_nonzero(np.isnan(self.indv_ccfs[submovie, combo_number])) / self.total_bins * 100
                    submovie_summary[f'Ch{combo[0]+1}-Ch{combo[1]+1} Pcnt No Shifts'] = pcnt_no_shift
                    for stat_name, func in stat_name_and_func.items():
                        submovie_summary[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} {stat_name} Shift'] = func(self.indv_shifts[submovie, combo_number])

            if hasattr(self, 'peak_widths'):
                for channel in range(self.num_channels):
                    # using widths, but because these are all assigned together it applies to all peak properties
                    pcnt_no_peaks = np.count_nonzero(np.isnan(self.ind_peak_widths[submovie, channel])) / self.total_bins * 100
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
            # Define data metrics to extract
            metrics_to_extract = [f"Ch {channel + 1} {data}" for data in ['Mean Period', 'Mean Peak Width', 'Mean Peak Max', 'Mean Peak Min', 'Mean Peak Amp', 'Mean Peak Rel Amp']]
            
            # Create folder for storing results
            folder_path = os.path.join(main_save_path, "!channel_mean_measurements")
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Extract data for each group and metric
            result_df = pd.DataFrame(columns=['Data Type', 'Group Name', 'Value'])
            for metric in metrics_to_extract:
                for group_name in group_names:
                    group_data = summary_df.loc[summary_df['File Name'].str.contains(group_name)]
                    values = group_data[metric].tolist()
                    result_df = pd.concat([result_df, pd.DataFrame({'Data Type': metric, 'Group Name': group_name, 'Value': values})], ignore_index=True)

            # Save individual tables for each metric
            for metric in metrics_to_extract:
                # Define output path for CSV
                output_path = os.path.join(folder_path, f"{metric.lower().replace(' ', '_')}_means.csv")

                # Prepare and sort table 
                metric_table = result_df[result_df['Data Type'] == metric][['Group Name', 'Value']]
                metric_table = pd.pivot_table(metric_table, index=metric_table.index, columns='Group Name', values='Value')
                for col in metric_table.columns:
                    metric_table[col] = sorted(metric_table[col], key=lambda x: 1 if pd.isna(x) or x == '' else 0)
                
                # Save table to CSV
                metric_table.to_csv(output_path, index=False)

