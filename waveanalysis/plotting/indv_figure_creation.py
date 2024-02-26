import numpy as np
import matplotlib.pyplot as plt

def return_indv_peak_prop_figure(
        bin_signal: np.ndarray, 
        prop_dict: dict, 
        Ch_name: str):
    '''
    Space saving function to generate the plots for the individual peak prop plots
    '''
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

def return_indv_acf_figure(
        raw_signal: np.ndarray, 
        acf_curve: np.ndarray, 
        Ch_name: str, 
        period: int,
        num_frames: int):
        '''
        Space saving function to generate the plots for the individual ACF plots
        '''
        # Create subplots for raw signal and autocorrelation curve
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(raw_signal)
        ax1.set_xlabel(f'{Ch_name} Raw Signal')
        ax1.set_ylabel('Mean bin px value')
        ax2.plot(np.arange(-num_frames + 1, num_frames), acf_curve)
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
