import pandas as pd
import matplotlib.pyplot as plt
from waveanalysis.housekeeping.housekeeping_functions import relabel_channels, relabel_metric_text

def plot_rolling_summary(
    num_channels: int,
    fullmovie_summary: pd.DataFrame,
    channel_combos: list[tuple[int, int]],
    dark_plots: bool = False,
    channel_names: list = None,
    live_injection_dict: dict = {}
):
    '''
    Generate rolling summary plots for wave analysis.

    Parameters:
    - num_channels (int): The number of channels.
    - fullmovie_summary (pd.DataFrame): The summary data for the full movie.
    - channel_combos (list[tuple[int, int]]): The combinations of channels.
    - dark_plots (bool): Inidicates whether the plots should be dark.
    - live_injection_dict (dict): Stores info relevant for plots used when imaging during injections.

    Returns:
    - rolling_mean_plots_dict (dict): A dictionary containing the rolling mean plots.
    '''
    # Initialize the dictionary to store the rolling mean plots
    rolling_mean_plots_dict = {}
    rolling_mean_periods = {}
    rolling_mean_shifts = {}
    rolling_mean_peak_props = {}

    # Generate the rolling mean plots for the mean period
    for channel in range(num_channels):
        rolling_mean_periods[f'Ch{channel + 1} Period'] = _return_mean_periods_shifts_props_plots(
            independent_variable='Submovie',
            dependent_variable=f'Ch {channel + 1} Mean Period',
            dependent_error=f'Ch {channel + 1} StdDev Period',
            y_label=relabel_channels(f'Ch {channel + 1}: mean period ± SD (seconds)', channel_names),
            fullmovie_summary=fullmovie_summary,
            dark_plots=dark_plots,
            injection_ch1=live_injection_dict["injection_ch1"],
            injection_ch2=live_injection_dict["injection_ch2"],
            injection_frame=live_injection_dict["injection_frame"]
            )
            
    # Update the dictionary with the rolling mean plots for the mean period
    rolling_mean_plots_dict.update(rolling_mean_periods)

    # Generate the rolling mean plots for the mean shifts
    if num_channels > 1:
        for combo_number, combo in enumerate(channel_combos):
            rolling_mean_shifts[f'Ch{combo[0]+1}-Ch{combo[1]+1} Shift'] = _return_mean_periods_shifts_props_plots(
                independent_variable='Submovie',
                dependent_variable=f'Ch{combo[0]+1}-Ch{combo[1]+1} Mean Shift',
                dependent_error=f'Ch{combo[0]+1}-Ch{combo[1]+1} StdDev Shift',
                y_label=relabel_channels(f'Ch{combo[0]+1}-Ch{combo[1]+1}: mean CCF shift ± SD (seconds)', channel_names),
                fullmovie_summary=fullmovie_summary,
                dark_plots=dark_plots,
                injection_ch1=live_injection_dict["injection_ch1"],
                injection_ch2=live_injection_dict["injection_ch2"],
                injection_frame=live_injection_dict["injection_frame"]
                )
            
    # Update the dictionary with the rolling mean plots for the mean shifts
    rolling_mean_plots_dict.update(rolling_mean_shifts)

    # Generate the rolling mean plots for the peak properties
    for channel in range(num_channels):
        for prop_name in ['Width', 'Max', 'Min', 'Amp', 'Rel Amp', 'Offset', 'Area']:
            rolling_mean_peak_props[f'Ch{channel+1} {prop_name}'] = _return_mean_periods_shifts_props_plots(
                independent_variable='Submovie',
                dependent_variable=f'Ch {channel+1} Mean Peak {prop_name}',
                dependent_error=f'Ch {channel+1} StdDev Peak {prop_name}',
                y_label=relabel_metric_text(f'Ch {channel+1}: mean ± SD Peak {prop_name}', channel_names),
                fullmovie_summary=fullmovie_summary,
                dark_plots=dark_plots,
                injection_ch1=live_injection_dict["injection_ch1"],
                injection_ch2=live_injection_dict["injection_ch2"],
                injection_frame=live_injection_dict["injection_frame"]
                )
                    
    # Update the dictionary with the rolling mean plots for the peak properties
    rolling_mean_plots_dict.update(rolling_mean_peak_props)

    return rolling_mean_plots_dict

def _return_mean_periods_shifts_props_plots(
    independent_variable: str, 
    dependent_variable: str, 
    dependent_error: str, 
    y_label: str,
    fullmovie_summary: pd.DataFrame,
    dark_plots: bool = False,
    injection_ch1: bool = False,
    injection_ch2: bool = False,
    injection_frame: int = None
) -> plt.Figure:    
    '''
    Space saving function to generate the rolling summary plots
    '''      
    style = 'dark_background' if dark_plots else 'default'
    with plt.style.context(style):
        fig, ax = plt.subplots()

        # plot the dataframe
        ax.plot(fullmovie_summary[independent_variable], 
                fullmovie_summary[dependent_variable],
                color = 'blue' if not dark_plots else 'lightblue')
        
        # fill between the ± standard deviation of the dependent variable
        ax.fill_between(x = fullmovie_summary[independent_variable],
                        y1 = fullmovie_summary[dependent_variable] - fullmovie_summary[dependent_error],
                        y2 = fullmovie_summary[dependent_variable] + fullmovie_summary[dependent_error],
                        color = 'blue' if not dark_plots else 'lightblue',
                        alpha = 0.25)
        
        # plot average signal of injection channel over all the frames
        if (injection_ch1 != injection_ch2):
            injection_channel = 1 if injection_ch1 else 2
            overlay_signal = fullmovie_summary[f'Ch {injection_channel} Mean Signal']
            ymin = (fullmovie_summary[dependent_variable] - fullmovie_summary[dependent_error]).min()

            ymax = (fullmovie_summary[dependent_variable] + fullmovie_summary[dependent_error]).max()

            signal_scaled = ((overlay_signal - overlay_signal.min()) / (overlay_signal.max() - overlay_signal.min()))

            signal_scaled = signal_scaled * (ymax - ymin) + ymin

            ax.plot(
                fullmovie_summary[independent_variable],
                signal_scaled,
                color='orange' if not dark_plots else 'yellow',
                linewidth=2,
                alpha=0.8
            )
        
        # plot vertical line indicating when injection takes place
        if (injection_frame is not None and injection_frame != 0):
            ax.axvline(x=injection_frame, color='red', linestyle='--', alpha=0.5)
        

        # set axis labels
        ax.set_xlabel('Rolling submovie index')
        ax.set_ylabel(y_label)
        ax.set_title(f'{y_label} over rolling windows')
        plt.close(fig)

    return fig
