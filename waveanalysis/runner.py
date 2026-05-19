import waveanalysis.housekeeping.housekeeping_functions as hf
from waveanalysis.custom_gui import BaseGUI, RollingGUI, KymographGUI
from waveanalysis.data_workflows.combined_workflow import combined_workflow
from waveanalysis.data_workflows.rolling_workflow import rolling_workflow

def main():
    '''
    Main function to run the wave analysis GUI and analysis workflows.
    '''
    # Show BaseGUI first to determine which analysis mode the user wants
    base_gui = BaseGUI()
    base_gui.mainloop()

    if base_gui.rolling:
        gui = RollingGUI()
        gui.mainloop()
    elif base_gui.kymograph:
        gui = KymographGUI()
        gui.mainloop()
    else:
        gui = base_gui

    params = gui.vars
    analysis_type = params["analysis_type"]

    # Build log_params from the shared base, then add mode-specific keys
    log_params = {
        "Base Directory": params["folder_path"],
        "ACF Peak Prominence": params["acf_peak_thresh"],
        "CCF Peak Prominence": params["ccf_peak_thresh"],
        "Small Shifts Correction": params["small_shifts_correction"],
        "Smoothing": params["smoothing"],
        "Smoothing Params": params["smoothing_params"],
        "Files Processed": [],
        "Files Not Processed": [],
        "Errors": [],
        "Frame Interval": [],
        "Pixel Size": [],
    }

    if analysis_type == "standard":
        log_params.update({
            "Box Size(px)": params["box_size"],
            "Box Shift(px)": params["bin_shift"],
            "Group Names": params["group_names"],
            "Plot Summary ACFs": params["plot_flags"]["plot_summary_ACFs"],
            "Plot Summary CCFs": params["plot_flags"]["plot_summary_CCFs"],
            "Plot Summary Peaks": params["plot_flags"]["plot_summary_peaks"],
            "Plot Individual ACFs": params["plot_flags"]["plot_indv_ACFs"],
            "Plot Individual CCFs": params["plot_flags"]["plot_indv_CCFs"],
            "Plot Individual Peaks": params["plot_flags"]["plot_indv_peaks"],
            "Plot Heatmaps": params["plot_flags"]["plot_heatmaps"],
            "Plot FTs": params["plot_flags"]["plot_fts"],
            "Dark Plots": params["plot_flags"]["dark_plots"],
        })
    elif analysis_type == "rolling":
        log_params.update({
            "Box Size(px)": params["box_size"],
            "Box Shift(px)": params["bin_shift"],
            "Plot sub-movie ACFs": params["plot_subframe_ACFs"],
            "Plot movie CCFs": params["plot_subframe_CCFs"],
            "Plot movie Peaks": params["plot_subframe_peaks"],
            "Submovies Used": [],
            "Plotting errors": [],
        })
    elif analysis_type == "kymograph":
        log_params.update({
            "Line width": params["line_width"],
            "Line Shift(px)": params["bin_shift"],
            "Group Names": params["group_names"],
            "Plot Summary ACFs": params["plot_flags"]["plot_summary_ACFs"],
            "Plot Summary CCFs": params["plot_flags"]["plot_summary_CCFs"],
            "Plot Summary Peaks": params["plot_flags"]["plot_summary_peaks"],
            "Plot Individual ACFs": params["plot_flags"]["plot_indv_ACFs"],
            "Plot Individual CCFs": params["plot_flags"]["plot_indv_CCFs"],
            "Plot Individual Peaks": params["plot_flags"]["plot_indv_peaks"],
            "Plot Heatmaps": params["plot_flags"]["plot_heatmaps"],
            "Dark Plots": params["plot_flags"]["dark_plots"],
            "Calc Wave Speeds": params["calculate_wave_speeds"],
            "Plot Wave Speeds": True,
        })

    # Validate inputs
    hf.threshold_check(params["acf_peak_thresh"], log_params)
    if len(params["folder_path"]) < 1:
        log_params["Errors"].append("You didn't enter a directory to analyze")

    # Run the selected workflow
    if analysis_type == "standard":
        combined_workflow(
            folder_path=params["folder_path"],
            group_names=params["group_names"],
            log_params=log_params,
            analysis_type=analysis_type,
            acf_peak_thresh=params["acf_peak_thresh"],
            ccf_peak_thresh=params["ccf_peak_thresh"],
            small_shifts_correction=params["small_shifts_correction"],
            plot_flags=params["plot_flags"],
            peak_prominence_fraction=params["peak_prominence_fraction"],
            calc_wave_speeds=False,
            plot_wave_speeds=False,
            box_size=params["box_size"],
            bin_shift=params["bin_shift"],
            line_width=None,
            test=False,
            smoothing_params=params["smoothing_params"],
            smoothing=params["smoothing"],
        )

    elif analysis_type == "rolling":
        rolling_workflow(
            folder_path=params["folder_path"],
            log_params=log_params,
            box_size=params["box_size"],
            bin_shift=params["bin_shift"],
            subframe_size=params["subframe_size"],
            subframe_roll=params["subframe_roll"],
            acf_peak_thresh=params["acf_peak_thresh"],
            ccf_peak_thresh=params["ccf_peak_thresh"],
            small_shifts_correction=params["small_shifts_correction"],
            peak_prominence_fraction=params["peak_prominence_fraction"],
            test=False,
            smoothing_params=params["smoothing_params"],
            smoothing=params["smoothing"],
            dark_plots=params["dark_plots"],
        )

    elif analysis_type == "kymograph":
        combined_workflow(
            folder_path=params["folder_path"],
            group_names=params["group_names"],
            log_params=log_params,
            analysis_type=analysis_type,
            acf_peak_thresh=params["acf_peak_thresh"],
            ccf_peak_thresh=params["ccf_peak_thresh"],
            small_shifts_correction=params["small_shifts_correction"],
            plot_flags=params["plot_flags"],
            peak_prominence_fraction=params["peak_prominence_fraction"],
            calc_wave_speeds=params["calculate_wave_speeds"],
            plot_wave_speeds=True,
            box_size=None,
            bin_shift=params["bin_shift"],
            line_width=params["line_width"],
            test=False,
            smoothing_params=params["smoothing_params"],
            smoothing=params["smoothing"]
            )

if __name__ == "__main__":
    main()
