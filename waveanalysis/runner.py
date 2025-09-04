import waveanalysis.housekeeping.housekeeping_functions as hf
from waveanalysis.custom_gui import BaseGUI, RollingGUI, KymographGUI
from waveanalysis.data_workflows.combined_workflow import combined_workflow
from waveanalysis.data_workflows.rolling_workflow import rolling_workflow

def main():
    '''
    Main function to run the wave analysis GUI and analysis workflows.
    '''
    # make GUI object and display the window
    gui = BaseGUI()
    gui.mainloop()

    params = gui.vars

    analysis_type = params["analysis_type"]
    box_size = params["box_size"]
    bin_shift = params["bin_shift"]
    folder_path = params["folder_path"]
    group_names = params["group_names"]
    acf_peak_thresh = params["acf_peak_thresh"]
    small_shifts_correction = params["small_shifts_correction"]
    plot_summary_ACFs = params["plot_summary_ACFs"]
    plot_summary_CCFs = params["plot_summary_CCFs"]
    plot_summary_peaks = params["plot_summary_peaks"]
    plot_indv_ACFs = params["plot_indv_ACFs"]
    plot_indv_CCFs = params["plot_indv_CCFs"]
    plot_indv_peaks = params["plot_indv_peaks"]
    ccf_peak_thresh = params["ccf_peak_thresh"]

    smoothing = params["smoothing"]
    Ch1_window = params.get("Ch1_window")
    Ch1_poly_order = params.get("Ch1_poly_order")
    Ch2_window = params.get("Ch2_window")
    Ch2_poly_order = params.get("Ch2_poly_order")
    Ch3_window = params.get("Ch3_window")
    Ch3_poly_order = params.get("Ch3_poly_order")
    Ch4_window = params.get("Ch4_window")
    Ch4_poly_order = params.get("Ch4_poly_order")
    CCF_window = params.get("CCF_window")
    CCF_poly_order = params.get("CCF_poly_order")

    # if rolling GUI specified, make rolling GUI object and display the window
    if gui.rolling:
        # make GUI object and display the window
        gui = RollingGUI()
        gui.mainloop()
        
        params = gui.vars

        analysis_type = params["analysis_type"]
        box_size = params["box_size"]
        bin_shift = params["bin_shift"]
        folder_path = params["folder_path"]
        acf_peak_thresh = params["acf_peak_thresh"]
        small_shifts_correction = params["small_shifts_correction"]
        ccf_peak_thresh = params["ccf_peak_thresh"]
        plot_sf_ACFs = params["plot_subframe_ACFs"]
        plot_sf_CCFs = params["plot_subframe_CCFs"]
        plot_sf_peaks = params["plot_subframe_peaks"]
        subframe_size = params["subframe_size"]
        subframe_roll = params["subframe_roll"]
        
        smoothing = params["smoothing"]
        Ch1_window = params.get("Ch1_window")
        Ch1_poly_order = params.get("Ch1_poly_order")
        Ch2_window = params.get("Ch2_window")
        Ch2_poly_order = params.get("Ch2_poly_order")
        Ch3_window = params.get("Ch3_window")
        Ch3_poly_order = params.get("Ch3_poly_order")
        Ch4_window = params.get("Ch4_window")
        Ch4_poly_order = params.get("Ch4_poly_order")
        CCF_window = params.get("CCF_window")
        CCF_poly_order = params.get("CCF_poly_order")

    # if kymograph GUI specified, make kymograph GUI object and display the window
    if gui.kymograph:
        # make GUI object and display the window
        gui = KymographGUI()
        gui.mainloop()
        
        params = gui.vars

        analysis_type = params["analysis_type"]
        line_width = params["line_width"]
        bin_shift = params["bin_shift"]
        folder_path = params["folder_path"]
        group_names = params["group_names"]
        acf_peak_thresh = params["acf_peak_thresh"]
        small_shifts_correction = params["small_shifts_correction"]
        plot_summary_ACFs = params["plot_summary_ACFs"]
        plot_summary_CCFs = params["plot_summary_CCFs"]
        plot_summary_peaks = params["plot_summary_peaks"]
        plot_indv_ACFs = params["plot_indv_ACFs"]
        plot_indv_CCFs = params["plot_indv_CCFs"]
        plot_indv_peaks = params["plot_indv_peaks"]
        ccf_peak_thresh = params["ccf_peak_thresh"]
        calc_wave_speeds = params["calculate_wave_speeds"]

        smoothing = params["smoothing"]
        Ch1_window = params.get("Ch1_window")
        Ch1_poly_order = params.get("Ch1_poly_order")
        Ch2_window = params.get("Ch2_window")
        Ch2_poly_order = params.get("Ch2_poly_order")
        Ch3_window = params.get("Ch3_window")
        Ch3_poly_order = params.get("Ch3_poly_order")
        Ch4_window = params.get("Ch4_window")
        Ch4_poly_order = params.get("Ch4_poly_order")
        CCF_window = params.get("CCF_window")
        CCF_poly_order = params.get("CCF_poly_order")

    #make dictionary of parameters for log file use
    log_params = {  
        "Box Size(px)": box_size,
        "Box Shift(px)": bin_shift,
        "Base Directory": folder_path,
        "ACF Peak Prominence": acf_peak_thresh,
        "CCF Peak Prominence": ccf_peak_thresh,
        "Small Shifts Correction": small_shifts_correction,
        "Group Names": group_names,
        "Plot Summary ACFs": plot_summary_ACFs,
        "Plot Summary CCFs": plot_summary_CCFs,
        "Plot Summary Peaks": plot_summary_peaks,
        "Plot Individual ACFs": plot_indv_ACFs,
        "Plot Individual CCFs": plot_indv_CCFs,
        "Plot Individual Peaks": plot_indv_peaks,
        "Ch1 Window": Ch1_window,
        "Ch1 Poly Order": Ch1_poly_order,
        "Ch2 Window": Ch2_window,
        "Ch2 Poly Order": Ch2_poly_order,
        "Ch3 Window": Ch3_window,
        "Ch3 Poly Order": Ch3_poly_order,
        "Ch4 Window": Ch4_window,
        "Ch4 Poly Order": Ch4_poly_order,
        "CCF Window": CCF_window,
        "CCF Poly Order": CCF_poly_order,
        "Smoothing": smoothing,
        "Files Processed": [],
        "Files Not Processed": [],
        "Errors": [],
        'Frame Interval': [],
        'Pixel Size': [],
    }
        
    if analysis_type == 'rolling':
        log_params = {  "Box Size(px)" : box_size,
                        "Box Shift(px)" : bin_shift,
                        "Base Directory" : folder_path,
                        "ACF Peak Prominence" : acf_peak_thresh,
                        "CCF Peak Prominence" : ccf_peak_thresh,
                        "Small Shifts Correction" : small_shifts_correction,
                        "Plot sub-movie ACFs" : plot_sf_ACFs,
                        "Plot movie CCFs" : plot_sf_CCFs,
                        "Plot movie Peaks" : plot_sf_peaks,
                        "Ch1 Window": Ch1_window,
                        "Ch1 Poly Order": Ch1_poly_order,
                        "Ch2 Window": Ch2_window,
                        "Ch2 Poly Order": Ch2_poly_order,
                        "Ch3 Window": Ch3_window,
                        "Ch3 Poly Order": Ch3_poly_order,
                        "Ch4 Window": Ch4_window,
                        "Ch4 Poly Order": Ch4_poly_order,
                        "CCF Window": CCF_window,
                        "CCF Poly Order": CCF_poly_order,
                        "Smoothing": smoothing,
                        'Files Processed': [],
                        'Files Not Processed': [],
                        'Plotting errors': [],
                        'Submovies Used' : [],
                        'Errors': [],
                        'Frame Interval': [],
                        'Pixel Size': []
                } 
    if analysis_type == 'kymograph':
        log_params = {  "Line width": line_width,
                        "Line Shift(px)": bin_shift,
                        "Base Directory": folder_path,
                        "ACF Peak Prominence" : acf_peak_thresh,
                        "CCF Peak Prominence" : ccf_peak_thresh,
                        "Small Shifts Correction" : small_shifts_correction,
                        "Group Names" : group_names,
                        "Plot Summary ACFs": plot_summary_ACFs,
                        "Plot Summary CCFs": plot_summary_CCFs,
                        "Plot Summary Peaks": plot_summary_peaks,
                        "Plot Individual ACFs": plot_indv_ACFs,
                        "Plot Individual CCFs": plot_indv_CCFs,
                        "Plot Individual Peaks": plot_indv_peaks,  
                        'Calc Wave Speeds': calc_wave_speeds,
                        'Plot Wave Speeds': True,
                        "Ch1 Window": Ch1_window,
                        "Ch1 Poly Order": Ch1_poly_order,
                        "Ch2 Window": Ch2_window,
                        "Ch2 Poly Order": Ch2_poly_order,
                        "Ch3 Window": Ch3_window,
                        "Ch3 Poly Order": Ch3_poly_order,
                        "Ch4 Window": Ch4_window,
                        "Ch4 Poly Order": Ch4_poly_order,
                        "CCF Window": CCF_window,
                        "CCF Poly Order": CCF_poly_order,
                        "Smoothing": smoothing,
                        "Files Processed": [],
                        "Files Not Processed": [],
                        "Errors" : [],
                        'Frame Interval': [],
                        'Pixel Size': [],
                }
        
    # identify and report errors in GUI input
    hf.threshold_check(acf_peak_thresh, log_params)
    
    # check if a directory was entered
    if len(folder_path) < 1 :        
        log_params["Errors"].append("You didn't enter a directory to analyze")        
        
    # Run the analysis based on the GUI input
    if analysis_type == "standard":                         
        combined_workflow(
            folder_path=folder_path,
            group_names=group_names,
            log_params=log_params,
            analysis_type=analysis_type,
            acf_peak_thresh=acf_peak_thresh,
            ccf_peak_thresh=ccf_peak_thresh,
            small_shifts_correction=small_shifts_correction,
            plot_summary_ACFs=plot_summary_ACFs,
            plot_summary_CCFs=plot_summary_CCFs,
            plot_summary_peaks=plot_summary_peaks,
            plot_indv_ACFs=plot_indv_ACFs,
            plot_indv_CCFs=plot_indv_CCFs,
            calc_wave_speeds=False,
            plot_wave_speeds=False,
            plot_indv_peaks=plot_indv_peaks,
            box_size=box_size,
            bin_shift=bin_shift,
            line_width=None,
            test=False,
            Ch1_window=Ch1_window,
            Ch1_poly_order=Ch1_poly_order,
            Ch2_window=Ch2_window,
            Ch2_poly_order=Ch2_poly_order,
            Ch3_window=Ch3_window,
            Ch3_poly_order=Ch3_poly_order,
            Ch4_window=Ch4_window,
            Ch4_poly_order=Ch4_poly_order,
            CCF_window=CCF_window,
            CCF_poly_order=CCF_poly_order,
            smoothing=smoothing
        )
    
    if analysis_type == "rolling":
        rolling_workflow(
            folder_path=folder_path,
            log_params=log_params,
            box_size=box_size,
            box_shift=bin_shift,
            roll_size=subframe_size,    
            roll_by=subframe_roll,
            acf_peak_thresh=acf_peak_thresh,
            ccf_peak_thresh=ccf_peak_thresh,
            small_shifts_correction=small_shifts_correction,
            test=False,
            Ch1_window=Ch1_window,
            Ch1_poly_order=Ch1_poly_order,
            Ch2_window=Ch2_window,
            Ch2_poly_order=Ch2_poly_order,
            Ch3_window=Ch3_window,
            Ch3_poly_order=Ch3_poly_order,
            Ch4_window=Ch4_window,
            Ch4_poly_order=Ch4_poly_order,
            CCF_window=CCF_window,
            CCF_poly_order=CCF_poly_order,
            smoothing=smoothing
        )

    if analysis_type == "kymograph":                         
        combined_workflow(
            folder_path=folder_path,
            group_names=group_names,
            log_params=log_params,
            analysis_type=analysis_type,
            acf_peak_thresh=acf_peak_thresh,
            ccf_peak_thresh=ccf_peak_thresh,
            small_shifts_correction=small_shifts_correction,
            plot_summary_ACFs=plot_summary_ACFs,
            plot_summary_CCFs=plot_summary_CCFs,
            plot_summary_peaks=plot_summary_peaks,
            plot_indv_ACFs=plot_indv_ACFs,
            plot_indv_CCFs=plot_indv_CCFs,
            plot_indv_peaks=plot_indv_peaks,
            calc_wave_speeds=calc_wave_speeds,
            plot_wave_speeds=True, # always plot wave speeds for now
            box_size=None,
            bin_shift=bin_shift,
            line_width=line_width,
            test=False,
            Ch1_window=Ch1_window,
            Ch1_poly_order=Ch1_poly_order,
            Ch2_window=Ch2_window,
            Ch2_poly_order=Ch2_poly_order,
            Ch3_window=Ch3_window,
            Ch3_poly_order=Ch3_poly_order,
            Ch4_window=Ch4_window,
            Ch4_poly_order=Ch4_poly_order,
            CCF_window=CCF_window,
            CCF_poly_order=CCF_poly_order,
            smoothing=smoothing
        )

if __name__ == "__main__":
    main()