import os
import re
import sys
import threading

os.environ["COLUMNS"] = "80"

import matplotlib
matplotlib.use("Agg")

import waveanalysis.housekeeping.housekeeping_functions as hf
from waveanalysis.custom_gui import BaseGUI, RollingGUI, KymographGUI
from waveanalysis.data_workflows.combined_workflow import combined_workflow
from waveanalysis.data_workflows.rolling_workflow import rolling_workflow

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_TQDM_RE = re.compile(r"\d+%\|")


class _StopAnalysis(Exception):
    """Raised when the user clicks Stop."""
    pass


class _GUIWriter:
    """Redirects stdout/stderr writes to a GUI's log_message method.
    tqdm progress bars update in-place; other output is appended."""
    def __init__(self, gui):
        self.gui = gui
        self._total_files = 0
        self._files_done = 0

    def set_file_tracking(self, total):
        self._total_files = total
        self._files_done = 0

    def write(self, msg):
        if not msg:
            return
        cleaned = _ANSI_RE.sub("", msg)
        if "\r" in cleaned:
            cleaned = cleaned.rsplit("\r", 1)[-1]
        cleaned = cleaned.strip()
        if not cleaned:
            return
        for line in cleaned.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("cleanup:"):
                continue

            tqdm_match = _TQDM_RE.search(line)
            if tqdm_match:
                label = line[:tqdm_match.start()].strip() or "progress"
                self.gui.update_progress_line(label, line)
                continue

            if (line.startswith("Processing ")
                    and line.endswith("...")
                    and ".tif" in line):
                self._files_done += 1
                filename = line[len("Processing "):-3]
                self.gui.log_message(
                    f"[{self._files_done}/{self._total_files}] {filename}"
                )
                self.gui.set_status(
                    f"Processing file {self._files_done}/{self._total_files}: {filename}"
                )
                self.gui.update_file_progress(self._files_done, self._total_files)
                if self.gui._stop_requested:
                    self.gui.log_message("Analysis stopped by user.")
                    self.gui.set_status("Stopped")
                    raise _StopAnalysis()
                continue

            self.gui.log_message(line)

    def flush(self):
        pass


def _build_log_params(params, analysis_type):
    log_params = {
        "Base Directory": params["folder_path"],
        "ACF Peak Prominence": params["acf_peak_thresh"],
        "CCF Peak Prominence": params["ccf_peak_thresh"],
        "Rise/Fall Landmark Height": params.get("edge_height_fraction", 0.5),
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
            "Plot Landmark Shifts": params["plot_flags"].get("plot_landmark_shifts", False),
            "Plot Individual Landmark Shifts": params["plot_flags"].get("plot_indv_landmark_shifts", False),
            "Plot FTs": params["plot_flags"].get("plot_fts", False),
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
            "Rise/Fall Landmark Height": params.get("edge_height_fraction", 0.5),
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
            "Plot Landmark Shifts": params["plot_flags"].get("plot_landmark_shifts", False),
            "Plot Individual Landmark Shifts": params["plot_flags"].get("plot_indv_landmark_shifts", False),
            "Plot FTs": params["plot_flags"].get("plot_fts", False),
            "Dark Plots": params["plot_flags"]["dark_plots"],
            "Calc Wave Speeds": params["calculate_wave_speeds"],
            "Plot Wave Speeds": True,
        })

    return log_params


def _run_analysis(gui):
    """Run the appropriate analysis workflow in a background thread."""

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    writer = _GUIWriter(gui)
    sys.stdout = writer
    sys.stderr = writer

    try:
        gui.clear_log()
        gui.reset_progress()
        gui.set_status("Running analysis...")
        gui.start_timer()
        gui.log_message("Starting analysis...")

        params = gui._resolved_params
        analysis_type = params["analysis_type"]
        log_params = _build_log_params(params, analysis_type)

        hf.threshold_check(params["acf_peak_thresh"], log_params)

        if log_params["Errors"]:
            for err in log_params["Errors"]:
                gui.log_message(f"ERROR: {err}")
            gui.set_status("Error - check log above")
            return

        folder = params["folder_path"]
        total_files = len([f for f in os.listdir(folder) if f.endswith(".tif") and not f.startswith(".")])
        gui.log_message(f"Found {total_files} file(s) to process")
        gui.log_message("")
        writer.set_file_tracking(total_files)

        # Custom channel display names from the GUI (blanks fall back to 'ChN' in plots)
        channel_names = [params.get(f"Ch{i}_name", "") for i in range(1, 5)]

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
                channel_names=channel_names,
                edge_height_fraction=params["edge_height_fraction"],
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
                channel_names=channel_names,
                injection_ch1=params.get("injection_ch1", False),
                injection_ch2=params.get("injection_ch2", False),
                injection_frame=params.get("injection_frame", None),
                edge_height_fraction=params.get("edge_height_fraction", 0.5),
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
                smoothing=params["smoothing"],
                channel_names=channel_names,
                edge_height_fraction=params["edge_height_fraction"],
            )

        if log_params["Errors"]:
            gui.log_message("")
            gui.log_message("--- ERRORS ENCOUNTERED ---")
            for err in log_params["Errors"]:
                gui.log_message(f"  ERROR: {err}")
            gui.log_message("Check the log file for more information.")
            gui.set_status("Complete (with errors)")
        else:
            gui.log_message("")
            gui.log_message("--- Analysis finished successfully ---")
            gui.set_status("Analysis complete!")

        processed = log_params.get("Files Processed", [])
        not_processed = log_params.get("Files Not Processed", [])
        if processed or not_processed:
            gui.log_message(f"Files processed: {len(processed)}  |  Skipped: {len(not_processed)}")

        import glob
        results_dirs = sorted(
            glob.glob(os.path.join(folder, "0_signalProcessing-*")),
            key=os.path.getmtime,
        )
        if results_dirs:
            gui.after(0, lambda: gui.show_results_buttons(results_dirs[-1]))

    except _StopAnalysis:
        pass  # already logged in the writer
    except Exception as e:
        gui.log_message(f"ERROR: {e}")
        gui.set_status("Error occurred")
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        gui.stop_timer()
        gui._is_running = False
        try:
            gui.after(0, lambda: gui.start_button.configure(state="normal"))
            gui.after(0, lambda: gui.stop_button.configure(state="disabled"))
        except Exception:
            pass


def _setup_and_run(gui):
    def on_start():
        thread = threading.Thread(target=_run_analysis, args=(gui,), daemon=True)
        thread.start()
    gui.on_start = on_start


def main():
    '''
    Main function to run the wave analysis GUI and analysis workflows.
    Loops so the user can navigate between standard / rolling / kymograph
    GUIs without restarting the program.
    '''
    import gc

    _GUI_FOR_MODE = {
        "standard": BaseGUI,
        "rolling": RollingGUI,
        "kymograph": KymographGUI,
    }

    mode = "standard"
    while mode is not None:
        gui = _GUI_FOR_MODE[mode]()
        _setup_and_run(gui)
        gui.mainloop()

        # Work out where to go next from the flags the GUI set before closing.
        if mode == "standard":
            if getattr(gui, "rolling", False):
                next_mode = "rolling"
            elif getattr(gui, "kymograph", False):
                next_mode = "kymograph"
            else:
                next_mode = None
        else:  # rolling or kymograph
            next_mode = "standard" if getattr(gui, "_back_to_standard", False) else None

        # Each mode uses its own Tk root (its own Tcl interpreter). The window is
        # already destroyed at this point, but the Python object survives in a
        # reference cycle until a garbage-collection sweep frees it -- and that
        # sweep could otherwise happen inside the *next* GUI's background
        # analysis thread. Deleting a Tcl interpreter from any thread other than
        # the one that created it makes Tcl abort the whole process
        # (Tcl_AsyncDelete panic). Dropping the reference and collecting here
        # forces the interpreter to be torn down on the main thread, before the
        # next GUI (and its worker thread) ever starts.
        gui = None
        gc.collect()

        mode = next_mode


if __name__ == "__main__":
    main()
