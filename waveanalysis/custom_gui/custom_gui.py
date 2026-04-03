import sys
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askdirectory


class _GUIBase(tk.Tk):
    """Shared widget helpers and actions used by all three analysis GUI classes."""

    _has_group_names = False  # subclasses with a group_names field override to True
    _has_plot_flags = False  # subclasses with plot flag checkboxes override to True

    # ---- WIDGET HELPERS ----

    def _add_entry(self, row, col, var, label_text, width=3):
        entry = ttk.Entry(self, width=width, textvariable=var)
        entry.grid(row=row, column=col, padx=10, sticky="E")
        label = ttk.Label(self, text=label_text)
        label.grid(row=row, column=col+1, padx=10, sticky="W")
        return entry, label

    def _add_checkbutton(self, row, col, var, label_text):
        checkbox = ttk.Checkbutton(self, variable=var)
        checkbox.grid(row=row, column=col, padx=10, sticky="E")
        label = ttk.Label(self, text=label_text)
        label.grid(row=row, column=col+1, padx=10, sticky="W")
        return checkbox, label

    def _add_smoothing_channel(self, row, col, name, label_prefix, default_window=11, default_poly=2):
        cb = ttk.Checkbutton(self, variable=self.vars[f"{name}_smoothing"])
        cb.grid(row=row, column=col, padx=(10, 4), sticky="E")

        ttk.Label(self, text=label_prefix, width=4).grid(row=row, column=col+1, sticky="W")

        ttk.Label(self, text="win").grid(row=row, column=col+2, padx=(8, 2), sticky="E")
        win_spinbox = ttk.Spinbox(self, from_=3, to=101, increment=2, width=4)
        win_spinbox.set(default_window)
        win_spinbox.grid(row=row, column=col+3, padx=(0, 8), sticky="W")

        ttk.Label(self, text="poly").grid(row=row, column=col+4, padx=(4, 2), sticky="E")
        poly_spinbox = ttk.Spinbox(self, from_=1, to=10, increment=1, width=3)
        poly_spinbox.set(default_poly)
        poly_spinbox.grid(row=row, column=col+5, padx=(0, 10), sticky="W")

        return cb, win_spinbox, poly_spinbox

    def _build_smoothing_section(self, default_poly=2):
        ttk.Label(self, text="SMOOTHING OPTIONS",
                  font=("TkDefaultFont", 16, "bold")).grid(row=0, column=3, columnspan=7, padx=10, pady=10, sticky="EW")
        self.smoothing_widgets = {
            "Ch1": self._add_smoothing_channel(1, 3, "Ch1", "Ch1", default_poly=default_poly),
            "Ch2": self._add_smoothing_channel(2, 3, "Ch2", "Ch2", default_poly=default_poly),
            "Ch3": self._add_smoothing_channel(3, 3, "Ch3", "Ch3", default_poly=default_poly),
            "Ch4": self._add_smoothing_channel(4, 3, "Ch4", "Ch4", default_poly=default_poly),
            "CCF": self._add_smoothing_channel(5, 3, "CCF", "CCF", default_poly=default_poly),
        }

    # ---- ACTIONS ----

    def get_folder_path(self):
        self.vars["folder_path"].set(askdirectory())

    def cancel_analysis(self):
        sys.exit("You have cancelled the analysis")

    def start_analysis(self):
        for k, v in self.vars.items():
            self.vars[k] = v.get()
        if self._has_group_names:
            self.vars["group_names"] = [g.strip() for g in self.vars["group_names"].split(",")]
        smoothing_params = {}
        for ch in ["Ch1", "Ch2", "Ch3", "Ch4", "CCF"]:
            if self.vars[f"{ch}_smoothing"]:
                smoothing_params[ch] = {
                    "window": int(self.smoothing_widgets[ch][1].get()),
                    "poly_order": int(self.smoothing_widgets[ch][2].get()),
                }
            else:
                smoothing_params[ch] = None
        self.vars["smoothing_params"] = smoothing_params
        if self._has_plot_flags:
            self.vars["plot_flags"] = {
                "plot_summary_ACFs": self.vars["plot_summary_ACFs"],
                "plot_summary_CCFs": self.vars["plot_summary_CCFs"],
                "plot_summary_peaks": self.vars["plot_summary_peaks"],
                "plot_indv_ACFs": self.vars["plot_indv_ACFs"],
                "plot_indv_CCFs": self.vars["plot_indv_CCFs"],
                "plot_indv_peaks": self.vars["plot_indv_peaks"],
                "dark_plots": self.vars["dark_plots"],
            }
        self.destroy()


class BaseGUI(_GUIBase):
    _has_group_names = True
    _has_plot_flags = True

    def __init__(self):
        super().__init__()

        self.title("Define your analysis parameters")
        self.main_frame = ttk.Frame(self, padding="20")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # ---- VARIABLES ----
        self.vars = {
            "analysis_type": tk.StringVar(value="standard"),
            "box_size": tk.IntVar(value=20),
            "bin_shift": tk.IntVar(value=20),
            "small_shifts_correction": tk.BooleanVar(value=True),
            "plot_summary_ACFs": tk.BooleanVar(value=True),
            "plot_summary_CCFs": tk.BooleanVar(value=True),
            "plot_summary_peaks": tk.BooleanVar(value=True),
            "plot_indv_ACFs": tk.BooleanVar(value=False),
            "plot_indv_CCFs": tk.BooleanVar(value=False),
            "plot_indv_peaks": tk.BooleanVar(value=False),
            "dark_plots": tk.BooleanVar(value=True),
            "acf_peak_thresh": tk.DoubleVar(value=0.1),
            "ccf_peak_thresh": tk.DoubleVar(value=0.1),
            "group_names": tk.StringVar(value=""),
            "smoothing": tk.BooleanVar(value=True),
            "folder_path": tk.StringVar(value=""),
            "Ch1_smoothing": tk.BooleanVar(value=True),
            "Ch2_smoothing": tk.BooleanVar(value=True),
            "Ch3_smoothing": tk.BooleanVar(value=True),
            "Ch4_smoothing": tk.BooleanVar(value=True),
            "CCF_smoothing": tk.BooleanVar(value=True),
        }

        self.rolling = False
        self.kymograph = False

        # ---- ANALYSIS OPTIONS ----
        ttk.Label(self, text="ANALYSIS OPTIONS",
                  font=("TkDefaultFont", 16, "bold"), anchor="center").grid(
            row=0, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="ew")

        # ---- FILE SELECTION ----
        ttk.Entry(self, textvariable=self.vars["folder_path"]).grid(row=1, column=0, padx=10, sticky="E")
        ttk.Button(self, text="Select folder", command=self.get_folder_path).grid(row=1, column=1, padx=10, sticky="W")

        # ---- GROUP NAMES ----
        ttk.Label(self, text="Group names").grid(row=2, column=1, padx=10, sticky="W")
        ttk.Entry(self, textvariable=self.vars["group_names"]).grid(row=2, column=0, padx=10, sticky="E")

        # ---- BASIC ENTRIES ----
        self._add_entry(3, 0, self.vars["box_size"], "Box size (pixels)")
        self._add_entry(4, 0, self.vars["bin_shift"], "Box shift (pixels)")
        self._add_entry(5, 0, self.vars["acf_peak_thresh"], "ACF peak threshold")
        self._add_entry(6, 0, self.vars["ccf_peak_thresh"], "CCF peak threshold")
        self._add_checkbutton(7, 0, self.vars["small_shifts_correction"], "Small shifts correction")

        # ---- SEPARATORS ----
        ttk.Separator(self, orient="horizontal").grid(row=8, column=0, columnspan=11, sticky="ew", pady=10)
        ttk.Separator(self, orient="vertical").grid(row=0, column=2, rowspan=9, sticky="ns", pady=10)

        # ---- PLOT OPTIONS ----
        ttk.Label(self, text="PLOT OPTIONS",
                  font=("TkDefaultFont", 16, "bold"), anchor="center").grid(
            row=9, column=1, columnspan=3, padx=10, pady=(0, 10), sticky="ew")
        self._add_checkbutton(10, 0, self.vars["plot_summary_ACFs"], "Plot summary ACFs")
        self._add_checkbutton(11, 0, self.vars["plot_summary_CCFs"], "Plot summary CCFs")
        self._add_checkbutton(12, 0, self.vars["plot_summary_peaks"], "Plot summary peaks")
        self._add_checkbutton(10, 2, self.vars["plot_indv_ACFs"], "Plot individual ACFs")
        self._add_checkbutton(11, 2, self.vars["plot_indv_CCFs"], "Plot individual CCFs")
        self._add_checkbutton(12, 2, self.vars["plot_indv_peaks"], "Plot individual peaks")
        self._add_checkbutton(10, 4, self.vars["dark_plots"], "Dark plots")

        # ---- SMOOTHING OPTIONS ----
        self._build_smoothing_section(default_poly=2)
        ttk.Checkbutton(self, variable=self.vars["smoothing"]).grid(row=6, column=3, padx=(10, 4), sticky="E")
        ttk.Label(self, text="Enable smoothing").grid(row=6, column=4, sticky="W")

        # ---- BUTTONS ----
        ttk.Button(self, text="Start analysis", command=self.start_analysis).grid(row=10, column=8, columnspan=2, padx=10, sticky="E")
        ttk.Button(self, text="Cancel", command=self.cancel_analysis).grid(row=9, column=8, columnspan=2, padx=10, sticky="E")
        ttk.Button(self, text="Launch rolling analysis", command=self.launch_rolling_analysis).grid(row=11, column=8, columnspan=2, padx=10, sticky="E")
        ttk.Button(self, text="Launch kymograph analysis", command=self.launch_kymograph_analysis).grid(row=12, column=8, columnspan=2, padx=10, sticky="E")

    def launch_rolling_analysis(self):
        self.rolling = True
        self.destroy()

    def launch_kymograph_analysis(self):
        self.kymograph = True
        self.destroy()


class RollingGUI(_GUIBase):

    def __init__(self):
        super().__init__()

        self.title("Define your analysis parameters")
        self.main_frame = ttk.Frame(self, padding="20")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # ---- VARIABLES ----
        self.vars = {
            "analysis_type": tk.StringVar(value="rolling"),
            "box_size": tk.IntVar(value=20),
            "bin_shift": tk.IntVar(value=20),
            "subframe_size": tk.IntVar(value=50),
            "subframe_roll": tk.IntVar(value=5),
            "small_shifts_correction": tk.BooleanVar(value=True),
            "plot_subframe_ACFs": tk.BooleanVar(value=True), # mandatory for now
            "plot_subframe_CCFs": tk.BooleanVar(value=True), # mandatory for now
            "plot_subframe_peaks": tk.BooleanVar(value=True), # mandatory for now
            "dark_plots": tk.BooleanVar(value=False),
            "acf_peak_thresh": tk.DoubleVar(value=0.1),
            "ccf_peak_thresh": tk.DoubleVar(value=0.1),
            "smoothing": tk.BooleanVar(value=True),
            "folder_path": tk.StringVar(value=""),
            "Ch1_smoothing": tk.BooleanVar(value=True),
            "Ch2_smoothing": tk.BooleanVar(value=True),
            "Ch3_smoothing": tk.BooleanVar(value=True),
            "Ch4_smoothing": tk.BooleanVar(value=True),
            "CCF_smoothing": tk.BooleanVar(value=True),
        }

        self.kymograph = False

        # ---- ANALYSIS OPTIONS ----
        ttk.Label(self, text="ANALYSIS OPTIONS",
                  font=("TkDefaultFont", 16, "bold"), anchor="center").grid(
            row=0, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="ew")

        # ---- FILE SELECTION ----
        ttk.Entry(self, textvariable=self.vars["folder_path"]).grid(row=1, column=0, padx=10, sticky="E")
        ttk.Button(self, text="Select folder", command=self.get_folder_path).grid(row=1, column=1, padx=10, sticky="W")

        # ---- BASIC ENTRIES ----
        self._add_entry(3, 0, self.vars["box_size"], "Box size (pixels)")
        self._add_entry(4, 0, self.vars["bin_shift"], "Box shift (pixels)")
        self._add_entry(5, 0, self.vars["subframe_size"], "Subframe size (frames)")
        self._add_entry(6, 0, self.vars["subframe_roll"], "Subframe roll (frames)")
        self._add_entry(7, 0, self.vars["acf_peak_thresh"], "ACF peak threshold")
        self._add_entry(8, 0, self.vars["ccf_peak_thresh"], "CCF peak threshold")
        self._add_checkbutton(9, 0, self.vars["small_shifts_correction"], "Small shifts correction")
        self._add_checkbutton(10, 0, self.vars["dark_plots"], "Dark plots")

        # ---- SEPARATORS ----
        ttk.Separator(self, orient="vertical").grid(row=0, column=2, rowspan=11, sticky="ns", pady=10)

        # ---- SMOOTHING OPTIONS ----
        self._build_smoothing_section(default_poly=3)

        # ---- BUTTONS ----
        ttk.Button(self, text="Start analysis", command=self.start_analysis).grid(row=11, column=7, columnspan=2, padx=10, sticky="E")
        ttk.Button(self, text="Cancel", command=self.cancel_analysis).grid(row=10, column=7, columnspan=2, padx=10, sticky="E")


class KymographGUI(_GUIBase):
    _has_group_names = True
    _has_plot_flags = True

    def __init__(self):
        super().__init__()

        self.title("Define your analysis parameters")
        self.main_frame = ttk.Frame(self, padding="20")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # ---- VARIABLES ----
        self.vars = {
            "analysis_type": tk.StringVar(value="kymograph"),
            "line_width": tk.IntVar(value=5),
            "bin_shift": tk.IntVar(value=5),
            "small_shifts_correction": tk.BooleanVar(value=False),
            "plot_summary_ACFs": tk.BooleanVar(value=True),
            "plot_summary_CCFs": tk.BooleanVar(value=True),
            "plot_summary_peaks": tk.BooleanVar(value=True),
            "plot_indv_ACFs": tk.BooleanVar(value=False),
            "plot_indv_CCFs": tk.BooleanVar(value=False),
            "plot_indv_peaks": tk.BooleanVar(value=False),
            "dark_plots": tk.BooleanVar(value=False),
            "acf_peak_thresh": tk.DoubleVar(value=0.1),
            "ccf_peak_thresh": tk.DoubleVar(value=0.1),
            "group_names": tk.StringVar(value=""),
            "smoothing": tk.BooleanVar(value=True),
            "folder_path": tk.StringVar(value=""),
            "calculate_wave_speeds": tk.BooleanVar(value=False),
            "Ch1_smoothing": tk.BooleanVar(value=True),
            "Ch2_smoothing": tk.BooleanVar(value=True),
            "Ch3_smoothing": tk.BooleanVar(value=True),
            "Ch4_smoothing": tk.BooleanVar(value=True),
            "CCF_smoothing": tk.BooleanVar(value=True),
        }

        self.rolling = False

        # ---- ANALYSIS OPTIONS ----
        ttk.Label(self, text="ANALYSIS OPTIONS",
                  font=("TkDefaultFont", 16, "bold"), anchor="center").grid(
            row=0, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="ew")

        # ---- FILE SELECTION ----
        ttk.Entry(self, textvariable=self.vars["folder_path"]).grid(row=1, column=0, padx=10, sticky="E")
        ttk.Button(self, text="Select folder", command=self.get_folder_path).grid(row=1, column=1, padx=10, sticky="W")

        # ---- GROUP NAMES ----
        ttk.Label(self, text="Group names").grid(row=2, column=1, padx=10, sticky="W")
        ttk.Entry(self, textvariable=self.vars["group_names"]).grid(row=2, column=0, padx=10, sticky="E")

        # ---- BASIC ENTRIES ----
        self._add_entry(3, 0, self.vars["line_width"], "Line width (pixels)")
        self._add_entry(4, 0, self.vars["bin_shift"], "Line shift (pixels)")
        self._add_entry(5, 0, self.vars["acf_peak_thresh"], "ACF peak threshold")
        self._add_entry(6, 0, self.vars["ccf_peak_thresh"], "CCF peak threshold")
        self._add_checkbutton(7, 0, self.vars["small_shifts_correction"], "Small shifts correction")

        # ---- SEPARATORS ----
        ttk.Separator(self, orient="horizontal").grid(row=8, column=0, columnspan=11, sticky="ew", pady=10)
        ttk.Separator(self, orient="vertical").grid(row=0, column=2, rowspan=9, sticky="ns", pady=10)

        # ---- PLOT OPTIONS ----
        ttk.Label(self, text="PLOT OPTIONS",
                  font=("TkDefaultFont", 16, "bold"), anchor="center").grid(
            row=9, column=1, columnspan=3, padx=10, pady=(0, 10), sticky="ew")
        self._add_checkbutton(10, 0, self.vars["plot_summary_ACFs"], "Plot summary ACFs")
        self._add_checkbutton(11, 0, self.vars["plot_summary_CCFs"], "Plot summary CCFs")
        self._add_checkbutton(12, 0, self.vars["plot_summary_peaks"], "Plot summary peaks")
        self._add_checkbutton(10, 2, self.vars["plot_indv_ACFs"], "Plot individual ACFs")
        self._add_checkbutton(11, 2, self.vars["plot_indv_CCFs"], "Plot individual CCFs")
        self._add_checkbutton(12, 2, self.vars["plot_indv_peaks"], "Plot individual peaks")
        self._add_checkbutton(10, 4, self.vars["dark_plots"], "Dark plots")

        # ---- SMOOTHING OPTIONS ----
        self._build_smoothing_section(default_poly=3)
        ttk.Checkbutton(self, variable=self.vars["smoothing"]).grid(row=6, column=3, padx=(10, 4), sticky="E")
        ttk.Label(self, text="Enable smoothing").grid(row=6, column=4, sticky="W")

        # ---- BUTTONS ----
        ttk.Button(self, text="Start analysis", command=self.start_analysis).grid(row=10, column=8, columnspan=2, padx=10, sticky="E")
        ttk.Button(self, text="Cancel", command=self.cancel_analysis).grid(row=9, column=8, columnspan=2, padx=10, sticky="E")
