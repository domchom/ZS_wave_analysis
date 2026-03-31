import sys
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askdirectory

class BaseGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        # configure root window
        self.title("Define your analysis parameters")
        self.main_frame = ttk.Frame(self, padding="20")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # ---- VARIABLES ----
        self.vars = {
            "analysis_type": tk.StringVar(value="standard"),
            "box_size": tk.IntVar(value=20),
            "bin_shift": tk.IntVar(value=20),
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
            "Ch1_smoothing": tk.BooleanVar(value=True),
            "Ch2_smoothing": tk.BooleanVar(value=True),
            "Ch3_smoothing": tk.BooleanVar(value=True),
            "Ch4_smoothing": tk.BooleanVar(value=True),
            "CCF_smoothing": tk.BooleanVar(value=True),
        }

        self.rolling = False
        self.kymograph = False

        # ---- HELPERS ----
        def add_entry(row, col, var, label_text, width=3):
            entry = ttk.Entry(self, width=width, textvariable=var)
            entry.grid(row=row, column=col, padx=10, sticky="E")
            label = ttk.Label(self, text=label_text)
            label.grid(row=row, column=col+1, padx=10, sticky="W")
            return entry, label

        def add_checkbutton(row, col, var, label_text):
            checkbox = ttk.Checkbutton(self, variable=var)
            checkbox.grid(row=row, column=col, padx=10, sticky="E")
            label = ttk.Label(self, text=label_text)
            label.grid(row=row, column=col+1, padx=10, sticky="W")
            return checkbox, label

        def add_smoothing_channel(row, col, name, label_prefix):
            cb = ttk.Checkbutton(self, variable=self.vars[f"{name}_smoothing"])
            cb.grid(row=row, column=col, padx=10, sticky="E")

            win_label = ttk.Label(self, text=f"{label_prefix} window size")
            win_label.grid(row=row, column=col+1, padx=10, sticky="W")
            win_scale = tk.Scale(self, from_=3, to=101, orient="horizontal", resolution=2)
            win_scale.set(11)
            win_scale.grid(row=row, column=col+2, padx=10, pady=5, sticky="W")

            poly_label = ttk.Label(self, text=f"{label_prefix} poly order")
            poly_label.grid(row=row+1, column=col+1, padx=10, sticky="W")
            poly_scale = tk.Scale(self, from_=1, to=10, orient="horizontal", resolution=1)
            poly_scale.set(3)
            poly_scale.grid(row=row+1, column=col+2, padx=10, sticky="W")

            return cb, win_scale, poly_scale
        
        # ---- ANALYSIS OPTIONS ----
        analysis_label = ttk.Label(
            self,
            text="ANALYSIS OPTIONS",
            font=("TkDefaultFont", 16, "bold"),
            anchor="center"
        )
        analysis_label.grid(row=0, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="ew")

        # ---- FILE SELECTION ----
        self.file_path_entry = ttk.Entry(self, textvariable=self.vars["folder_path"])
        self.file_path_entry.grid(row=1, column=0, padx=10, sticky="E")
        self.file_path_button = ttk.Button(self, text="Select folder", command=self.get_folder_path)
        self.file_path_button.grid(row=1, column=1, padx=10, sticky="W")
        
        # ---- GROUP NAMES ----
        self.group_names_label = ttk.Label(self, text="Group names")
        self.group_names_label.grid(row=2, column=1, padx=10, sticky="W")
        self.group_names_entry = ttk.Entry(self, textvariable=self.vars["group_names"])
        self.group_names_entry.grid(row=2, column=0, padx=10, sticky="E")

        # ---- BASIC ENTRIES ----
        add_entry(3, 0, self.vars["box_size"], "Box size (pixels)")
        add_entry(4, 0, self.vars["bin_shift"], "Box shift (pixels)")
        add_entry(5, 0, self.vars["acf_peak_thresh"], "ACF peak threshold")
        add_entry(6, 0, self.vars["ccf_peak_thresh"], "CCF peak threshold")
        add_checkbutton(7, 0, self.vars["small_shifts_correction"], "Small shifts correction")

        # ---- SEPARATORS ----
        ttk.Separator(self, orient="horizontal").grid(row=8, column=0, columnspan=11, sticky="ew", pady=10)
        ttk.Separator(self, orient="vertical").grid(row=0, column=2, rowspan=9, sticky="ns", pady=10)

        # ---- PLOT OPTIONS ----
        plotting_label = ttk.Label(
            self,
            text="PLOT OPTIONS",
            font=("TkDefaultFont", 16, "bold"),
            anchor="center"
        )
        plotting_label.grid(row=9, column=1, columnspan=3, padx=10, pady=(0, 10), sticky="ew")

        add_checkbutton(10, 0, self.vars["plot_summary_ACFs"], "Plot summary ACFs")
        add_checkbutton(11, 0, self.vars["plot_summary_CCFs"], "Plot summary CCFs")
        add_checkbutton(12, 0, self.vars["plot_summary_peaks"], "Plot summary peaks")
        add_checkbutton(10, 2, self.vars["plot_indv_ACFs"], "Plot individual ACFs")
        add_checkbutton(11, 2, self.vars["plot_indv_CCFs"], "Plot individual CCFs")
        add_checkbutton(12, 2, self.vars["plot_indv_peaks"], "Plot individual peaks")
        add_checkbutton(10, 4, self.vars["dark_plots"], "Dark plots")

        # ---- SMOOTHING OPTIONS ----
        ttk.Label(self, text="SMOOTHING OPTIONS",
                  font=("TkDefaultFont", 16, "bold")).grid(row=0, column=5, columnspan=6, padx=10, pady=10, sticky="EW")

        self.smoothing_widgets = {
            "Ch1": add_smoothing_channel(1, 3, "Ch1", "Ch1"),
            "Ch2": add_smoothing_channel(3, 3, "Ch2", "Ch2"),
            "Ch3": add_smoothing_channel(1, 7, "Ch3", "Ch3"),
            "Ch4": add_smoothing_channel(3, 7, "Ch4", "Ch4"),
            "CCF": add_smoothing_channel(5, 3, "CCF", "CCF"),
        }
        
        ttk.Label(self, text="Enable smoothing").grid(row=5, column=8, padx=10, sticky="E")
        ttk.Checkbutton(self, variable=self.vars["smoothing"]).grid(row=5, column=7, padx=10, sticky="E")
        
        # ---- BUTTONS ----
        self.start_button = ttk.Button(self, text="Start analysis", command=self.start_analysis)
        self.start_button.grid(row=10, column=8, columnspan=2, padx=10, sticky="E")

        self.cancel_button = ttk.Button(self, text="Cancel", command=self.cancel_analysis)
        self.cancel_button.grid(row=9, column=8, columnspan=2, padx=10, sticky="E")

        self.rolling_button = ttk.Button(self, text="Launch rolling analysis", command=self.launch_rolling_analysis)
        self.rolling_button.grid(row=11, column=8, columnspan=2, padx=10, sticky="E")

        self.kymograph_button = ttk.Button(self, text="Launch kymograph analysis", command=self.launch_kymograph_analysis)
        self.kymograph_button.grid(row=12, column=8, columnspan=2, padx=10, sticky="E")

    # ---- METHODS ----
    def get_folder_path(self):
        self.vars["folder_path"].set(askdirectory())

    def launch_rolling_analysis(self):
        self.rolling = True
        self.destroy()

    def launch_kymograph_analysis(self):
        self.kymograph = True
        self.destroy()

    def cancel_analysis(self):
        sys.exit("You have cancelled the analysis")

    def start_analysis(self):
        # Collect values
        for k, v in self.vars.items():
            self.vars[k] = v.get()

        # Group names
        self.vars["group_names"] = [g.strip() for g in self.vars["group_names"].split(",")]
        
        for ch in ["Ch1", "Ch2", "Ch3", "Ch4", "CCF"]:
            if self.vars[f"{ch}_smoothing"]:
                self.vars[f"{ch}_window"] = self.smoothing_widgets[ch][1].get()
                self.vars[f"{ch}_poly_order"] = self.smoothing_widgets[ch][2].get()
            else:
                self.vars[f"{ch}_window"] = None
                self.vars[f"{ch}_poly_order"] = None
        
        # Destroy widget
        self.destroy()

class RollingGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        # configure root window
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
        
        # ---- HELPERS ----
        def add_entry(row, col, var, label_text, width=3):
            entry = ttk.Entry(self, width=width, textvariable=var)
            entry.grid(row=row, column=col, padx=10, sticky="E")
            label = ttk.Label(self, text=label_text)
            label.grid(row=row, column=col+1, padx=10, sticky="W")
            return entry, label

        def add_checkbutton(row, col, var, label_text):
            checkbox = ttk.Checkbutton(self, variable=var)
            checkbox.grid(row=row, column=col, padx=10, sticky="E")
            label = ttk.Label(self, text=label_text)
            label.grid(row=row, column=col+1, padx=10, sticky="W")
            return checkbox, label

        def add_smoothing_channel(row, col, name, label_prefix):
            cb = ttk.Checkbutton(self, variable=self.vars[f"{name}_smoothing"])
            cb.grid(row=row, column=col, padx=10, sticky="E")

            win_label = ttk.Label(self, text=f"{label_prefix} window size")
            win_label.grid(row=row, column=col+1, padx=10, sticky="W")
            win_scale = tk.Scale(self, from_=3, to=101, orient="horizontal", resolution=2)
            win_scale.set(11)
            win_scale.grid(row=row, column=col+2, padx=10, pady=5, sticky="W")

            poly_label = ttk.Label(self, text=f"{label_prefix} poly order")
            poly_label.grid(row=row+1, column=col+1, padx=10, sticky="W")
            poly_scale = tk.Scale(self, from_=1, to=10, orient="horizontal", resolution=1)
            poly_scale.set(3)
            poly_scale.grid(row=row+1, column=col+2, padx=10, sticky="W")

            return cb, win_scale, poly_scale
        
        # ---- ANALYSIS OPTIONS ----
        analysis_label = ttk.Label(
            self,
            text="ANALYSIS OPTIONS",
            font=("TkDefaultFont", 16, "bold"),
            anchor="center"
        )
        analysis_label.grid(row=0, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="ew")

        # ---- FILE SELECTION ----
        self.file_path_entry = ttk.Entry(self, textvariable=self.vars["folder_path"])
        self.file_path_entry.grid(row=1, column=0, padx=10, sticky="E")
        self.file_path_button = ttk.Button(self, text="Select folder", command=self.get_folder_path)
        self.file_path_button.grid(row=1, column=1, padx=10, sticky="W")
        
        # ---- BASIC ENTRIES ----
        add_entry(3, 0, self.vars["box_size"], "Box size (pixels)")
        add_entry(4, 0, self.vars["bin_shift"], "Box shift (pixels)")
        add_entry(5, 0, self.vars["subframe_size"], "Subframe size (frames)")
        add_entry(6, 0, self.vars["subframe_roll"], "Subframe roll (frames)")
        add_entry(7, 0, self.vars["acf_peak_thresh"], "ACF peak threshold")
        add_entry(8, 0, self.vars["ccf_peak_thresh"], "CCF peak threshold")
        add_checkbutton(9, 0, self.vars["small_shifts_correction"], "Small shifts correction")  
        add_checkbutton(10,0, self.vars["dark_plots"], "Dark plots")

        # ---- SEPARATORS ----
        ttk.Separator(self, orient="vertical").grid(row=0, column=2, rowspan=11, sticky="ns", pady=10)
        
        # ---- SMOOTHING OPTIONS ----
        ttk.Label(self, text="SMOOTHING OPTIONS",
                  font=("TkDefaultFont", 16, "bold")).grid(row=0, column=5, columnspan=6, padx=10, pady=10, sticky="EW")

        self.smoothing_widgets = {
            "Ch1": add_smoothing_channel(1, 3, "Ch1", "Ch1"),
            "Ch2": add_smoothing_channel(3, 3, "Ch2", "Ch2"),
            "Ch3": add_smoothing_channel(1, 7, "Ch3", "Ch3"),
            "Ch4": add_smoothing_channel(3, 7, "Ch4", "Ch4"),
            "CCF": add_smoothing_channel(5, 3, "CCF", "CCF"),
        }
    
        # ---- BUTTONS ----
        self.start_button = ttk.Button(self, text="Start analysis", command=self.start_analysis)
        self.start_button.grid(row=10, column=7, columnspan=2, padx=10, sticky="E")

        self.cancel_button = ttk.Button(self, text="Cancel", command=self.cancel_analysis)
        self.cancel_button.grid(row=10, column=8, columnspan=2, padx=10, sticky="E")

    # ---- METHODS ----
    def get_folder_path(self):
        self.vars["folder_path"].set(askdirectory())

    def cancel_analysis(self):
        sys.exit('You have cancelled the analysis')
    
    def cancel_analysis(self):
        sys.exit("You have cancelled the analysis")

    def start_analysis(self):
        # Collect values
        for k, v in self.vars.items():
            self.vars[k] = v.get()
        
        for ch in ["Ch1", "Ch2", "Ch3", "Ch4", "CCF"]:
            if self.vars[f"{ch}_smoothing"]:
                self.vars[f"{ch}_window"] = self.smoothing_widgets[ch][1].get()
                self.vars[f"{ch}_poly_order"] = self.smoothing_widgets[ch][2].get()
            else:
                self.vars[f"{ch}_window"] = None
                self.vars[f"{ch}_poly_order"] = None
        
        # Destroy widget
        self.destroy()

class KymographGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        # configure root window
        self.title("Define your analysis parameters")
        # Configure the main frame
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

        # ---- HELPERS ----
        def add_entry(row, col, var, label_text, width=3):
            entry = ttk.Entry(self, width=width, textvariable=var)
            entry.grid(row=row, column=col, padx=10, sticky="E")
            label = ttk.Label(self, text=label_text)
            label.grid(row=row, column=col+1, padx=10, sticky="W")
            return entry, label

        def add_checkbutton(row, col, var, label_text):
            checkbox = ttk.Checkbutton(self, variable=var)
            checkbox.grid(row=row, column=col, padx=10, sticky="E")
            label = ttk.Label(self, text=label_text)
            label.grid(row=row, column=col+1, padx=10, sticky="W")
            return checkbox, label

        def add_smoothing_channel(row, col, name, label_prefix):
            cb = ttk.Checkbutton(self, variable=self.vars[f"{name}_smoothing"])
            cb.grid(row=row, column=col, padx=10, sticky="E")

            win_label = ttk.Label(self, text=f"{label_prefix} window size")
            win_label.grid(row=row, column=col+1, padx=10, sticky="W")
            win_scale = tk.Scale(self, from_=3, to=101, orient="horizontal", resolution=2)
            win_scale.set(11)
            win_scale.grid(row=row, column=col+2, padx=10, pady=5, sticky="W")

            poly_label = ttk.Label(self, text=f"{label_prefix} poly order")
            poly_label.grid(row=row+1, column=col+1, padx=10, sticky="W")
            poly_scale = tk.Scale(self, from_=1, to=10, orient="horizontal", resolution=1)
            poly_scale.set(3)
            poly_scale.grid(row=row+1, column=col+2, padx=10, sticky="W")

            return cb, win_scale, poly_scale
        
        # ---- ANALYSIS OPTIONS ----
        analysis_label = ttk.Label(
            self,
            text="ANALYSIS OPTIONS",
            font=("TkDefaultFont", 16, "bold"),
            anchor="center"
        )
        analysis_label.grid(row=0, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="ew")

        # ---- FILE SELECTION ----
        self.file_path_entry = ttk.Entry(self, textvariable=self.vars["folder_path"])
        self.file_path_entry.grid(row=1, column=0, padx=10, sticky="E")
        self.file_path_button = ttk.Button(self, text="Select folder", command=self.get_folder_path)
        self.file_path_button.grid(row=1, column=1, padx=10, sticky="W")
        
        # ---- GROUP NAMES ----
        self.group_names_label = ttk.Label(self, text="Group names")
        self.group_names_label.grid(row=2, column=1, padx=10, sticky="W")
        self.group_names_entry = ttk.Entry(self, textvariable=self.vars["group_names"])
        self.group_names_entry.grid(row=2, column=0, padx=10, sticky="E")

        # ---- BASIC ENTRIES ----
        add_entry(3, 0, self.vars["line_width"], "Line width (pixels)")
        add_entry(4, 0, self.vars["bin_shift"], "Line shift (pixels)")
        add_entry(5, 0, self.vars["acf_peak_thresh"], "ACF peak threshold")
        add_entry(6, 0, self.vars["ccf_peak_thresh"], "CCF peak threshold")
        add_checkbutton(7, 0, self.vars["small_shifts_correction"], "Small shifts correction")

        # ---- SEPARATORS ----
        ttk.Separator(self, orient="horizontal").grid(row=8, column=0, columnspan=11, sticky="ew", pady=10)
        ttk.Separator(self, orient="vertical").grid(row=0, column=2, rowspan=9, sticky="ns", pady=10)

        # ---- PLOT OPTIONS ----
        plotting_label = ttk.Label(
            self,
            text="PLOT OPTIONS",
            font=("TkDefaultFont", 16, "bold"),
            anchor="center"
        )
        plotting_label.grid(row=9, column=1, columnspan=3, padx=10, pady=(0, 10), sticky="ew")

        add_checkbutton(10, 0, self.vars["plot_summary_ACFs"], "Plot summary ACFs")
        add_checkbutton(11, 0, self.vars["plot_summary_CCFs"], "Plot summary CCFs")
        add_checkbutton(12, 0, self.vars["plot_summary_peaks"], "Plot summary peaks")
        add_checkbutton(10, 2, self.vars["plot_indv_ACFs"], "Plot individual ACFs")
        add_checkbutton(11, 2, self.vars["plot_indv_CCFs"], "Plot individual CCFs")
        add_checkbutton(12, 2, self.vars["plot_indv_peaks"], "Plot individual peaks")
        add_checkbutton(10, 4, self.vars["dark_plots"], "Dark plots")

        # ---- SMOOTHING OPTIONS ----
        ttk.Label(self, text="SMOOTHING OPTIONS",
                  font=("TkDefaultFont", 16, "bold")).grid(row=0, column=5, columnspan=6, padx=10, pady=10, sticky="EW")

        self.smoothing_widgets = {
            "Ch1": add_smoothing_channel(1, 3, "Ch1", "Ch1"),
            "Ch2": add_smoothing_channel(3, 3, "Ch2", "Ch2"),
            "Ch3": add_smoothing_channel(1, 7, "Ch3", "Ch3"),
            "Ch4": add_smoothing_channel(3, 7, "Ch4", "Ch4"),
            "CCF": add_smoothing_channel(5, 3, "CCF", "CCF"),
        }
        
        ttk.Label(self, text="Enable smoothing").grid(row=5, column=8, padx=10, sticky="E")
        ttk.Checkbutton(self, variable=self.vars["smoothing"]).grid(row=5, column=7, padx=10, sticky="E")
        
        # ---- BUTTONS ----
        self.start_button = ttk.Button(self, text="Start analysis", command=self.start_analysis)
        self.start_button.grid(row=10, column=8, columnspan=2, padx=10, sticky="E")

        self.cancel_button = ttk.Button(self, text="Cancel", command=self.cancel_analysis)
        self.cancel_button.grid(row=9, column=8, columnspan=2, padx=10, sticky="E")

    # ---- METHODS ----
    def get_folder_path(self):
        self.vars["folder_path"].set(askdirectory())

    def cancel_analysis(self):
        sys.exit("You have cancelled the analysis")

    def start_analysis(self):
        # Collect values
        for k, v in self.vars.items():
            self.vars[k] = v.get()

        # Group names
        self.vars["group_names"] = [g.strip() for g in self.vars["group_names"].split(",")]
        
        for ch in ["Ch1", "Ch2", "Ch3", "Ch4", "CCF"]:
            if self.vars[f"{ch}_smoothing"]:
                self.vars[f"{ch}_window"] = self.smoothing_widgets[ch][1].get()
                self.vars[f"{ch}_poly_order"] = self.smoothing_widgets[ch][2].get()
            else:
                self.vars[f"{ch}_window"] = None
                self.vars[f"{ch}_poly_order"] = None
        # Destroy widget
        self.destroy()