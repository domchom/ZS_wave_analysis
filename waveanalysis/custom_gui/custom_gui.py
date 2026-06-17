import csv
import glob
import json
import os
import subprocess
import sys as _sys
import time
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, scrolledtext, messagebox
from tkinter.filedialog import askdirectory

try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    _TkBase = TkinterDnD.Tk
    _DND_AVAILABLE = True
except ImportError:
    _TkBase = tk.Tk
    _DND_AVAILABLE = False

_CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".waveanalysis_config.json")


def _natural_sort_key(s):
    """Sort key that orders embedded numbers numerically: Bin 2 < Bin 10."""
    import re as _re
    return [int(t) if t.isdigit() else t.lower() for t in _re.split(r"(\d+)", s)]


def _load_config():
    try:
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_config(data):
    try:
        existing = _load_config()
        existing.update(data)
        with open(_CONFIG_PATH, "w") as f:
            json.dump(existing, f)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _GUIBase(_TkBase):
    """Shared widget helpers and actions used by all three analysis GUI classes."""

    _has_group_names = False
    _has_plot_flags = False

    # ---- widget helpers (accept a parent frame) ----

    @staticmethod
    def _add_entry(parent, row, col, var, label_text, width=5):
        entry = ttk.Entry(parent, width=width, textvariable=var)
        entry.grid(row=row, column=col, padx=2, pady=2, sticky="e")
        label = ttk.Label(parent, text=label_text)
        label.grid(row=row, column=col + 1, padx=(4, 8), pady=2, sticky="w")
        return entry, label

    @staticmethod
    def _add_check(parent, row, col, var, label_text):
        cb = ttk.Checkbutton(parent, variable=var)
        cb.grid(row=row, column=col, padx=2, pady=1, sticky="e")
        lbl = ttk.Label(parent, text=label_text)
        lbl.grid(row=row, column=col + 1, padx=(2, 8), pady=1, sticky="w")
        return cb, lbl

    def _build_smoothing(self, parent, default_poly=2):
        """Build smoothing channel rows inside *parent* frame."""
        self.smoothing_widgets = {}
        for i, ch in enumerate(["Ch1", "Ch2", "Ch3", "Ch4", "CCF"]):
            cb = ttk.Checkbutton(parent, variable=self.vars[f"{ch}_smoothing"])
            cb.grid(row=i, column=0, sticky="e")
            ttk.Label(parent, text=ch, width=4).grid(row=i, column=1, sticky="w")
            ttk.Label(parent, text="win").grid(row=i, column=2, padx=(6, 2), sticky="e")
            win = ttk.Spinbox(parent, from_=3, to=101, increment=2, width=4)
            win.set(11)
            win.grid(row=i, column=3, sticky="w")
            ttk.Label(parent, text="poly").grid(row=i, column=4, padx=(6, 2), sticky="e")
            poly = ttk.Spinbox(parent, from_=1, to=10, increment=1, width=3)
            poly.set(default_poly)
            poly.grid(row=i, column=5, sticky="w")
            self.smoothing_widgets[ch] = (cb, win, poly)

    def _build_edge_height_slider(self, parent):
        """Build a slider for rise/fall landmark height as a fraction of peak height."""
        value_label = ttk.Label(parent, width=4)

        def update_label(*_):
            value_label.configure(text=f"{int(round(self.vars['edge_height_fraction'].get() * 100))}%")

        # Keep the % label in sync whether the slider is dragged or the value is
        # restored from saved settings (which sets the variable directly).
        self.vars["edge_height_fraction"].trace_add("write", update_label)

        ttk.Label(parent, text="Rise/fall height").grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Scale(
            parent,
            from_=0.1,
            to=0.9,
            variable=self.vars["edge_height_fraction"],
            command=lambda _value: update_label(),
        ).grid(row=1, column=0, columnspan=2, sticky="ew", padx=(0, 4))
        value_label.grid(row=1, column=2, sticky="e")
        ttk.Label(parent, text="10%").grid(row=2, column=0, sticky="w")
        ttk.Label(parent, text="90%").grid(row=2, column=1, sticky="e")
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        update_label()

    # ---- log / progress / status area ----

    def _build_bottom(self, parent):
        """Build the status bar, progress bar, and log text inside *parent*."""

        # button row
        btn = ttk.Frame(parent)
        btn.pack(fill=tk.X, pady=(6, 4))
        self.start_button = ttk.Button(btn, text="Start Analysis", command=self.start_analysis)
        self.start_button.pack(side=tk.LEFT)
        ttk.Button(btn, text="Test File", command=self._test_first_file).pack(side=tk.LEFT, padx=4)
        self.stop_button = ttk.Button(btn, text="Stop", command=self.stop_analysis, state="disabled")
        self.stop_button.pack(side=tk.LEFT, padx=4)
        ttk.Button(btn, text="Load Results", command=self._load_existing_results).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn, text="Close Window", command=self.cancel_analysis).pack(side=tk.LEFT)
        self.elapsed_label = ttk.Label(btn, text="")
        self.elapsed_label.pack(side=tk.RIGHT)
        self._build_extra_buttons(btn)  # subclass hook

        # status + progress
        self.status_label = ttk.Label(parent, text="Status: Ready", font=("TkDefaultFont", 10, "bold"),
                                      anchor="w", justify="left")
        self.status_label.pack(fill=tk.X)

        # Wrap long status text (e.g. full filenames) to the available width
        # instead of clipping it. Guard against the relayout re-triggering us.
        self._status_wrap_w = None

        def _wrap_status(event):
            if self._status_wrap_w != event.width:
                self._status_wrap_w = event.width
                self.status_label.configure(wraplength=max(event.width - 4, 100))
        self.status_label.bind("<Configure>", _wrap_status)
        prog = ttk.Frame(parent)
        prog.pack(fill=tk.X, pady=(2, 4))
        self.progress_bar = ttk.Progressbar(prog, orient=tk.HORIZONTAL, mode="determinate")
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.progress_file_label = ttk.Label(prog, text="", width=12, anchor="e")
        self.progress_file_label.pack(side=tk.LEFT, padx=(6, 0))

        # log
        self.log_text = scrolledtext.ScrolledText(parent, height=10, state="disabled", wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=(2, 0))
        self.log_text.tag_configure("error", foreground="red")

        # timer state
        self._timer_start = None
        self._timer_after_id = None
        self._stop_requested = False

        # config
        cfg = _load_config()
        last_folder = cfg.get("last_folder", "")
        if last_folder and "folder_path" in self.vars:
            self.vars["folder_path"].set(last_folder)

        # drag-and-drop
        if _DND_AVAILABLE:
            self.drop_target_register(DND_FILES)
            self.dnd_bind("<<Drop>>", self._on_folder_drop)

    def _build_extra_buttons(self, parent):
        """Override in subclasses to add extra buttons to the button row."""
        pass

    # ---- settings persistence ----

    def _settings_key(self):
        """Config key namespaced by analysis type so each mode keeps its own
        settings (their fields and defaults differ)."""
        atype = "default"
        if "analysis_type" in getattr(self, "vars", {}):
            try:
                atype = self.vars["analysis_type"].get()
            except Exception:
                pass
        return f"settings_{atype}"

    def _collect_settings(self):
        """Snapshot every tk Variable plus the smoothing spinbox values."""
        data = {}
        for key, var in self.vars.items():
            try:
                data[key] = var.get()
            except Exception:
                pass
        sm = {}
        if hasattr(self, "smoothing_widgets"):
            for ch, (_cb, win, poly) in self.smoothing_widgets.items():
                try:
                    sm[ch] = {"window": win.get(), "poly": poly.get()}
                except Exception:
                    pass
        data["_smoothing_spinboxes"] = sm
        return data

    def _save_all_settings(self):
        """Persist all current settings for this analysis mode."""
        if not getattr(self, "vars", None):
            return
        try:
            _save_config({self._settings_key(): self._collect_settings()})
        except Exception:
            pass

    def _restore_settings(self):
        """Apply previously saved settings for this analysis mode, if any.

        Call at the end of __init__, after vars and smoothing widgets exist.
        """
        saved = _load_config().get(self._settings_key(), {})
        if not isinstance(saved, dict):
            return
        for key, value in saved.items():
            if key == "_smoothing_spinboxes":
                continue
            if key in self.vars:
                try:
                    self.vars[key].set(value)
                except Exception:
                    pass
        sm = saved.get("_smoothing_spinboxes", {})
        if isinstance(sm, dict) and hasattr(self, "smoothing_widgets"):
            for ch, vals in sm.items():
                if ch in self.smoothing_widgets and isinstance(vals, dict):
                    _cb, win, poly = self.smoothing_widgets[ch]
                    if "window" in vals:
                        win.set(vals["window"])
                    if "poly" in vals:
                        poly.set(vals["poly"])

    # ---- log helpers (unchanged logic) ----

    def log_message(self, msg):
        try:
            if not self.winfo_exists():
                return
        except tk.TclError:
            return
        stripped = msg.strip()
        if not stripped:
            self.after(0, lambda: self._do_append("\n"))
            return
        if set(stripped) <= {"*"} or stripped == "Processing files...":
            return

        def _append():
            try:
                if not self.winfo_exists():
                    return
            except tk.TclError:
                return
            self._clear_progress_marks()
            self._do_append(msg if msg.endswith("\n") else msg + "\n")
        self.after(0, _append)

    def update_progress_line(self, label, text):
        try:
            if not self.winfo_exists():
                return
        except tk.TclError:
            return

        def _update():
            try:
                if not self.winfo_exists():
                    return
            except tk.TclError:
                return
            safe = "".join(c if c.isalnum() else "_" for c in label)
            mark = f"_prog_{safe}"
            self.log_text.configure(state="normal")
            if mark in self.log_text.mark_names():
                idx = self.log_text.index(mark)
                self.log_text.delete(idx, f"{idx} lineend + 1c")
                self.log_text.insert(idx, text + "\n")
            else:
                self.log_text.mark_set(mark, tk.END)
                self.log_text.mark_gravity(mark, tk.LEFT)
                self.log_text.insert(tk.END, text + "\n")
            self.log_text.see(tk.END)
            self.log_text.configure(state="disabled")
        self.after(0, _update)

    def _do_append(self, text):
        self.log_text.configure(state="normal")
        if "ERROR" in text.upper():
            self.log_text.insert(tk.END, text, "error")
        else:
            self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def _clear_progress_marks(self):
        for m in list(self.log_text.mark_names()):
            if m.startswith("_prog_"):
                self.log_text.mark_unset(m)

    def clear_log(self):
        def _clear():
            try:
                if not self.winfo_exists():
                    return
            except tk.TclError:
                return
            self.log_text.configure(state="normal")
            self.log_text.delete("1.0", tk.END)
            self.log_text.configure(state="disabled")
        self.after(0, _clear)

    def set_status(self, status):
        try:
            if not self.winfo_exists():
                return
        except tk.TclError:
            return
        self.after(0, lambda: self.status_label.configure(text=f"Status: {status}"))

    # ---- timer ----

    def start_timer(self):
        self._timer_start = time.time()
        self._tick_timer()

    def stop_timer(self):
        if self._timer_after_id is not None:
            try:
                self.after_cancel(self._timer_after_id)
            except Exception:
                pass
            self._timer_after_id = None
        if self._timer_start is not None:
            self._set_elapsed(time.time() - self._timer_start)
            self._timer_start = None

    def _tick_timer(self):
        if self._timer_start is None:
            return
        self._set_elapsed(time.time() - self._timer_start)
        self._timer_after_id = self.after(1000, self._tick_timer)

    def _set_elapsed(self, secs):
        m, s = divmod(int(secs), 60)
        h, m = divmod(m, 60)
        txt = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
        try:
            if self.winfo_exists():
                self.elapsed_label.configure(text=f"Elapsed: {txt}")
        except tk.TclError:
            pass

    # ---- progress bar ----

    def update_file_progress(self, current, total):
        def _upd():
            try:
                if not self.winfo_exists():
                    return
            except tk.TclError:
                return
            self.progress_bar.configure(maximum=total, value=current)
            self.progress_file_label.configure(text=f"File {current}/{total}")
        self.after(0, _upd)

    def reset_progress(self):
        def _rst():
            try:
                if not self.winfo_exists():
                    return
            except tk.TclError:
                return
            self.progress_bar.configure(value=0)
            self.progress_file_label.configure(text="")
            self.elapsed_label.configure(text="")
        self.after(0, _rst)

    # ---- validation ----

    def _validate_inputs(self):
        errors = []
        p = self._resolved_params
        folder = p.get("folder_path", "")
        if not folder:
            errors.append("No folder selected")
        elif not os.path.isdir(folder):
            errors.append(f"Folder does not exist: {folder}")
        else:
            tifs = [f for f in os.listdir(folder) if f.endswith(".tif") and not f.startswith(".")]
            if not tifs:
                errors.append("No .tif files found in the selected folder")
        for key in ("box_size", "bin_shift", "line_width", "subframe_size", "subframe_roll"):
            if key in p and isinstance(p[key], (int, float)) and p[key] <= 0:
                errors.append(f"{key.replace('_',' ').title()} must be > 0")
        for key in ("acf_peak_thresh", "ccf_peak_thresh"):
            if key in p and not (0 < p[key] <= 1):
                errors.append(f"{key.replace('_',' ').title()} must be between 0 and 1")
        return errors

    # ---- results ----

    def show_results_buttons(self, results_path):
        self._results_path = results_path
        if hasattr(self, "_results_frame") and self._results_frame:
            self._results_frame.destroy()
        self._results_frame = ttk.Frame(self._bottom)
        self._results_frame.pack(fill=tk.X, pady=(4, 0))
        for text, cmd in [
            ("Open Results", self._open_results_folder),
            ("View Plots", self._open_plot_viewer),
            ("View Summary", self._open_summary_viewer),
            ("Re-run on New Folder", self._rerun_different_folder),
        ]:
            ttk.Button(self._results_frame, text=text, command=cmd).pack(side=tk.LEFT, padx=(0, 6))

    def _hide_results_buttons(self):
        if hasattr(self, "_results_frame") and self._results_frame:
            self._results_frame.destroy()
            self._results_frame = None

    def _open_results_folder(self):
        p = self._results_path
        if _sys.platform == "darwin":
            subprocess.Popen(["open", p])
        elif _sys.platform == "win32":
            os.startfile(p)
        else:
            subprocess.Popen(["xdg-open", p])

    def _open_plot_viewer(self):
        _PlotViewer(self, self._results_path)

    def _open_summary_viewer(self):
        _SummaryViewer(self, self._results_path)

    def _load_existing_results(self):
        selected = askdirectory(title="Select analysis results folder")
        if not selected:
            return

        results_path = self._resolve_results_path(selected)
        if results_path is None:
            self.log_message(
                "ERROR: No analysis results found. Select a 0_signalProcessing-* folder "
                "or a folder containing one."
            )
            self.set_status("No existing results found")
            return

        self.show_results_buttons(results_path)
        self.set_status(f"Loaded existing results: {os.path.basename(results_path)}")
        self.log_message(f"Loaded existing results folder: {results_path}")

    @staticmethod
    def _resolve_results_path(path):
        if not os.path.isdir(path):
            return None

        has_outputs = (
            glob.glob(os.path.join(path, "!*_summary.csv"))
            or glob.glob(os.path.join(path, "**", "*.png"), recursive=True)
        )
        if os.path.basename(path).startswith("0_signalProcessing-") or has_outputs:
            return path

        results_dirs = sorted(
            glob.glob(os.path.join(path, "0_signalProcessing-*")),
            key=os.path.getmtime,
        )
        return results_dirs[-1] if results_dirs else None

    def _rerun_different_folder(self):
        new = askdirectory()
        if not new:
            return
        self.vars["folder_path"].set(new)
        self.start_analysis()

    def _on_folder_drop(self, event):
        path = event.data.strip().strip("{}")
        if os.path.isdir(path):
            self.vars["folder_path"].set(path)

    # ---- actions ----

    def get_folder_path(self):
        self.vars["folder_path"].set(askdirectory())

    def _test_first_file(self):
        """Run analysis on a single .tif file from the folder (quick test)."""
        if self._is_running:
            return
        folder = self.vars["folder_path"].get()
        if not folder or not os.path.isdir(folder):
            self.clear_log()
            self.log_message("ERROR: Select a valid folder first")
            return
        tifs = sorted(
            (f for f in os.listdir(folder) if f.endswith(".tif") and not f.startswith(".")),
            key=_natural_sort_key,
        )
        if not tifs:
            self.clear_log()
            self.log_message("ERROR: No .tif files found in the selected folder")
            return

        if len(tifs) == 1:
            chosen = tifs[0]
        else:
            chosen = _FilePickerDialog(self, tifs).result
            if chosen is None:
                return

        import tempfile
        tmp = tempfile.mkdtemp(prefix="wavetest_")
        src = os.path.join(folder, chosen)
        dst = os.path.join(tmp, chosen)
        try:
            os.symlink(src, dst)
        except OSError:
            import shutil
            shutil.copy2(src, dst)

        self.log_message(f"Test mode: running on {chosen} only")
        self._test_folder_path_once = tmp
        self._ignore_group_names_once = self._has_group_names
        self.start_analysis()

    def _finalize_vars(self):
        """Release this GUI's tk Variables on the main thread while the Tcl
        interpreter is still alive.

        When the program switches modes it destroys this window and builds a
        new Tk root, but the old Variable objects linger in reference cycles
        until a GC sweep frees them. If that sweep happens inside the next
        GUI's background analysis thread, each Variable.__del__ calls into Tcl
        off the main thread and raises "main thread is not in main loop"
        (and can wedge the interpreter). Clearing them here forces their
        finalizers to run now, on the main thread, before we navigate away.
        """
        import gc
        if hasattr(self, "vars"):
            self.vars.clear()
        if hasattr(self, "smoothing_widgets"):
            self.smoothing_widgets.clear()
        gc.collect()

    def cancel_analysis(self):
        self._save_all_settings()
        self.destroy()

    def stop_analysis(self):
        self._stop_requested = True
        self.log_message("Stop requested -- will stop after the current file finishes...")
        self.set_status("Stopping...")

    def _on_close(self):
        if self._is_running:
            self._stop_requested = True
        self._save_all_settings()
        self.destroy()

    def start_analysis(self):
        if self._is_running:
            return
        try:
            self._resolved_params = {}
            for k, v in self.vars.items():
                self._resolved_params[k] = v.get()
            original_folder_path = self._resolved_params.get("folder_path", "")
            test_folder_path = getattr(self, "_test_folder_path_once", None)
            if test_folder_path is not None:
                self._resolved_params["folder_path"] = test_folder_path
            if self._has_group_names:
                self._resolved_params["group_names"] = [
                    g.strip() for g in self._resolved_params["group_names"].split(",")
                ]
                if getattr(self, "_ignore_group_names_once", False):
                    self._resolved_params["group_names"] = [""]
            self._test_folder_path_once = None
            self._ignore_group_names_once = False
            sm = {}
            for ch in ["Ch1", "Ch2", "Ch3", "Ch4", "CCF"]:
                if self._resolved_params[f"{ch}_smoothing"]:
                    sm[ch] = {
                        "window": int(self.smoothing_widgets[ch][1].get()),
                        "poly_order": int(self.smoothing_widgets[ch][2].get()),
                    }
                else:
                    sm[ch] = None
            self._resolved_params["smoothing_params"] = sm
            if self._has_plot_flags:
                self._resolved_params["plot_flags"] = {
                    k: self._resolved_params[k]
                    for k in ("plot_summary_ACFs", "plot_summary_CCFs", "plot_summary_peaks",
                              "plot_indv_ACFs", "plot_indv_CCFs", "plot_indv_peaks",
                              "plot_heatmaps", "plot_landmark_shifts", "plot_indv_landmark_shifts",
                              "plot_fts", "dark_plots")
                    if k in self._resolved_params
                }
            errs = self._validate_inputs()
            if errs:
                self.clear_log()
                for e in errs:
                    self.log_message(f"ERROR: {e}")
                self.set_status("Fix errors above and try again")
                return
            # Don't persist anything for a quick test run -- in particular avoid
            # remembering the temporary single-file folder it runs from.
            if test_folder_path is None:
                _save_config({"last_folder": original_folder_path})
                self._save_all_settings()
            self._hide_results_buttons()
            self._is_running = True
            self._stop_requested = False
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            if self.on_start:
                self.on_start()
        except Exception as e:
            self._test_folder_path_once = None
            self._ignore_group_names_once = False
            self._is_running = False
            self.start_button.configure(state="normal")
            self.log_message(f"ERROR: Failed to start: {e}")
            self.set_status("Error")


# ---------------------------------------------------------------------------
# Standard analysis GUI
# ---------------------------------------------------------------------------

class BaseGUI(_GUIBase):
    _has_group_names = True
    _has_plot_flags = True

    def __init__(self):
        super().__init__()
        self.title("Wave Analysis")
        self.vars = {
            "analysis_type": tk.StringVar(value="standard"),
            "Ch1_name": tk.StringVar(value=""),
            "Ch2_name": tk.StringVar(value=""),
            "Ch3_name": tk.StringVar(value=""),
            "Ch4_name": tk.StringVar(value=""),
            "box_size": tk.IntVar(value=20),
            "bin_shift": tk.IntVar(value=20),
            "small_shifts_correction": tk.BooleanVar(value=True),
            "plot_summary_ACFs": tk.BooleanVar(value=True),
            "plot_summary_CCFs": tk.BooleanVar(value=True),
            "plot_summary_peaks": tk.BooleanVar(value=True),
            "plot_indv_ACFs": tk.BooleanVar(value=False),
            "plot_indv_CCFs": tk.BooleanVar(value=False),
            "plot_indv_peaks": tk.BooleanVar(value=False),
            "plot_heatmaps": tk.BooleanVar(value=False),
            "plot_landmark_shifts": tk.BooleanVar(value=False),
            "plot_indv_landmark_shifts": tk.BooleanVar(value=False),
            "plot_fts": tk.BooleanVar(value=False),
            "dark_plots": tk.BooleanVar(value=True),
            "acf_peak_thresh": tk.DoubleVar(value=0.1),
            "ccf_peak_thresh": tk.DoubleVar(value=0.1),
            "peak_prominence_fraction": tk.DoubleVar(value=0.1),
            "edge_height_fraction": tk.DoubleVar(value=0.5),
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

        root = ttk.Frame(self, padding=8)
        root.pack(fill=tk.BOTH, expand=True)

        # ---- top: two columns ----
        top = ttk.Frame(root)
        top.pack(fill=tk.X)

        # left: analysis options
        opts = ttk.LabelFrame(top, text="Analysis Options", padding=6)
        opts.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))

        pf = ttk.Frame(opts)
        pf.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        pf.columnconfigure(0, weight=1)
        ttk.Entry(pf, textvariable=self.vars["folder_path"]).grid(row=0, column=0, sticky="ew")
        ttk.Button(pf, text="Browse", command=self.get_folder_path, width=7).grid(row=0, column=1, padx=(4, 0))

        self._add_entry(opts, 1, 0, self.vars["group_names"], "Group names", width=14)
        self._add_entry(opts, 2, 0, self.vars["box_size"], "Box size (px)")
        self._add_entry(opts, 3, 0, self.vars["bin_shift"], "Box shift (px)")
        self._add_entry(opts, 4, 0, self.vars["acf_peak_thresh"], "ACF peak thresh")
        self._add_entry(opts, 5, 0, self.vars["ccf_peak_thresh"], "CCF peak thresh")
        self._add_entry(opts, 6, 0, self.vars["peak_prominence_fraction"], "Peak prom. frac.")
        self._add_check(opts, 7, 0, self.vars["small_shifts_correction"], "Small shifts correction")

        # middle column: smoothing + channel names
        middle = ttk.Frame(top)
        middle.pack(side=tk.LEFT, fill=tk.BOTH, padx=(4, 0))

        # smoothing
        sm = ttk.LabelFrame(middle, text="Smoothing", padding=4)
        sm.pack(fill=tk.X, pady=(0, 4))
        self._build_smoothing(sm, default_poly=2)
        self._add_check(sm, 5, 0, self.vars["smoothing"], "Enable smoothing")

        # channel display names (used in plot labels only; blank = default ChN)
        cn = ttk.LabelFrame(middle, text="Channel Names (optional)", padding=4)
        cn.pack(fill=tk.X, pady=(0, 4))
        self._add_entry(cn, 0, 0, self.vars["Ch1_name"], "Ch1", width=10)
        self._add_entry(cn, 1, 0, self.vars["Ch2_name"], "Ch2", width=10)
        self._add_entry(cn, 0, 2, self.vars["Ch3_name"], "Ch3", width=10)
        self._add_entry(cn, 1, 2, self.vars["Ch4_name"], "Ch4", width=10)

        # right column: edge-height slider + plot options
        right = ttk.Frame(top)
        right.pack(side=tk.LEFT, fill=tk.BOTH, padx=(4, 0))

        edge = ttk.LabelFrame(right, text="Landmark Edge Height", padding=4)
        edge.pack(fill=tk.X, pady=(0, 4))
        self._build_edge_height_slider(edge)

        # plot options
        pl = ttk.LabelFrame(right, text="Plot Options", padding=4)
        pl.pack(fill=tk.X)
        self._add_check(pl, 0, 0, self.vars["plot_summary_ACFs"], "Summary ACFs")
        self._add_check(pl, 1, 0, self.vars["plot_summary_CCFs"], "Summary CCFs")
        self._add_check(pl, 2, 0, self.vars["plot_summary_peaks"], "Summary peaks")
        self._add_check(pl, 0, 2, self.vars["plot_indv_ACFs"], "Indv ACFs")
        self._add_check(pl, 1, 2, self.vars["plot_indv_CCFs"], "Indv CCFs")
        self._add_check(pl, 2, 2, self.vars["plot_indv_peaks"], "Indv peaks")
        self._add_check(pl, 0, 4, self.vars["dark_plots"], "Dark plots")
        self._add_check(pl, 1, 4, self.vars["plot_heatmaps"], "Heatmaps")
        self._add_check(pl, 2, 4, self.vars["plot_fts"], "Fourier transforms")
        self._add_check(pl, 3, 0, self.vars["plot_landmark_shifts"], "Summary landmark")
        self._add_check(pl, 3, 2, self.vars["plot_indv_landmark_shifts"], "Indv landmark")

        # ---- separator ----
        ttk.Separator(root, orient="horizontal").pack(fill=tk.X, pady=6)

        # ---- bottom: buttons + progress + log ----
        self._bottom = ttk.Frame(root)
        self._bottom.pack(fill=tk.BOTH, expand=True)
        self._build_bottom(self._bottom)

        self.on_start = None
        self._is_running = False
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._restore_settings()

    def _build_extra_buttons(self, parent):
        ttk.Button(parent, text="Kymograph", command=self.launch_kymograph_analysis).pack(side=tk.RIGHT)
        ttk.Button(parent, text="Rolling", command=self.launch_rolling_analysis).pack(side=tk.RIGHT, padx=(0, 4))

    def launch_rolling_analysis(self):
        self.rolling = True
        self.kymograph = False
        self._save_all_settings()
        self._finalize_vars()
        self.destroy()

    def launch_kymograph_analysis(self):
        self.kymograph = True
        self.rolling = False
        self._save_all_settings()
        self._finalize_vars()
        self.destroy()


# ---------------------------------------------------------------------------
# Rolling analysis GUI
# ---------------------------------------------------------------------------

class RollingGUI(_GUIBase):

    def __init__(self):
        super().__init__()
        self.title("Wave Analysis — Rolling")
        self.vars = {
            "analysis_type": tk.StringVar(value="rolling"),
            "Ch1_name": tk.StringVar(value=""),
            "Ch2_name": tk.StringVar(value=""),
            "Ch3_name": tk.StringVar(value=""),
            "Ch4_name": tk.StringVar(value=""),
            "box_size": tk.IntVar(value=20),
            "bin_shift": tk.IntVar(value=20),
            "subframe_size": tk.IntVar(value=50),
            "subframe_roll": tk.IntVar(value=5),
            "small_shifts_correction": tk.BooleanVar(value=True),
            "plot_subframe_ACFs": tk.BooleanVar(value=True),
            "plot_subframe_CCFs": tk.BooleanVar(value=True),
            "plot_subframe_peaks": tk.BooleanVar(value=True),
            "dark_plots": tk.BooleanVar(value=False),
            "acf_peak_thresh": tk.DoubleVar(value=0.1),
            "ccf_peak_thresh": tk.DoubleVar(value=0.1),
            "peak_prominence_fraction": tk.DoubleVar(value=0.1),
            "smoothing": tk.BooleanVar(value=True),
            "folder_path": tk.StringVar(value=""),
            "Ch1_smoothing": tk.BooleanVar(value=True),
            "Ch2_smoothing": tk.BooleanVar(value=True),
            "Ch3_smoothing": tk.BooleanVar(value=True),
            "Ch4_smoothing": tk.BooleanVar(value=True),
            "CCF_smoothing": tk.BooleanVar(value=True),
            "edge_height_fraction": tk.DoubleVar(value=0.5),
            "injection_frame": tk.IntVar(value=0),
            "injection_ch1": tk.BooleanVar(value=False),
            "injection_ch2": tk.BooleanVar(value=False),
        }
        self.kymograph = False
        self._back_to_standard = False

        root = ttk.Frame(self, padding=8)
        root.pack(fill=tk.BOTH, expand=True)
        top = ttk.Frame(root)
        top.pack(fill=tk.X)

        opts = ttk.LabelFrame(top, text="Analysis Options", padding=6)
        opts.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))

        pf = ttk.Frame(opts)
        pf.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        pf.columnconfigure(0, weight=1)
        ttk.Entry(pf, textvariable=self.vars["folder_path"]).grid(row=0, column=0, sticky="ew")
        ttk.Button(pf, text="Browse", command=self.get_folder_path, width=7).grid(row=0, column=1, padx=(4, 0))

        self._add_entry(opts, 1, 0, self.vars["box_size"], "Box size (px)")
        self._add_entry(opts, 2, 0, self.vars["bin_shift"], "Box shift (px)")
        self._add_entry(opts, 3, 0, self.vars["subframe_size"], "Subframe size")
        self._add_entry(opts, 4, 0, self.vars["subframe_roll"], "Subframe roll")
        self._add_entry(opts, 5, 0, self.vars["acf_peak_thresh"], "ACF peak thresh")
        self._add_entry(opts, 6, 0, self.vars["ccf_peak_thresh"], "CCF peak thresh")
        self._add_entry(opts, 7, 0, self.vars["peak_prominence_fraction"], "Peak prom. frac.")
        self._add_check(opts, 8, 0, self.vars["small_shifts_correction"], "Small shifts correction")
        self._add_check(opts, 9, 0, self.vars["dark_plots"], "Dark plots")

        sm = ttk.LabelFrame(top, text="Smoothing", padding=4)
        sm.pack(side=tk.LEFT, fill=tk.BOTH, padx=(4, 0))
        self._build_smoothing(sm, default_poly=3)

        cn = ttk.LabelFrame(top, text="Channel Names (optional)", padding=4)
        cn.pack(side=tk.LEFT, fill=tk.BOTH, padx=(4, 0))
        self._add_entry(cn, 0, 0, self.vars["Ch1_name"], "Ch1", width=10)
        self._add_entry(cn, 1, 0, self.vars["Ch2_name"], "Ch2", width=10)
        self._add_entry(cn, 2, 0, self.vars["Ch3_name"], "Ch3", width=10)
        self._add_entry(cn, 3, 0, self.vars["Ch4_name"], "Ch4", width=10)

        edge = ttk.LabelFrame(top, text="Landmark Edge Height", padding=4)
        edge.pack(side=tk.LEFT, fill=tk.BOTH, padx=(4, 0))
        self._build_edge_height_slider(edge)

        # live-injection options (leave injection frame at 0 to ignore)
        inj = ttk.LabelFrame(top, text="Live Injection (optional)", padding=4)
        inj.pack(side=tk.LEFT, fill=tk.BOTH, padx=(4, 0))
        self._add_entry(inj, 0, 0, self.vars["injection_frame"], "Injection frame", width=6)
        self._add_check(inj, 1, 0, self.vars["injection_ch1"], "Injection signal Ch1")
        self._add_check(inj, 2, 0, self.vars["injection_ch2"], "Injection signal Ch2")

        ttk.Separator(root, orient="horizontal").pack(fill=tk.X, pady=6)
        self._bottom = ttk.Frame(root)
        self._bottom.pack(fill=tk.BOTH, expand=True)
        self._build_bottom(self._bottom)

        self.on_start = None
        self._is_running = False
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._restore_settings()

    def _build_extra_buttons(self, parent):
        ttk.Button(parent, text="Back to Standard", command=self._go_back).pack(side=tk.RIGHT)

    def _go_back(self):
        self._back_to_standard = True
        self._save_all_settings()
        self._finalize_vars()
        self.destroy()


# ---------------------------------------------------------------------------
# Kymograph analysis GUI
# ---------------------------------------------------------------------------

class KymographGUI(_GUIBase):
    _has_group_names = True
    _has_plot_flags = True

    def __init__(self):
        super().__init__()
        self.title("Wave Analysis — Kymograph")
        self.vars = {
            "analysis_type": tk.StringVar(value="kymograph"),
            "Ch1_name": tk.StringVar(value=""),
            "Ch2_name": tk.StringVar(value=""),
            "Ch3_name": tk.StringVar(value=""),
            "Ch4_name": tk.StringVar(value=""),
            "line_width": tk.IntVar(value=5),
            "bin_shift": tk.IntVar(value=5),
            "small_shifts_correction": tk.BooleanVar(value=False),
            "plot_summary_ACFs": tk.BooleanVar(value=True),
            "plot_summary_CCFs": tk.BooleanVar(value=True),
            "plot_summary_peaks": tk.BooleanVar(value=True),
            "plot_indv_ACFs": tk.BooleanVar(value=False),
            "plot_indv_CCFs": tk.BooleanVar(value=False),
            "plot_indv_peaks": tk.BooleanVar(value=False),
            "plot_heatmaps": tk.BooleanVar(value=False),
            "plot_landmark_shifts": tk.BooleanVar(value=False),
            "plot_indv_landmark_shifts": tk.BooleanVar(value=False),
            "plot_fts": tk.BooleanVar(value=False),
            "dark_plots": tk.BooleanVar(value=False),
            "acf_peak_thresh": tk.DoubleVar(value=0.1),
            "ccf_peak_thresh": tk.DoubleVar(value=0.1),
            "peak_prominence_fraction": tk.DoubleVar(value=0.1),
            "edge_height_fraction": tk.DoubleVar(value=0.5),
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
        self._back_to_standard = False

        root = ttk.Frame(self, padding=8)
        root.pack(fill=tk.BOTH, expand=True)
        top = ttk.Frame(root)
        top.pack(fill=tk.X)

        # left: analysis options
        opts = ttk.LabelFrame(top, text="Analysis Options", padding=6)
        opts.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))

        pf = ttk.Frame(opts)
        pf.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        pf.columnconfigure(0, weight=1)
        ttk.Entry(pf, textvariable=self.vars["folder_path"]).grid(row=0, column=0, sticky="ew")
        ttk.Button(pf, text="Browse", command=self.get_folder_path, width=7).grid(row=0, column=1, padx=(4, 0))

        self._add_entry(opts, 1, 0, self.vars["group_names"], "Group names", width=14)
        self._add_entry(opts, 2, 0, self.vars["line_width"], "Line width (px)")
        self._add_entry(opts, 3, 0, self.vars["bin_shift"], "Line shift (px)")
        self._add_entry(opts, 4, 0, self.vars["acf_peak_thresh"], "ACF peak thresh")
        self._add_entry(opts, 5, 0, self.vars["ccf_peak_thresh"], "CCF peak thresh")
        self._add_entry(opts, 6, 0, self.vars["peak_prominence_fraction"], "Peak prom. frac.")
        self._add_check(opts, 7, 0, self.vars["small_shifts_correction"], "Small shifts correction")

        # middle column: smoothing + channel names
        middle = ttk.Frame(top)
        middle.pack(side=tk.LEFT, fill=tk.BOTH, padx=(4, 0))

        sm = ttk.LabelFrame(middle, text="Smoothing", padding=4)
        sm.pack(fill=tk.X, pady=(0, 4))
        self._build_smoothing(sm, default_poly=3)
        self._add_check(sm, 5, 0, self.vars["smoothing"], "Enable smoothing")

        # channel display names (used in plot labels only; blank = default ChN)
        cn = ttk.LabelFrame(middle, text="Channel Names (optional)", padding=4)
        cn.pack(fill=tk.X, pady=(0, 4))
        self._add_entry(cn, 0, 0, self.vars["Ch1_name"], "Ch1", width=10)
        self._add_entry(cn, 1, 0, self.vars["Ch2_name"], "Ch2", width=10)
        self._add_entry(cn, 0, 2, self.vars["Ch3_name"], "Ch3", width=10)
        self._add_entry(cn, 1, 2, self.vars["Ch4_name"], "Ch4", width=10)

        # right column: edge-height slider + plot options
        right = ttk.Frame(top)
        right.pack(side=tk.LEFT, fill=tk.BOTH, padx=(4, 0))

        edge = ttk.LabelFrame(right, text="Landmark Edge Height", padding=4)
        edge.pack(fill=tk.X, pady=(0, 4))
        self._build_edge_height_slider(edge)

        pl = ttk.LabelFrame(right, text="Plot Options", padding=4)
        pl.pack(fill=tk.X)
        self._add_check(pl, 0, 0, self.vars["plot_summary_ACFs"], "Summary ACFs")
        self._add_check(pl, 1, 0, self.vars["plot_summary_CCFs"], "Summary CCFs")
        self._add_check(pl, 2, 0, self.vars["plot_summary_peaks"], "Summary peaks")
        self._add_check(pl, 0, 2, self.vars["plot_indv_ACFs"], "Indv ACFs")
        self._add_check(pl, 1, 2, self.vars["plot_indv_CCFs"], "Indv CCFs")
        self._add_check(pl, 2, 2, self.vars["plot_indv_peaks"], "Indv peaks")
        self._add_check(pl, 0, 4, self.vars["dark_plots"], "Dark plots")
        self._add_check(pl, 1, 4, self.vars["plot_heatmaps"], "Heatmaps")
        self._add_check(pl, 2, 4, self.vars["plot_fts"], "Fourier transforms")
        self._add_check(pl, 3, 0, self.vars["plot_landmark_shifts"], "Summary landmark")
        self._add_check(pl, 3, 2, self.vars["plot_indv_landmark_shifts"], "Indv landmark")

        ttk.Separator(root, orient="horizontal").pack(fill=tk.X, pady=6)
        self._bottom = ttk.Frame(root)
        self._bottom.pack(fill=tk.BOTH, expand=True)
        self._build_bottom(self._bottom)

        self.on_start = None
        self._is_running = False
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._restore_settings()

    def _build_extra_buttons(self, parent):
        ttk.Button(parent, text="Back to Standard", command=self._go_back).pack(side=tk.RIGHT)

    def _go_back(self):
        self._back_to_standard = True
        self._save_all_settings()
        self._finalize_vars()
        self.destroy()


# ---------------------------------------------------------------------------
# File picker dialog
# ---------------------------------------------------------------------------

class _FilePickerDialog(tk.Toplevel):
    """Modal dialog that lets the user pick one file from a list."""

    def __init__(self, parent, file_list):
        super().__init__(parent)
        self.title("Select a file to test")
        self.result = None
        self.transient(parent)
        self.grab_set()

        ttk.Label(self, text="Choose a .tif file:").pack(padx=10, pady=(10, 4), anchor="w")

        frame = ttk.Frame(self)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 6))
        sb = ttk.Scrollbar(frame, orient=tk.VERTICAL)
        self.listbox = tk.Listbox(frame, yscrollcommand=sb.set, height=12, activestyle="dotbox")
        sb.configure(command=self.listbox.yview)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        for f in file_list:
            self.listbox.insert(tk.END, f)
        self._fit_listbox_to_filenames(file_list)
        self.listbox.selection_set(0)
        self.listbox.bind("<Double-1>", lambda e: self._ok())

        btn = ttk.Frame(self)
        btn.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Button(btn, text="OK", command=self._ok).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn, text="Cancel", command=self._cancel).pack(side=tk.LEFT)

        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.wait_window()

    def _fit_listbox_to_filenames(self, file_list):
        font = tkfont.nametofont(self.listbox.cget("font"))
        longest_px = max(font.measure(f) for f in file_list)
        char_px = max(font.measure("0"), 1)
        padding_px = 36
        available_px = max(self.winfo_screenwidth() - 120, 300)
        width_chars = (min(longest_px + padding_px, available_px) + char_px - 1) // char_px
        self.listbox.configure(width=max(20, int(width_chars)))

    def _ok(self):
        sel = self.listbox.curselection()
        if sel:
            self.result = self.listbox.get(sel[0])
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()


# ---------------------------------------------------------------------------
# Plot viewer
# ---------------------------------------------------------------------------

class _PlotViewer(tk.Toplevel):

    def __init__(self, parent, results_path):
        super().__init__(parent)
        self.title("Plot Viewer")
        self.geometry("1100x750")
        self.results_path = results_path
        self.default_dark_plots = self._infer_dark_plots(parent, results_path)
        self._summary_df = None
        self._metric_labels = []
        self._metric_lookup = {}
        self._load_summary_metrics()

        self.image_files = self._scan_image_files()

        self.current_index = 0
        self._photo = None

        pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        tree_frame = ttk.Frame(pane)
        pane.add(tree_frame, weight=1)
        self.tree = ttk.Treeview(tree_frame, show="tree")
        tsb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._populate_tree()
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        right = ttk.Frame(pane)
        pane.add(right, weight=3)
        self.canvas = tk.Canvas(right, bg="#2b2b2b", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self._show_current())

        nav = ttk.Frame(self)
        nav.pack(fill=tk.X, padx=10, pady=(0, 6))
        ttk.Button(nav, text="< Prev", command=self._prev).pack(side=tk.LEFT)
        ttk.Button(nav, text="Next >", command=self._next).pack(side=tk.LEFT, padx=4)
        ttk.Button(nav, text="Group Correlations", command=self._make_group_correlations).pack(side=tk.LEFT, padx=(0, 4))
        self.file_label = ttk.Label(nav, text="", wraplength=650)
        self.file_label.pack(side=tk.LEFT, padx=10)
        self.count_label = ttk.Label(nav, text=f"{len(self.image_files)} plots")
        self.count_label.pack(side=tk.RIGHT)

        self.bind("<Left>", lambda e: self._prev())
        self.bind("<Right>", lambda e: self._next())
        if self.image_files:
            self.after(100, lambda: self._show_image(0))
        else:
            self.after(100, self._show_empty)

        scatter = ttk.LabelFrame(self, text="Group Scatter", padding=4)
        scatter.pack(fill=tk.X, padx=10, pady=(0, 6))
        scatter.columnconfigure(1, weight=1)
        scatter.columnconfigure(3, weight=1)

        ttk.Label(scatter, text="X").grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.scatter_x_var = tk.StringVar(value=self._metric_labels[0] if self._metric_labels else "")
        self.scatter_x_combo = ttk.Combobox(
            scatter,
            textvariable=self.scatter_x_var,
            values=self._metric_labels,
            state="readonly",
            width=28,
        )
        self.scatter_x_combo.grid(row=0, column=1, sticky="ew", padx=(0, 8))

        ttk.Label(scatter, text="Y").grid(row=0, column=2, sticky="w", padx=(0, 4))
        self.scatter_y_var = tk.StringVar(value=self._default_scatter_y_metric())
        self.scatter_y_combo = ttk.Combobox(
            scatter,
            textvariable=self.scatter_y_var,
            values=self._metric_labels,
            state="readonly",
            width=28,
        )
        self.scatter_y_combo.grid(row=0, column=3, sticky="ew", padx=(0, 8))

        self.scatter_dark_var = tk.BooleanVar(value=self.default_dark_plots)
        ttk.Checkbutton(scatter, variable=self.scatter_dark_var, text="Dark").grid(row=0, column=4, padx=(0, 8))
        ttk.Button(scatter, text="Compute Plot", command=self._make_group_scatter).grid(row=0, column=5)
        self.scatter_status = ttk.Label(scatter, text="", wraplength=900)
        self.scatter_status.grid(row=1, column=0, columnspan=6, sticky="w", pady=(4, 0))

        if not self._metric_labels:
            self.scatter_x_combo.configure(state="disabled")
            self.scatter_y_combo.configure(state="disabled")
            self.scatter_status.configure(
                text="No summary metrics with numeric mean values were found.",
                foreground="red",
            )

    def _scan_image_files(self):
        image_files = []
        for root, dirs, files in os.walk(self.results_path):
            dirs[:] = sorted([d for d in dirs if not d.startswith(".")], key=_natural_sort_key)
            for f in sorted(files, key=_natural_sort_key):
                if f.lower().endswith(".png") and not f.startswith("."):
                    image_files.append(os.path.join(root, f))
        return image_files

    @staticmethod
    def _infer_dark_plots(parent, results_path):
        log_files = sorted(glob.glob(os.path.join(results_path, "!log-*.txt")))
        if log_files:
            try:
                with open(log_files[-1]) as f:
                    for line in f:
                        if line.startswith("Dark Plots:"):
                            return line.split(":", 1)[1].strip().lower() == "true"
            except OSError:
                pass

        try:
            if "dark_plots" in getattr(parent, "vars", {}):
                return bool(parent.vars["dark_plots"].get())
        except Exception:
            pass
        return False

    def _load_summary_metrics(self):
        csv_files = glob.glob(os.path.join(self.results_path, "!*_summary.csv"))
        if not csv_files:
            return

        import pandas as pd
        from waveanalysis.housekeeping.housekeeping_functions import relabel_metric_text

        self._summary_df = pd.read_csv(sorted(csv_files)[-1])
        metric_cols = [
            col for col in self._summary_df.columns
            if 'Mean' in col and pd.to_numeric(self._summary_df[col], errors='coerce').notna().any()
        ]
        metric_cols = sorted(metric_cols, key=self._metric_sort_key)
        for col in metric_cols:
            label = relabel_metric_text(col)
            self._metric_labels.append(label)
            self._metric_lookup[label] = col

    def _default_scatter_y_metric(self):
        if not self._metric_labels:
            return ""
        if len(self._metric_labels) == 1:
            return self._metric_labels[0]
        for label in self._metric_labels[1:]:
            col = self._metric_lookup[label]
            if 'Shift' in col:
                return label
        return self._metric_labels[1]

    @staticmethod
    def _metric_sort_key(col):
        metric_order = [
            'Period',
            'Peak Amp', 'Peak Rel Amp', 'Peak Max', 'Peak Min', 'Peak Width',
            'Peak Area', 'Peak Offset',
            'Rise Duration', 'Fall Duration', 'Rise minus Fall Duration',
            'Rising Slope', 'Falling Slope', 'Max Rising Slope', 'Max Falling Slope',
            'Rising/Falling Slope Ratio',
            '% Phase Shift', 'Peak Shift', 'Rise Shift', 'Fall Shift', 'Shift',
            'Rise-Peak Diff', 'Fall-Peak Diff',
        ]
        for idx, metric in enumerate(metric_order):
            if metric in col:
                return idx, col
        return len(metric_order), col

    def _populate_tree(self):
        added = {}
        for i, fp in enumerate(self.image_files):
            rel = os.path.relpath(fp, self.results_path)
            parts = rel.split(os.sep)
            parent = ""
            for j, part in enumerate(parts[:-1]):
                pk = os.sep.join(parts[:j + 1])
                if pk not in added:
                    added[pk] = self.tree.insert(parent, "end", text=part, open=True)
                parent = added[pk]
            self.tree.insert(parent, "end", text=parts[-1], values=(i,))

    def _on_tree_select(self, _e):
        sel = self.tree.selection()
        if sel:
            vals = self.tree.item(sel[0], "values")
            if vals:
                self._show_image(int(vals[0]))

    def _show_image(self, idx):
        self.current_index = idx % len(self.image_files)
        fp = self.image_files[self.current_index]
        rel = os.path.relpath(fp, self.results_path)
        self.file_label.configure(text=f"[{self.current_index+1}/{len(self.image_files)}]  {rel}")
        try:
            from PIL import Image, ImageTk
            img = Image.open(fp)
            cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
            if cw < 50 or ch < 50:
                cw, ch = 700, 600
            img.thumbnail((cw - 10, ch - 10), Image.LANCZOS)
            self._photo = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(cw // 2, ch // 2, image=self._photo)
        except ImportError:
            self.canvas.delete("all")
            self.canvas.create_text(350, 300, fill="white",
                                    text="Install Pillow to view plots:\n  pip install Pillow",
                                    font=("TkDefaultFont", 14))
        except Exception as e:
            self.canvas.delete("all")
            self.canvas.create_text(350, 300, fill="red", text=f"Cannot display:\n{e}",
                                    font=("TkDefaultFont", 12))

    def _show_current(self):
        if self.image_files:
            self._show_image(self.current_index)
        else:
            self._show_empty()

    def _show_empty(self):
        self.canvas.delete("all")
        self.file_label.configure(text="No plot images found.")
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 50 or ch < 50:
            cw, ch = 700, 600
        self.canvas.create_text(cw // 2, ch // 2, fill="white",
                                text="No plot images found.\nUse Group Scatter to compute one from the summary CSV.",
                                font=("TkDefaultFont", 14), justify="center")

    def _prev(self):
        if self.image_files:
            self._show_image(self.current_index - 1)

    def _next(self):
        if self.image_files:
            self._show_image(self.current_index + 1)

    def _make_group_scatter(self):
        x_label = self.scatter_x_var.get()
        y_label = self.scatter_y_var.get()
        if x_label not in self._metric_lookup or y_label not in self._metric_lookup:
            self.scatter_status.configure(text="Select metrics for both axes first.", foreground="red")
            return
        if x_label == y_label:
            self.scatter_status.configure(text="Select two different metrics.", foreground="red")
            return

        import waveanalysis.plotting as pt
        import waveanalysis.housekeeping.housekeeping_functions as hf

        x_col = self._metric_lookup[x_label]
        y_col = self._metric_lookup[y_label]
        try:
            fig = pt.generate_group_metric_scatter(
                summary_df=self._summary_df,
                x_param=x_col,
                y_param=y_col,
                dark_plots=self.scatter_dark_var.get(),
            )
            out_dir = os.path.join(self.results_path, "group_scatter_graphs")
            os.makedirs(out_dir, exist_ok=True)
            output_path = os.path.join(out_dir, hf.sanitize_filename(f"{x_col} vs {y_col}.png"))
            fig.savefig(output_path)
            self.scatter_status.configure(text=f"Saved: {os.path.basename(output_path)}", foreground="green")
            self._show_generated_plot(output_path)
        except Exception as e:
            self.scatter_status.configure(text=f"Could not create plot: {e}", foreground="red")

    def _make_group_correlations(self):
        try:
            import pandas as pd
            import waveanalysis.plotting as pt
            import waveanalysis.housekeeping.housekeeping_functions as hf

            csv_files = glob.glob(os.path.join(self.results_path, "!*_summary.csv"))
            if not csv_files:
                raise ValueError("No summary CSV found.")

            summary_df = pd.read_csv(sorted(csv_files)[-1])
            figs = pt.generate_group_metric_correlations(
                summary_df=summary_df,
                dark_plots=self.default_dark_plots,
            )

            out_dir = os.path.join(self.results_path, "group_metric_correlations")
            os.makedirs(out_dir, exist_ok=True)
            first_path = None
            for name, fig in figs.items():
                output_path = os.path.join(out_dir, f"{hf.sanitize_filename(name)}.png")
                fig.savefig(output_path)
                if first_path is None:
                    first_path = output_path
            if first_path:
                self._show_generated_plot(first_path)
        except Exception as e:
            messagebox.showerror("Group Correlations", f"Could not create group correlations:\n{e}")

    def _show_generated_plot(self, output_path):
        self.image_files = self._scan_image_files()
        self.tree.delete(*self.tree.get_children(""))
        self._populate_tree()
        self.count_label.configure(text=f"{len(self.image_files)} plots")
        try:
            idx = self.image_files.index(output_path)
        except ValueError:
            idx = len(self.image_files) - 1
        self._show_image(idx)


# ---------------------------------------------------------------------------
# Summary table viewer
# ---------------------------------------------------------------------------

class _SummaryViewer(tk.Toplevel):

    def __init__(self, parent, results_path):
        super().__init__(parent)
        self.title("Summary Table")
        self.geometry("1100x500")
        self._sort_col = None
        self._sort_rev = False

        csv_files = glob.glob(os.path.join(results_path, "!*_summary.csv"))
        if not csv_files:
            ttk.Label(self, text="No summary CSV found.").pack(padx=20, pady=20)
            return

        csv_path = sorted(csv_files)[-1]
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            headers = next(reader)
            rows = list(reader)

        if not headers:
            ttk.Label(self, text="Summary CSV is empty.").pack(padx=20, pady=20)
            return

        ttk.Label(self, text=os.path.basename(csv_path)).pack(padx=10, pady=(6, 0), anchor="w")

        tf = ttk.Frame(self)
        tf.pack(fill=tk.BOTH, expand=True, padx=5, pady=4)
        self.tree = ttk.Treeview(tf, columns=headers, show="headings", selectmode="browse")
        vsb = ttk.Scrollbar(tf, orient=tk.VERTICAL, command=self.tree.yview)
        hsb = ttk.Scrollbar(tf, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tf.grid_rowconfigure(0, weight=1)
        tf.grid_columnconfigure(0, weight=1)

        for c in headers:
            self.tree.heading(c, text=c, command=lambda col=c: self._sort_by(col))
            self.tree.column(c, width=110, minwidth=60)
        for r in rows:
            self.tree.insert("", "end", values=r)

        ttk.Label(self, text=f"{len(rows)} row(s)").pack(padx=10, pady=(0, 6), anchor="w")

    def _sort_by(self, col):
        if self._sort_col == col:
            self._sort_rev = not self._sort_rev
        else:
            self._sort_col = col
            self._sort_rev = False
        items = [(self.tree.set(iid, col), iid) for iid in self.tree.get_children("")]

        def key(it):
            try:
                return float(it[0])
            except (ValueError, TypeError):
                return it[0].lower()

        items.sort(key=key, reverse=self._sort_rev)
        for i, (_, iid) in enumerate(items):
            self.tree.move(iid, "", i)
        arrow = " v" if self._sort_rev else " ^"
        for c in self.tree["columns"]:
            self.tree.heading(c, text=c.rstrip(" ^v") + (arrow if c == col else ""))
