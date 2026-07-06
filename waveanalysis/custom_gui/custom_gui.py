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
# Retro theme
# ---------------------------------------------------------------------------

# Classic beveled palette (Win2k / Motif look). Two variants share every key so
# the GUI can flip between a warm light gray and a dark gray at runtime.
_RETRO_LIGHT = {
    "bg": "#d4d0c8",       # warm panel gray
    "field": "#ffffff",    # entry / list / text field background
    "text": "#1a1a1a",
    "disabled": "#9a968f",
    "light": "#ffffff",    # top-left bevel highlight
    "dark": "#808080",     # bottom-right bevel shadow
    "trough": "#bfbbb3",
    "select": "#4a6b8a",   # muted steel-blue selection
    "active": "#e3e0d9",   # hovered button face
    "pressed": "#bdb9b1",  # pressed button face
    "log_bg": "#f3eede",   # parchment log field
    "log_fg": "#2a2a2a",
    "error": "#b00000",
    "accent_text": "#3f6088",
    "header_title": "#2a2a2a",
    "header_shadow": "#b9b5ad",
    "header_sub": "#6a6a6a",
    "header_pin": "#a8a49c",
}

_RETRO_DARK = {
    "bg": "#3a3936",       # dark warm gray panel
    "field": "#262521",    # sunken dark field
    "text": "#e6e3da",
    "disabled": "#7d7a72",
    "light": "#56544e",    # top-left bevel highlight (lighter than bg)
    "dark": "#1c1b19",     # bottom-right bevel shadow (darker than bg)
    "trough": "#2a2926",
    "select": "#5a82ab",
    "active": "#4a4843",
    "pressed": "#2c2b28",
    "log_bg": "#22211d",   # near-black warm terminal
    "log_fg": "#d8d4c6",
    "error": "#ff7a7a",
    "accent_text": "#9bbbe0",
    "header_title": "#ececdf",
    "header_shadow": "#262522",
    "header_sub": "#9a978d",
    "header_pin": "#5a5852",
}

# Active palette, mutated in place so every runtime _RETRO[...] lookup follows
# the current choice. Start light; _set_palette / saved config can switch it.
_RETRO = dict(_RETRO_LIGHT)


def _set_palette(name):
    _RETRO.clear()
    _RETRO.update(_RETRO_DARK if name == "dark" else _RETRO_LIGHT)


def _apply_retro_theme(root, palette="light"):
    """Give *root* -- and every child/Toplevel sharing its interpreter -- the
    classic beveled light-gray look: grooved panels, raised buttons, white
    sunken fields, steel-blue selection. Call once per Tk interpreter; failures
    are swallowed so a missing theme never blocks the GUI."""
    _set_palette(palette)
    p = _RETRO
    try:
        style = ttk.Style(root)
        style.theme_use("clam")  # most themeable theme; bundled on every platform
    except tk.TclError:
        return

    # Retro fonts: classic Mac Geneva for the UI, Monaco for the monospace log.
    families = set(tkfont.families(root))
    ui_family = next((f for f in ("Geneva", "Chicago", "ChicagoFLF") if f in families), None)
    mono_family = next((f for f in ("Monaco", "Andale Mono", "Courier") if f in families), None)
    if ui_family:
        for named in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont",
                      "TkIconFont", "TkTooltipFont", "TkSmallCaptionFont"):
            try:
                tkfont.nametofont(named).configure(family=ui_family)
            except tk.TclError:
                pass
    root._retro_ui_family = ui_family
    root._retro_mono_family = mono_family

    base_font = tkfont.nametofont("TkDefaultFont")
    bold_font = (base_font.actual("family"), base_font.actual("size"), "bold")

    root.configure(bg=p["bg"])

    # Classic (non-ttk) tk widgets read their defaults from the option database.
    root.option_add("*Toplevel.background", p["bg"])
    for widget in ("Listbox", "Text", "*TCombobox*Listbox"):
        root.option_add(f"*{widget}.background", p["field"])
        root.option_add(f"*{widget}.foreground", p["text"])
        root.option_add(f"*{widget}.selectBackground", p["select"])
        root.option_add(f"*{widget}.selectForeground", "white")
    root.option_add("*Listbox.relief", "sunken")
    root.option_add("*Listbox.borderWidth", 1)

    style.configure(
        ".", background=p["bg"], foreground=p["text"], fieldbackground=p["field"],
        bordercolor=p["dark"], lightcolor=p["light"], darkcolor=p["dark"],
        troughcolor=p["trough"], focuscolor=p["bg"], font=base_font,
    )
    style.configure("TFrame", background=p["bg"])
    style.configure("TLabel", background=p["bg"], foreground=p["text"])
    style.configure("TLabelframe", background=p["bg"], relief="groove",
                    bordercolor=p["dark"], lightcolor=p["light"], darkcolor=p["dark"])
    style.configure("TLabelframe.Label", background=p["bg"], foreground=p["text"],
                    font=bold_font)
    style.configure("TSeparator", background=p["dark"])
    style.configure("TPanedwindow", background=p["bg"])

    # Raised, beveled buttons; sink in on press.
    style.configure("TButton", background=p["bg"], foreground=p["text"],
                    relief="raised", padding=(10, 4), anchor="center",
                    bordercolor=p["dark"], lightcolor=p["light"], darkcolor=p["dark"])
    style.map("TButton",
              background=[("pressed", p["pressed"]), ("active", p["active"]),
                          ("disabled", p["bg"])],
              foreground=[("disabled", p["disabled"])],
              relief=[("pressed", "sunken")])
    # Primary action button: bold, with a pale-blue raised face so it reads as
    # the default action while staying an obviously filled, beveled button. Its
    # text stays dark in both themes for contrast on the light-blue face.
    style.configure("Retro.Accent.TButton", font=bold_font, foreground="#15233a",
                    relief="raised", borderwidth=2, padding=(10, 4),
                    background="#8fb3dc", bordercolor="#2f4f73",
                    lightcolor="#bcd3ed", darkcolor="#5b7da6")
    style.map("Retro.Accent.TButton",
              background=[("pressed", "#6f97c6"), ("active", "#a3c2e6"),
                          ("disabled", p["bg"])],
              foreground=[("disabled", p["disabled"])],
              relief=[("pressed", "sunken")])

    # White, sunken text fields.
    for field in ("TEntry", "TSpinbox", "TCombobox"):
        style.configure(field, fieldbackground=p["field"], foreground=p["text"],
                        background=p["bg"], relief="sunken", arrowcolor=p["text"],
                        bordercolor=p["dark"], lightcolor=p["dark"],
                        darkcolor=p["dark"], insertcolor=p["text"])
    style.map("TCombobox",
              fieldbackground=[("readonly", p["field"]), ("disabled", p["bg"])],
              selectbackground=[("readonly", p["select"])],
              selectforeground=[("readonly", "white")],
              foreground=[("disabled", p["disabled"])])
    style.map("TSpinbox", arrowcolor=[("disabled", p["dark"])])
    style.map("TEntry", foreground=[("disabled", p["disabled"])])

    style.configure("TCheckbutton", background=p["bg"], foreground=p["text"],
                    indicatorcolor=p["field"], indicatorrelief="sunken",
                    focuscolor=p["bg"])
    style.map("TCheckbutton", background=[("active", p["bg"])],
              indicatorcolor=[("selected", p["field"]), ("pressed", p["active"])])
    style.configure("TRadiobutton", background=p["bg"], foreground=p["text"],
                    indicatorcolor=p["field"], focuscolor=p["bg"])
    style.map("TRadiobutton", background=[("active", p["bg"])])

    style.configure("TScale", background=p["bg"], troughcolor=p["trough"])

    style.configure("Treeview", background=p["field"], fieldbackground=p["field"],
                    foreground=p["text"], bordercolor=p["dark"])
    style.map("Treeview", background=[("selected", p["select"])],
              foreground=[("selected", "white")])
    style.configure("Treeview.Heading", background=p["bg"], foreground=p["text"],
                    relief="raised", font=bold_font)
    style.map("Treeview.Heading", background=[("active", p["active"])])

    for sb in ("Vertical.TScrollbar", "Horizontal.TScrollbar"):
        style.configure(sb, background=p["bg"], troughcolor=p["trough"],
                        bordercolor=p["dark"], arrowcolor=p["text"],
                        lightcolor=p["light"], darkcolor=p["dark"])

    style.configure("TNotebook", background=p["bg"], bordercolor=p["dark"])
    style.configure("TNotebook.Tab", background=p["bg"], foreground=p["text"],
                    padding=(12, 5), bordercolor=p["dark"], font=bold_font)
    style.map("TNotebook.Tab",
              background=[("selected", p["field"]), ("active", p["active"])])


def _add_popup_header(toplevel, gui, title, subtitle=None, height=56):
    """Compact header banner (rainbow logo + serif wordmark + beveled divider)
    matching the main window, for Toplevel popups. *gui* supplies the cached
    logo image and font helpers."""
    canvas = tk.Canvas(toplevel, height=height, highlightthickness=0,
                       bg=_RETRO["bg"], bd=0)
    canvas.pack(fill=tk.X)

    def _draw(_e=None):
        try:
            if not canvas.winfo_exists():
                return
        except tk.TclError:
            return
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w < 10:
            return
        canvas.delete("all")
        size = max(h - 16, 18)
        logo_right = gui._draw_logo(canvas, 9, 8, size)
        fam = gui._header_font_family()
        tx = logo_right + 14
        cy = h // 2
        tf = tkfont.Font(family=fam, size=19, weight="bold")
        canvas.create_text(tx + 1, cy - 7, text=title, anchor="w",
                           font=tf, fill=_RETRO["header_shadow"])
        canvas.create_text(tx, cy - 8, text=title, anchor="w",
                           font=tf, fill=_RETRO["header_title"])
        if subtitle:
            canvas.create_text(tx + 2, cy + 13, text="  ".join(subtitle.upper()),
                               anchor="w", font=(fam, 8), fill=_RETRO["header_sub"])
        canvas.create_line(0, h - 2, w, h - 2, fill=_RETRO["dark"])
        canvas.create_line(0, h - 1, w, h - 1, fill=_RETRO["light"])

    canvas.bind("<Configure>", _draw)
    toplevel.after(40, _draw)
    return canvas


class _SegmentedProgress(tk.Canvas):
    """Classic segmented progress bar: discrete blue blocks marching across a
    sunken gray trough. Drop-in for the bits of ttk.Progressbar we use --
    ``configure(maximum=..., value=...)``."""

    # Lighter blue shades for the trailing "comet" of the marching sweep.
    _COMET = ("#6f97c6", "#8fb3dc", "#a9c6e6", "#c4d8ef")

    def __init__(self, parent, **kw):
        super().__init__(parent, height=16, highlightthickness=0,
                         bg=_RETRO["trough"], bd=2, relief="sunken", **kw)
        self._max = 100.0
        self._val = 0.0
        self._marching = False
        self._phase = 0
        self._anim_id = None
        self._interval = 180  # ms per step; slow while idle, faster during runs
        self.bind("<Configure>", lambda _e: self._redraw())

    def set_speed(self, fast):
        """Faster sweep while analysis runs, slower gentle drift while idle."""
        self._interval = 90 if fast else 180

    def configure(self, cnf=None, **kw):
        if "maximum" in kw:
            self._max = max(float(kw.pop("maximum") or 1), 1.0)
        if "value" in kw:
            self._val = float(kw.pop("value") or 0)
        if cnf is not None or kw:
            super().configure(cnf, **kw)
        self._redraw()

    config = configure

    def start(self):
        """Begin the indeterminate marching sweep through the unfilled trough."""
        if self._marching:
            return
        self._marching = True
        self._step()

    def stop(self):
        self._marching = False
        if self._anim_id is not None:
            try:
                self.after_cancel(self._anim_id)
            except Exception:
                pass
            self._anim_id = None
        self._redraw()

    def _step(self):
        if not self._marching:
            return
        try:
            if not self.winfo_exists():
                return
            self._phase += 1
            self._redraw()
            self._anim_id = self.after(self._interval, self._step)
        except tk.TclError:
            self._marching = False

    def _redraw(self):
        try:
            if not self.winfo_exists():
                return
        except tk.TclError:
            return
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w <= 2:
            return
        pad = 2
        seg, gap = 11, 3
        period = seg + gap
        frac = min(self._val / self._max, 1.0) if self._max else 0.0
        filled = (w - 2 * pad) * frac

        # Marching comet first (drawn under the solid fill, so it only shows in
        # the empty trough and reads as "still working").
        if self._marching:
            n = max(int((w - 2 * pad) // period), 1)
            head = self._phase % (n + len(self._COMET))
            for k, shade in enumerate(self._COMET):
                idx = head - k
                if 0 <= idx < n:
                    sx = pad + idx * period
                    self.create_rectangle(sx, pad, min(sx + seg, w - pad), h - pad,
                                          fill=shade, outline=shade)

        x = pad
        while x - pad < filled:
            x1 = min(x + seg, pad + filled)
            self.create_rectangle(x, pad, x1, h - pad,
                                  fill=_RETRO["select"], outline=_RETRO["select"])
            x += period


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _GUIBase(_TkBase):
    """Shared widget helpers and actions used by all three analysis GUI classes."""

    _has_group_names = False
    _has_plot_flags = False

    # Chosen left-to-right order (list of original group names) and display-name
    # overrides for the group comparison graphs. Set via the "Order & names..."
    # dialog next to the group names entry; empty until the user edits them.
    _group_order = None
    _group_labels = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._theme = _load_config().get("theme", "light")
        if self._theme not in ("light", "dark"):
            self._theme = "light"
        _apply_retro_theme(self, self._theme)

    # ---- 80s Apple header banner ----

    # Classic six-color Apple logo stripes, top-to-bottom (used as a fallback).
    _APPLE_STRIPES = ("#5cb85c", "#f7d000", "#f5821f", "#e03a3e", "#8e44ad", "#3aa0dd")
    # Mid gray used for the logo's below-threshold gaps and its outer stroke.
    _LOGO_GAP = (110, 110, 110)

    @staticmethod
    def _bg_rgb():
        h = _RETRO["bg"].lstrip("#")
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

    def _turing_logo_base(self):
        """Generate (once) a rainbow Turing / excitable-media pattern as a PIL
        image. Returns the cached image, or ``False`` if the imaging libs are
        unavailable so the caller can fall back to the stripe mark."""
        if getattr(self, "_turing_base", None) is not None:
            return self._turing_base
        try:
            import numpy as np
            from scipy.ndimage import gaussian_filter
            from PIL import Image
        except Exception:
            self._turing_base = False
            return False

        # Reaction-diffusion at a single dominant wavelength: short-range
        # activation vs. long-range inhibition converges to clean, uniformly
        # spaced wavy stripes -- the classic pufferfish-skin labyrinth.
        n = 150
        rng = np.random.default_rng(8)
        grid = rng.standard_normal((n, n))
        activate, inhibit, amp = 4.0, 8.0, 0.15
        # Iterate well past pattern formation for a settled, steady-state look.
        for _ in range(150):
            act = gaussian_filter(grid, activate, mode="wrap")
            inh = gaussian_filter(grid, inhibit, mode="wrap")
            grid = np.clip(grid + amp * np.sign(act - inh), -1.0, 1.0)
        # Threshold the field into clean stripes; everything below goes black.
        sharp = gaussian_filter(grid, 0.6, mode="wrap")
        sharp = (sharp - sharp.min()) / (np.ptp(sharp) + 1e-9)
        stripes = sharp >= 0.5

        # The colors are pure decoration: a heavily smoothed random field
        # (uncorrelated with the stripes) indexes the classic, muted six-color
        # Apple rainbow, so soft color blobs land at random across the maze.
        color_field = gaussian_filter(
            np.random.default_rng(42).standard_normal((n, n)), 25.0, mode="wrap")
        color_field = (color_field - color_field.min()) / (np.ptp(color_field) + 1e-9)

        palette = np.array(
            [[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in self._APPLE_STRIPES],
            dtype=float,
        )
        stops = np.linspace(0.0, 1.0, len(palette))
        lut_x = np.linspace(0.0, 1.0, 256)
        lut = np.stack([np.interp(lut_x, stops, palette[:, k]) for k in range(3)], axis=1)
        # Posterize the color field into a few bands so the palette steps in
        # hard blocks instead of blending smoothly.
        levels = 6
        banded = np.clip(np.floor(color_field * levels) / (levels - 1), 0.0, 1.0)
        idx = (banded * 255).astype(int)
        rgb = lut[idx].astype("uint8").copy()
        rgb[~stripes] = self._LOGO_GAP  # below threshold -> mid gray for contrast
        self._turing_base = Image.fromarray(rgb, "RGB").convert("RGBA")
        return self._turing_base

    def _draw_logo(self, canvas, x, y, size):
        """Draw the circular Turing-pattern logo (or stripe fallback). Returns
        the x coordinate of the logo's right edge."""
        base = self._turing_logo_base()
        if base:
            try:
                from PIL import Image, ImageDraw, ImageTk
                ss = max(size * 2, 2)  # supersample for clean circular edges
                stroke = max(int(ss * 0.08), 1)
                inner = max(ss - 2 * stroke, 2)

                # Gray disc provides the outer stroke; pattern sits inset inside it.
                disc = Image.new("RGBA", (ss, ss), self._LOGO_GAP + (255,))
                pat = base.resize((inner, inner), Image.LANCZOS).convert("RGBA")
                pat_mask = Image.new("L", (inner, inner), 0)
                ImageDraw.Draw(pat_mask).ellipse((0, 0, inner - 1, inner - 1), fill=255)
                disc.paste(pat, (stroke, stroke), pat_mask)

                outer_mask = Image.new("L", (ss, ss), 0)
                ImageDraw.Draw(outer_mask).ellipse((0, 0, ss - 1, ss - 1), fill=255)
                bg = Image.new("RGBA", (ss, ss), self._bg_rgb() + (255,))
                composed = (Image.composite(disc, bg, outer_mask)
                            .resize((size, size), Image.LANCZOS).convert("RGB"))
                # Keep the PhotoImage alive on the canvas itself, so reusing this
                # for multiple windows (header + popups) doesn't clobber a shared
                # reference and let an image get garbage-collected.
                canvas._wa_logo_photo = ImageTk.PhotoImage(composed)
                canvas.create_image(x, y, anchor="nw", image=canvas._wa_logo_photo)
                return x + size
            except Exception:
                pass

        # Fallback: classic six-color Apple stripes.
        lw = int(size * 0.95)
        n = len(self._APPLE_STRIPES)
        stripe_h = size / n
        skew = 6
        for i, col in enumerate(self._APPLE_STRIPES):
            y0 = y + i * stripe_h
            y1 = y0 + stripe_h + 0.6
            canvas.create_polygon(x + skew, y0, x + lw + skew, y0,
                                  x + lw, y1, x, y1, fill=col, outline=col)
        return x + lw + skew

    def _header_font_family(self):
        """Pick the most Apple-Garamond-ish serif that's actually installed."""
        if getattr(self, "_hdr_family", None):
            return self._hdr_family
        available = set(tkfont.families())
        for cand in ("Apple Garamond", "Garamond", "Palatino", "Palatino Linotype",
                     "Hoefler Text", "Georgia", "Times New Roman"):
            if cand in available:
                self._hdr_family = cand
                break
        else:
            self._hdr_family = tkfont.nametofont("TkDefaultFont").actual("family")
        return self._hdr_family

    def _build_header(self, parent, subtitle=None):
        """Banner with the rainbow Apple logo block, a serif 'Wave Analysis'
        wordmark, classic Mac pinstripes, and a beveled divider underneath."""
        self._header_subtitle = subtitle
        canvas = tk.Canvas(parent, height=74, highlightthickness=0,
                           bg=_RETRO["bg"], bd=0)
        canvas.pack(fill=tk.X, pady=(0, 6))
        canvas.bind("<Configure>", lambda _e, c=canvas: self._draw_header(c))
        self.after(60, lambda c=canvas: self._draw_header(c))
        self._header_canvas = canvas
        return canvas

    def _draw_header(self, canvas):
        try:
            if not canvas.winfo_exists():
                return
        except tk.TclError:
            return
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w < 10:
            return
        canvas.delete("all")

        # Circular rainbow Turing / excitable-media logo (falls back to stripes).
        lx, ly = 10, 9
        size = max(h - 20, 24)
        logo_right = self._draw_logo(canvas, lx, ly, size)

        fam = self._header_font_family()
        title_x = logo_right + 18
        cy = h // 2
        title_font = tkfont.Font(family=fam, size=27, weight="bold")
        # Soft drop shadow, then the wordmark.
        canvas.create_text(title_x + 1, cy - 9, text="Wave Analysis", anchor="w",
                           font=title_font, fill=_RETRO["header_shadow"])
        canvas.create_text(title_x, cy - 10, text="Wave Analysis", anchor="w",
                           font=title_font, fill=_RETRO["header_title"])

        sub = getattr(self, "_header_subtitle", None)
        if sub:
            spaced = "  ".join(sub.upper())
            canvas.create_text(title_x + 2, cy + 16, text=spaced, anchor="w",
                               font=(fam, 9), fill=_RETRO["header_sub"])

        # Classic Mac title-bar pinstripes filling the empty space on the right.
        ps_x0 = title_x + title_font.measure("Wave Analysis") + 28
        ps_x1 = w - 12
        if ps_x1 - ps_x0 > 50:
            for i in range(6):
                yy = cy - 16 + i * 6
                canvas.create_line(ps_x0, yy, ps_x1, yy, fill=_RETRO["header_pin"])

        # Beveled divider under the banner.
        canvas.create_line(0, h - 2, w, h - 2, fill=_RETRO["dark"])
        canvas.create_line(0, h - 1, w, h - 1, fill=_RETRO["light"])

    # ---- widget helpers (accept a parent frame) ----

    @staticmethod
    def _add_entry(parent, row, col, var, label_text, width=5):
        entry = ttk.Entry(parent, width=width, textvariable=var)
        entry.grid(row=row, column=col, padx=2, pady=2, sticky="e")
        label = ttk.Label(parent, text=label_text)
        label.grid(row=row, column=col + 1, padx=(4, 8), pady=2, sticky="w")
        return entry, label

    def _add_group_names(self, parent, row):
        '''Group names entry + label, with the order/rename button below the label.'''
        holder = ttk.Frame(parent)
        holder.grid(row=row, column=0, columnspan=2, sticky="w")
        ttk.Entry(holder, width=14, textvariable=self.vars["group_names"]).grid(
            row=0, column=0, padx=2, pady=2, sticky="e")
        ttk.Label(holder, text="Group names").grid(
            row=0, column=1, padx=(4, 8), pady=2, sticky="w")
        ttk.Button(holder, text="Order & names...", command=self._edit_group_order).grid(
            row=1, column=1, padx=(4, 8), pady=(0, 2), sticky="w")

    def _current_group_names(self):
        '''De-duplicated, non-blank group names from the entry, in entry order.'''
        raw = self.vars["group_names"].get() if "group_names" in self.vars else ""
        return list(dict.fromkeys(g.strip() for g in raw.split(",") if g.strip()))

    def _edit_group_order(self):
        '''Open the reorder/rename dialog seeded from the current group names.'''
        groups = self._current_group_names()
        if len(groups) < 2:
            messagebox.showinfo(
                "Group order & names",
                "Enter at least two comma-separated group names first.",
            )
            return
        # Preserve any prior ordering/renames for groups that still exist.
        if self._group_order:
            ordered = [g for g in self._group_order if g in groups]
            ordered += [g for g in groups if g not in ordered]
        else:
            ordered = groups
        dialog = _GroupOrderDialog(self, ordered, initial_labels=self._group_labels)
        if dialog.result is not None:
            self._group_order, self._group_labels = dialog.result

    @staticmethod
    def _add_check(parent, row, col, var, label_text):
        cb = ttk.Checkbutton(parent, variable=var)
        cb.grid(row=row, column=col, padx=2, pady=1, sticky="e")
        lbl = ttk.Label(parent, text=label_text)
        lbl.grid(row=row, column=col + 1, padx=(2, 8), pady=1, sticky="w")
        return cb, lbl

    def _build_group_stats_controls(self, parent, row):
        """Group-comparison stats: a toggle plus the test-family selector.

        'Group stats' turns the significance annotation on group comparison and
        quality plots on/off. The test family is non-parametric (Mann–Whitney /
        Kruskal–Wallis, no distribution assumption) or parametric (t-test /
        one-way ANOVA, which assume roughly normal groups).
        """
        gs = ttk.Frame(parent)
        gs.grid(row=row, column=0, columnspan=6, sticky="w", pady=(4, 0))
        ttk.Checkbutton(gs, variable=self.vars["group_stats"],
                        text="Group stats").pack(side=tk.LEFT, padx=(2, 10))
        ttk.Label(gs, text="Test:").pack(side=tk.LEFT)
        ttk.Radiobutton(gs, text="Non-parametric", value="nonparametric",
                        variable=self.vars["group_stats_test"]).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Radiobutton(gs, text="Parametric (ANOVA / t-test)", value="parametric",
                        variable=self.vars["group_stats_test"]).pack(side=tk.LEFT, padx=(4, 0))

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

        # Button grid. Columns 0-3 hold the main controls in one uniform group
        # so every button is the same width; a weighted spacer (col 4) pushes the
        # mode-switch buttons (cols 5-6, their own uniform group) to the right.
        # Rows are grouped by purpose: run controls on top, utilities below.
        btnbar = ttk.Frame(parent)
        btnbar.pack(fill=tk.X, pady=(6, 4))
        for c in (0, 1, 2, 3):
            btnbar.columnconfigure(c, weight=0, uniform="mainbtn")
        btnbar.columnconfigure(4, weight=1)
        for c in (5, 6):
            btnbar.columnconfigure(c, weight=0, uniform="modebtn")

        def _cell(col):
            return {"sticky": "ew", "padx": (0 if col == 0 else 4, 0), "pady": (0, 4)}

        # Row 0: run controls
        self.start_button = ttk.Button(btnbar, text="Start Analysis", command=self.start_analysis,
                                       style="Retro.Accent.TButton")
        self.start_button.grid(row=0, column=0, **_cell(0))
        self.stop_button = ttk.Button(btnbar, text="Stop", command=self.stop_analysis, state="disabled")
        self.stop_button.grid(row=0, column=1, **_cell(1))
        ttk.Button(btnbar, text="Test File", command=self._test_first_file).grid(row=0, column=2, **_cell(2))
        ttk.Button(btnbar, text="Preview Bins", command=self._preview_bins).grid(row=0, column=3, **_cell(3))

        # Row 1: results / utilities / window
        ttk.Button(btnbar, text="Load Results", command=self._load_existing_results).grid(row=1, column=0, **_cell(0))
        ttk.Button(btnbar, text="Info", command=self._open_info).grid(row=1, column=1, **_cell(1))
        self.theme_button = ttk.Button(
            btnbar, text=("Light Mode" if self._theme == "dark" else "Dark Mode"),
            command=self._toggle_theme)
        self.theme_button.grid(row=1, column=2, **_cell(2))
        ttk.Button(btnbar, text="Close Window", command=self.cancel_analysis).grid(row=1, column=3, **_cell(3))

        self._build_extra_buttons(btnbar)  # subclass hook: mode-switch buttons (cols 5-6)

        # status + progress (status sits in a sunken inset strip, with a playful
        # ticker on the right that animates while analysis runs)
        status_box = tk.Frame(parent, bg=_RETRO["bg"], relief="sunken", bd=1)
        status_box.pack(fill=tk.X)
        self._status_box = status_box
        self.elapsed_label = ttk.Label(status_box, text="", font=("TkDefaultFont", 10, "bold"))
        self.elapsed_label.pack(side=tk.RIGHT, padx=(4, 6), pady=1)
        self.flavor_label = ttk.Label(status_box, text="", foreground=_RETRO["accent_text"],
                                      font=("TkDefaultFont", 10, "bold"))
        self.flavor_label.pack(side=tk.RIGHT, padx=(4, 0), pady=1)
        self.status_label = ttk.Label(status_box, text="Status: Ready",
                                      font=("TkDefaultFont", 10, "bold"),
                                      anchor="w", justify="left")
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3, pady=1)

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
        self.progress_bar = _SegmentedProgress(prog)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.subprogress_label = ttk.Label(prog, text="", width=24, anchor="e",
                                           foreground=_RETRO["accent_text"])
        self.subprogress_label.pack(side=tk.LEFT, padx=(6, 0))
        self.progress_file_label = ttk.Label(prog, text="", width=10, anchor="e")
        self.progress_file_label.pack(side=tk.LEFT, padx=(6, 0))
        self.progress_bar.start()  # gentle marching even while idle

        # log
        self.log_text = scrolledtext.ScrolledText(parent, height=10, state="disabled", wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=(2, 0))
        self.log_text.tag_configure("error", foreground=_RETRO["error"])
        # Classic terminal feel: parchment (or dark) field, sunken bevel.
        self.log_text.configure(background=_RETRO["log_bg"], foreground=_RETRO["log_fg"],
                                relief="sunken", borderwidth=2, highlightthickness=0,
                                insertbackground=_RETRO["log_fg"])
        mono = getattr(self, "_retro_mono_family", None)
        if mono:
            self.log_text.configure(font=(mono, 11))

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
        """Override in subclasses to add mode-switch buttons to the button grid.
        Place them in columns 5-6, row 0 (the right-aligned mode-switch group)."""
        pass

    def _open_info(self):
        _InfoPanel(self)

    # ---- light / dark theme toggle ----

    def _toggle_theme(self):
        self._theme = "dark" if getattr(self, "_theme", "light") == "light" else "light"
        _save_config({"theme": self._theme})
        _apply_retro_theme(self, self._theme)       # restyles every ttk widget + root bg
        self._refresh_palette_widgets()             # update the explicitly-colored bits
        try:
            self.theme_button.configure(
                text=("Light Mode" if self._theme == "dark" else "Dark Mode"))
        except Exception:
            pass

    def _refresh_palette_widgets(self):
        """Re-apply palette colors to widgets that set explicit (non-ttk) colors,
        since the ttk restyle alone doesn't touch them."""
        p = _RETRO
        try:
            self.configure(bg=p["bg"])
        except tk.TclError:
            return
        if getattr(self, "_status_box", None) is not None:
            self._status_box.configure(bg=p["bg"])
        if getattr(self, "log_text", None) is not None:
            self.log_text.configure(background=p["log_bg"], foreground=p["log_fg"],
                                    insertbackground=p["log_fg"])
            self.log_text.tag_configure("error", foreground=p["error"])
        for name in ("flavor_label", "subprogress_label"):
            w = getattr(self, name, None)
            if w is not None:
                w.configure(foreground=p["accent_text"])
        if getattr(self, "progress_bar", None) is not None:
            try:
                self.progress_bar.configure(bg=p["trough"])
            except tk.TclError:
                pass
        canvas = getattr(self, "_header_canvas", None)
        if canvas is not None:
            try:
                canvas.configure(bg=p["bg"])
                self._draw_header(canvas)
            except tk.TclError:
                pass

    # ---- playful run + idle animations ----

    _RUN_FLAVORS = (
        "Crunching waves", "Chasing oscillations", "Correlating channels",
        "Measuring peaks", "Summoning Fourier", "Aligning phases",
        "Counting bins", "Smoothing signals", "Tracing landmarks",
    )

    def _start_run_animations(self):
        """Kick off the marching progress sweep and the flavor ticker. Each
        self-stops once _is_running flips false (set by the analysis thread)."""
        self._spin_idx = 0
        self._flavor_idx = 0
        self._flavor_ctr = 0
        try:
            self.progress_bar.set_speed(True)
            self.progress_bar.start()
        except Exception:
            pass
        self._run_anim_tick()

    def _run_anim_tick(self):
        try:
            if not self.winfo_exists():
                return
        except tk.TclError:
            return
        if not getattr(self, "_is_running", False):
            # finalize: clear the ticker and sub-progress; the bar keeps marching
            # gently while idle.
            try:
                self.flavor_label.configure(text="")
                self.subprogress_label.configure(text="")
                self.progress_bar.set_speed(False)  # back to gentle idle drift
            except Exception:
                pass
            return
        self._spin_idx += 1
        self._flavor_ctr += 1
        if self._flavor_ctr >= 22:
            self._flavor_ctr = 0
            self._flavor_idx = (self._flavor_idx + 1) % len(self._RUN_FLAVORS)
        dots = "." * (self._spin_idx % 4)
        try:
            self.flavor_label.configure(text=f"{self._RUN_FLAVORS[self._flavor_idx]}{dots}")
        except Exception:
            return
        self.after(110, self._run_anim_tick)

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
            self._do_append(msg if msg.endswith("\n") else msg + "\n")
        self.after(0, _append)

    def _do_append(self, text):
        self.log_text.configure(state="normal")
        if "ERROR" in text.upper():
            self.log_text.insert(tk.END, text, "error")
        else:
            self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

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

    def set_subprogress(self, text):
        """Compact live indicator for the per-file tqdm sub-bars, shown next to
        the progress bar instead of flooding the log."""
        def _upd():
            try:
                if not self.winfo_exists():
                    return
            except tk.TclError:
                return
            self.subprogress_label.configure(text=text)
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
            self.subprogress_label.configure(text="")
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

    def _preview_bins(self):
        """Show the current box/line bin geometry over one image from the folder."""
        folder = self.vars["folder_path"].get()
        if not folder or not os.path.isdir(folder):
            self.log_message("ERROR: Select a valid folder first")
            return

        tifs = sorted(
            (f for f in os.listdir(folder) if f.endswith(".tif") and not f.startswith(".")),
            key=_natural_sort_key,
        )
        if not tifs:
            self.log_message("ERROR: No .tif files found in the selected folder")
            return

        if len(tifs) == 1:
            chosen = tifs[0]
        else:
            chosen = _FilePickerDialog(self, tifs, title="Select an image to preview bins").result
            if chosen is None:
                return

        try:
            preview_image, overlay_info = self._create_bin_preview(os.path.join(folder, chosen))
            _ImagePreviewWindow(self, preview_image, f"Bin Preview - {chosen}", overlay_info)
        except Exception as e:
            self.log_message(f"ERROR: Could not preview bins: {e}")

    def _create_bin_preview(self, image_path):
        from PIL import Image, ImageDraw
        import numpy as np
        from waveanalysis.image_props.image_to_np_arrays import (
            tiff_to_np_array_multi_frame,
            tiff_to_np_array_single_frame,
        )

        analysis_type = self.vars["analysis_type"].get()
        step = int(self.vars["bin_shift"].get())
        if step <= 0:
            raise ValueError("Bin shift must be > 0")

        if analysis_type == "kymograph":
            line_width = int(self.vars["line_width"].get())
            if line_width <= 0:
                raise ValueError("Line width must be > 0")
            image = tiff_to_np_array_single_frame(image_path)
            channel_images = []
            display_shape = np.asarray(image[0], dtype=float).shape
            height, width = display_shape
            count = 0
            for channel in range(min(image.shape[0], 2)):
                display = np.asarray(image[channel], dtype=float)
                preview = self._normalize_preview_image(display)
                channel_image = Image.fromarray(preview).convert("RGB")
                draw = ImageDraw.Draw(channel_image, "RGBA")
                channel_count = 0
                for x0 in range(0, width, step):
                    x1 = x0 + line_width
                    if x1 <= width:
                        draw.rectangle((x0, 0, x1 - 1, height - 1), outline=(245, 245, 245, 150), width=1)
                        channel_count += 1
                count = max(count, channel_count)
                channel_images.append(channel_image)
            pil_image = self._join_preview_channels(channel_images)
            info = f"Kymograph bins: line width {line_width}px, shift {step}px, {count} bins"
        else:
            box_size = int(self.vars["box_size"].get())
            if box_size <= 0:
                raise ValueError("Box size must be > 0")
            image = tiff_to_np_array_multi_frame(image_path)
            channel_images = []
            display_shape = np.asarray(image[0, 0, 0], dtype=float).shape
            height, width = display_shape
            half = box_size // 2
            count = 0
            for channel in range(min(image.shape[2], 2)):
                display = np.asarray(image[0, 0, channel], dtype=float)
                preview = self._normalize_preview_image(display)
                channel_image = Image.fromarray(preview).convert("RGB")
                draw = ImageDraw.Draw(channel_image, "RGBA")
                channel_count = 0
                for y_center in range(half, height - half, step):
                    for x_center in range(half, width - half, step):
                        x0 = x_center - half
                        y0 = y_center - half
                        x1 = x0 + box_size - 1
                        y1 = y0 + box_size - 1
                        draw.rectangle((x0, y0, x1, y1), outline=(245, 245, 245, 150), width=1)
                        channel_count += 1
                count = max(count, channel_count)
                channel_images.append(channel_image)
            pil_image = self._join_preview_channels(channel_images)
            info = f"Box bins: box size {box_size}px, shift {step}px, {count} bins"

        return pil_image, info

    @staticmethod
    def _join_preview_channels(channel_images):
        from PIL import Image

        if len(channel_images) <= 1:
            return channel_images[0]
        gap = 8
        width = sum(img.width for img in channel_images) + gap * (len(channel_images) - 1)
        height = max(img.height for img in channel_images)
        joined = Image.new("RGB", (width, height), (24, 24, 24))
        x = 0
        for img in channel_images:
            joined.paste(img, (x, 0))
            x += img.width + gap
        return joined

    @staticmethod
    def _normalize_preview_image(image):
        import numpy as np

        finite = image[np.isfinite(image)]
        if finite.size == 0:
            return np.zeros(image.shape, dtype=np.uint8)
        lo, hi = np.percentile(finite, [1, 99])
        if hi <= lo:
            lo, hi = np.nanmin(finite), np.nanmax(finite)
        if hi <= lo:
            return np.zeros(image.shape, dtype=np.uint8)
        scaled = np.clip((image - lo) / (hi - lo), 0, 1)
        return (scaled * 255).astype(np.uint8)

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
                # Pass the chosen comparison-graph order and display-name overrides,
                # keeping only groups that survive in the final group names list.
                present = set(self._resolved_params["group_names"])
                self._resolved_params["group_order"] = (
                    [g for g in self._group_order if g in present]
                    if self._group_order else None
                )
                self._resolved_params["group_labels"] = (
                    {k: v for k, v in self._group_labels.items() if k in present}
                    if self._group_labels else None
                )
                if getattr(self, "_ignore_group_names_once", False):
                    self._resolved_params["group_names"] = [""]
                    self._resolved_params["group_order"] = None
                    self._resolved_params["group_labels"] = None
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
                              "plot_metric_correlations",
                              "plot_indv_ACFs", "plot_indv_CCFs", "plot_indv_peaks",
                              "plot_heatmaps", "plot_landmark_shifts", "plot_indv_landmark_shifts",
                              "plot_fts", "dark_plots", "group_stats", "group_stats_test")
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
            self._start_run_animations()
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
            "plot_metric_correlations": tk.BooleanVar(value=True),
            "plot_indv_ACFs": tk.BooleanVar(value=False),
            "plot_indv_CCFs": tk.BooleanVar(value=False),
            "plot_indv_peaks": tk.BooleanVar(value=False),
            "plot_heatmaps": tk.BooleanVar(value=False),
            "plot_landmark_shifts": tk.BooleanVar(value=False),
            "plot_indv_landmark_shifts": tk.BooleanVar(value=False),
            "plot_fts": tk.BooleanVar(value=False),
            "dark_plots": tk.BooleanVar(value=True),
            "group_stats": tk.BooleanVar(value=True),
            "group_stats_test": tk.StringVar(value="nonparametric"),
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

        self._build_header(root, subtitle="Standard Analysis")

        # ---- top: three weighted columns that share the width and stay
        # top-aligned, so resizing distributes space instead of leaving a gap ----
        top = ttk.Frame(root)
        top.pack(fill=tk.X)
        top.columnconfigure(0, weight=3, uniform="topcol")
        top.columnconfigure(1, weight=3, uniform="topcol")
        top.columnconfigure(2, weight=4, uniform="topcol")

        # left: analysis options
        opts = ttk.LabelFrame(top, text="Analysis Options", padding=6)
        opts.grid(row=0, column=0, sticky="new", padx=(0, 4))

        pf = ttk.Frame(opts)
        pf.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        pf.columnconfigure(0, weight=1)
        ttk.Entry(pf, textvariable=self.vars["folder_path"]).grid(row=0, column=0, sticky="ew")
        ttk.Button(pf, text="Browse", command=self.get_folder_path, width=7).grid(row=0, column=1, padx=(4, 0))

        self._add_group_names(opts, 1)
        self._add_entry(opts, 2, 0, self.vars["box_size"], "Box size (px)")
        self._add_entry(opts, 3, 0, self.vars["bin_shift"], "Box shift (px)")
        self._add_entry(opts, 4, 0, self.vars["acf_peak_thresh"], "ACF peak thresh")
        self._add_entry(opts, 5, 0, self.vars["ccf_peak_thresh"], "CCF peak thresh")
        self._add_entry(opts, 6, 0, self.vars["peak_prominence_fraction"], "Peak prom. frac.")
        self._add_check(opts, 7, 0, self.vars["small_shifts_correction"], "Small shifts correction")

        # middle column: smoothing + channel names
        middle = ttk.Frame(top)
        middle.grid(row=0, column=1, sticky="new", padx=4)

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
        right.grid(row=0, column=2, sticky="new", padx=(4, 0))

        edge = ttk.LabelFrame(right, text="Landmark Edge Height", padding=4)
        edge.pack(fill=tk.X, pady=(0, 4))
        self._build_edge_height_slider(edge)

        # plot options (three aligned groups: Summary / Individual / Output)
        pl = ttk.LabelFrame(right, text="Plot Options", padding=4)
        pl.pack(fill=tk.X)
        pl.columnconfigure(1, minsize=120)
        pl.columnconfigure(3, minsize=92)
        self._add_check(pl, 0, 0, self.vars["plot_summary_ACFs"], "Summary ACFs")
        self._add_check(pl, 1, 0, self.vars["plot_summary_CCFs"], "Summary CCFs")
        self._add_check(pl, 2, 0, self.vars["plot_summary_peaks"], "Summary peaks")
        self._add_check(pl, 3, 0, self.vars["plot_metric_correlations"], "Metric correlations")
        self._add_check(pl, 0, 2, self.vars["plot_indv_ACFs"], "Indv ACFs")
        self._add_check(pl, 1, 2, self.vars["plot_indv_CCFs"], "Indv CCFs")
        self._add_check(pl, 2, 2, self.vars["plot_indv_peaks"], "Indv peaks")
        self._add_check(pl, 0, 4, self.vars["dark_plots"], "Dark plots")
        self._add_check(pl, 1, 4, self.vars["plot_heatmaps"], "Heatmaps")
        self._add_check(pl, 2, 4, self.vars["plot_fts"], "Fourier transforms")
        self._add_check(pl, 4, 0, self.vars["plot_landmark_shifts"], "Summary landmark")
        self._add_check(pl, 3, 2, self.vars["plot_indv_landmark_shifts"], "Indv landmark")
        self._build_group_stats_controls(pl, 5)

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
        self.minsize(940, 680)

    def _build_extra_buttons(self, parent):
        ttk.Button(parent, text="Rolling", command=self.launch_rolling_analysis).grid(
            row=0, column=5, sticky="ew", padx=(0, 4), pady=(0, 4))
        ttk.Button(parent, text="Kymograph", command=self.launch_kymograph_analysis).grid(
            row=0, column=6, sticky="ew", pady=(0, 4))

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
            "injection_ch3": tk.BooleanVar(value=False),
            "injection_ch4": tk.BooleanVar(value=False),
        }
        self.kymograph = False
        self._back_to_standard = False

        root = ttk.Frame(self, padding=8)
        root.pack(fill=tk.BOTH, expand=True)

        self._build_header(root, subtitle="Rolling Analysis")

        # Three balanced columns (mirrors the Standard window): options on the
        # left, smoothing + channel names stacked in the middle, edge height +
        # live injection stacked on the right.
        top = ttk.Frame(root)
        top.pack(fill=tk.X)
        top.columnconfigure(0, weight=3, uniform="topcol")
        top.columnconfigure(1, weight=3, uniform="topcol")
        top.columnconfigure(2, weight=3, uniform="topcol")

        # left: analysis options
        opts = ttk.LabelFrame(top, text="Analysis Options", padding=6)
        opts.grid(row=0, column=0, sticky="new", padx=(0, 4))

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

        # middle: smoothing + channel names
        middle = ttk.Frame(top)
        middle.grid(row=0, column=1, sticky="new", padx=4)

        sm = ttk.LabelFrame(middle, text="Smoothing", padding=4)
        sm.pack(fill=tk.X, pady=(0, 4))
        self._build_smoothing(sm, default_poly=3)

        cn = ttk.LabelFrame(middle, text="Channel Names (optional)", padding=4)
        cn.pack(fill=tk.X)
        self._add_entry(cn, 0, 0, self.vars["Ch1_name"], "Ch1", width=10)
        self._add_entry(cn, 1, 0, self.vars["Ch2_name"], "Ch2", width=10)
        self._add_entry(cn, 0, 2, self.vars["Ch3_name"], "Ch3", width=10)
        self._add_entry(cn, 1, 2, self.vars["Ch4_name"], "Ch4", width=10)

        # right: edge height + live injection
        right = ttk.Frame(top)
        right.grid(row=0, column=2, sticky="new", padx=(4, 0))

        edge = ttk.LabelFrame(right, text="Landmark Edge Height", padding=4)
        edge.pack(fill=tk.X, pady=(0, 4))
        self._build_edge_height_slider(edge)

        # live-injection options (leave injection frame at 0 to ignore)
        inj = ttk.LabelFrame(right, text="Live Injection (optional)", padding=4)
        inj.pack(fill=tk.X)
        self._add_entry(inj, 0, 0, self.vars["injection_frame"], "Injection frame", width=6)
        self._add_check(inj, 1, 0, self.vars["injection_ch1"], "Overlay signal Ch1")
        self._add_check(inj, 2, 0, self.vars["injection_ch2"], "Overlay signal Ch2")
        self._add_check(inj, 3, 0, self.vars["injection_ch3"], "Overlay signal Ch3")
        self._add_check(inj, 4, 0, self.vars["injection_ch4"], "Overlay signal Ch4")

        ttk.Separator(root, orient="horizontal").pack(fill=tk.X, pady=6)
        self._bottom = ttk.Frame(root)
        self._bottom.pack(fill=tk.BOTH, expand=True)
        self._build_bottom(self._bottom)

        self.on_start = None
        self._is_running = False
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._restore_settings()
        self.minsize(940, 680)

    def _build_extra_buttons(self, parent):
        ttk.Button(parent, text="Back to Standard", command=self._go_back).grid(
            row=0, column=6, sticky="ew", pady=(0, 4))

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
            "plot_metric_correlations": tk.BooleanVar(value=True),
            "plot_indv_ACFs": tk.BooleanVar(value=False),
            "plot_indv_CCFs": tk.BooleanVar(value=False),
            "plot_indv_peaks": tk.BooleanVar(value=False),
            "plot_heatmaps": tk.BooleanVar(value=False),
            "plot_landmark_shifts": tk.BooleanVar(value=False),
            "plot_indv_landmark_shifts": tk.BooleanVar(value=False),
            "plot_fts": tk.BooleanVar(value=False),
            "dark_plots": tk.BooleanVar(value=False),
            "group_stats": tk.BooleanVar(value=True),
            "group_stats_test": tk.StringVar(value="nonparametric"),
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

        self._build_header(root, subtitle="Kymograph Analysis")

        top = ttk.Frame(root)
        top.pack(fill=tk.X)
        top.columnconfigure(0, weight=3, uniform="topcol")
        top.columnconfigure(1, weight=3, uniform="topcol")
        top.columnconfigure(2, weight=4, uniform="topcol")

        # left: analysis options
        opts = ttk.LabelFrame(top, text="Analysis Options", padding=6)
        opts.grid(row=0, column=0, sticky="new", padx=(0, 4))

        pf = ttk.Frame(opts)
        pf.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        pf.columnconfigure(0, weight=1)
        ttk.Entry(pf, textvariable=self.vars["folder_path"]).grid(row=0, column=0, sticky="ew")
        ttk.Button(pf, text="Browse", command=self.get_folder_path, width=7).grid(row=0, column=1, padx=(4, 0))

        self._add_group_names(opts, 1)
        self._add_entry(opts, 2, 0, self.vars["line_width"], "Line width (px)")
        self._add_entry(opts, 3, 0, self.vars["bin_shift"], "Line shift (px)")
        self._add_entry(opts, 4, 0, self.vars["acf_peak_thresh"], "ACF peak thresh")
        self._add_entry(opts, 5, 0, self.vars["ccf_peak_thresh"], "CCF peak thresh")
        self._add_entry(opts, 6, 0, self.vars["peak_prominence_fraction"], "Peak prom. frac.")
        self._add_check(opts, 7, 0, self.vars["small_shifts_correction"], "Small shifts correction")

        # middle column: smoothing + channel names
        middle = ttk.Frame(top)
        middle.grid(row=0, column=1, sticky="new", padx=4)

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
        right.grid(row=0, column=2, sticky="new", padx=(4, 0))

        edge = ttk.LabelFrame(right, text="Landmark Edge Height", padding=4)
        edge.pack(fill=tk.X, pady=(0, 4))
        self._build_edge_height_slider(edge)

        # plot options (three aligned groups: Summary / Individual / Output)
        pl = ttk.LabelFrame(right, text="Plot Options", padding=4)
        pl.pack(fill=tk.X)
        pl.columnconfigure(1, minsize=120)
        pl.columnconfigure(3, minsize=92)
        self._add_check(pl, 0, 0, self.vars["plot_summary_ACFs"], "Summary ACFs")
        self._add_check(pl, 1, 0, self.vars["plot_summary_CCFs"], "Summary CCFs")
        self._add_check(pl, 2, 0, self.vars["plot_summary_peaks"], "Summary peaks")
        self._add_check(pl, 3, 0, self.vars["plot_metric_correlations"], "Metric correlations")
        self._add_check(pl, 0, 2, self.vars["plot_indv_ACFs"], "Indv ACFs")
        self._add_check(pl, 1, 2, self.vars["plot_indv_CCFs"], "Indv CCFs")
        self._add_check(pl, 2, 2, self.vars["plot_indv_peaks"], "Indv peaks")
        self._add_check(pl, 0, 4, self.vars["dark_plots"], "Dark plots")
        self._add_check(pl, 1, 4, self.vars["plot_heatmaps"], "Heatmaps")
        self._add_check(pl, 2, 4, self.vars["plot_fts"], "Fourier transforms")
        self._add_check(pl, 4, 0, self.vars["plot_landmark_shifts"], "Summary landmark")
        self._add_check(pl, 3, 2, self.vars["plot_indv_landmark_shifts"], "Indv landmark")
        self._build_group_stats_controls(pl, 5)

        ttk.Separator(root, orient="horizontal").pack(fill=tk.X, pady=6)
        self._bottom = ttk.Frame(root)
        self._bottom.pack(fill=tk.BOTH, expand=True)
        self._build_bottom(self._bottom)

        self.on_start = None
        self._is_running = False
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._restore_settings()
        self.minsize(940, 680)

    def _build_extra_buttons(self, parent):
        ttk.Button(parent, text="Back to Standard", command=self._go_back).grid(
            row=0, column=6, sticky="ew", pady=(0, 4))

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

    def __init__(self, parent, file_list, title="Select a file to test"):
        super().__init__(parent)
        self.title(title)
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


class _ImagePreviewWindow(tk.Toplevel):
    """Display a generated PIL image preview scaled to fit the screen."""

    def __init__(self, parent, pil_image, title, info_text):
        super().__init__(parent)
        self.title(title)
        self._source_image = pil_image
        self._photo = None

        max_w = max(parent.winfo_screenwidth() - 180, 400)
        max_h = max(parent.winfo_screenheight() - 220, 300)
        start_w = min(pil_image.width + 24, max_w)
        start_h = min(pil_image.height + 78, max_h)
        self.geometry(f"{start_w}x{start_h}")

        ttk.Label(self, text=info_text).pack(fill=tk.X, padx=10, pady=(8, 4), anchor="w")
        self.canvas = tk.Canvas(self, bg="#202020", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.canvas.bind("<Configure>", lambda _e: self._draw())
        self.after(50, self._draw)

    def _draw(self):
        from PIL import ImageTk

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 20 or ch < 20:
            return
        image = self._source_image.copy()
        image.thumbnail((cw - 8, ch - 8))
        self._photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._photo)


# ---------------------------------------------------------------------------
# Group order / rename dialog
# ---------------------------------------------------------------------------

class _GroupOrderDialog(tk.Toplevel):
    """Modal dialog to reorder groups and rename them for comparison graphs.

    Returns ``(order, labels)`` where ``order`` is the list of original group
    names in the chosen left-to-right order and ``labels`` maps each original
    name to its display name (only entries that were actually changed).
    """

    def __init__(self, parent, groups, initial_labels=None):
        super().__init__(parent)
        self.title("Group order & names")
        self.result = None
        self.transient(parent)
        self.grab_set()

        # Preserve the original name with each row so reordering never loses it.
        # Pre-fill display names from a prior edit when one is supplied.
        initial_labels = initial_labels or {}
        self._rows = [
            {"original": g, "name_var": tk.StringVar(value=initial_labels.get(g, g))}
            for g in groups
        ]

        ttk.Label(
            self,
            text="Reorder groups (top = left) and edit the display names:",
        ).pack(padx=10, pady=(10, 4), anchor="w")

        body = ttk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 6))

        self.listbox = tk.Listbox(body, height=min(max(len(groups), 4), 14),
                                  activestyle="dotbox", exportselection=False)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.listbox.bind("<<ListboxSelect>>", lambda _e: self._sync_entry())

        side = ttk.Frame(body)
        side.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 0))
        ttk.Button(side, text="Move up", command=lambda: self._move(-1)).pack(fill=tk.X)
        ttk.Button(side, text="Move down", command=lambda: self._move(1)).pack(fill=tk.X, pady=(4, 0))

        rename = ttk.Frame(self)
        rename.pack(fill=tk.X, padx=10, pady=(0, 6))
        ttk.Label(rename, text="Display name:").pack(side=tk.LEFT)
        self.name_entry = ttk.Entry(rename)
        self.name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))
        self.name_entry.bind("<KeyRelease>", lambda _e: self._commit_entry())

        btn = ttk.Frame(self)
        btn.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Button(btn, text="OK", command=self._ok).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn, text="Cancel", command=self._cancel).pack(side=tk.LEFT)

        self._refresh()
        if self._rows:
            self.listbox.selection_set(0)
            self._sync_entry()

        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.wait_window()

    def _selected_index(self):
        sel = self.listbox.curselection()
        return sel[0] if sel else None

    def _refresh(self):
        self.listbox.delete(0, tk.END)
        for row in self._rows:
            name = row["name_var"].get()
            label = name if name == row["original"] else f"{name}  (was {row['original']})"
            self.listbox.insert(tk.END, label)

    def _sync_entry(self):
        idx = self._selected_index()
        self.name_entry.delete(0, tk.END)
        if idx is not None:
            self.name_entry.insert(0, self._rows[idx]["name_var"].get())

    def _commit_entry(self):
        idx = self._selected_index()
        if idx is not None:
            self._rows[idx]["name_var"].set(self.name_entry.get())
            self._refresh()
            self.listbox.selection_set(idx)

    def _move(self, delta):
        idx = self._selected_index()
        if idx is None:
            return
        new_idx = idx + delta
        if not 0 <= new_idx < len(self._rows):
            return
        self._rows[idx], self._rows[new_idx] = self._rows[new_idx], self._rows[idx]
        self._refresh()
        self.listbox.selection_set(new_idx)
        self._sync_entry()

    def _ok(self):
        self._commit_entry()
        order = [row["original"] for row in self._rows]
        labels = {}
        for row in self._rows:
            name = row["name_var"].get().strip()
            if name and name != row["original"]:
                labels[row["original"]] = name
        self.result = (order, labels)
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
        self.minsize(880, 560)
        self.results_path = results_path
        self.default_dark_plots = self._infer_dark_plots(parent, results_path)
        self._summary_df = None
        self._metric_labels = []
        self._metric_lookup = {}
        self._load_summary_metrics()

        self.image_files = self._scan_image_files()

        self.current_index = 0
        self._photo = None

        _add_popup_header(self, parent, "Plot Viewer", "Results")

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

        # Keep the tree pane from opening at half-width; give it a sensible
        # initial share once the paned window has a real size.
        self.after(80, lambda: self._init_sash(pane))

        nav = ttk.Frame(self)
        nav.pack(fill=tk.X, padx=10, pady=(0, 6))
        ttk.Button(nav, text="< Prev", command=self._prev).pack(side=tk.LEFT)
        ttk.Button(nav, text="Next >", command=self._next).pack(side=tk.LEFT, padx=4)
        ttk.Button(nav, text="Group Comparison", command=self._make_group_comparison).pack(side=tk.LEFT, padx=(0, 4))
        # Stats options for the group comparison plots (see _make_group_comparison).
        self.compare_stats_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(nav, variable=self.compare_stats_var, text="Stats").pack(side=tk.LEFT)
        self.compare_test_var = tk.StringVar(value="Non-parametric")
        ttk.Combobox(nav, textvariable=self.compare_test_var, state="readonly", width=15,
                     values=["Non-parametric", "Parametric"]).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Button(nav, text="Group Correlations", command=self._make_group_correlations).pack(side=tk.LEFT, padx=(0, 4))
        self.count_label = ttk.Label(nav, text=f"{len(self.image_files)} plots")
        self.count_label.pack(side=tk.RIGHT)
        self.file_label = ttk.Label(nav, text="", wraplength=650, anchor="w")
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

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
        self.scatter_stats_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(scatter, variable=self.scatter_stats_var, text="Stats/trends").grid(row=0, column=5, padx=(0, 8))
        ttk.Button(scatter, text="Compute Plot", command=self._make_group_scatter).grid(row=0, column=6)
        self.scatter_status = ttk.Label(scatter, text="", wraplength=900)
        self.scatter_status.grid(row=1, column=0, columnspan=7, sticky="w", pady=(4, 0))

        if not self._metric_labels:
            self.scatter_x_combo.configure(state="disabled")
            self.scatter_y_combo.configure(state="disabled")
            self.scatter_status.configure(
                text="No summary metrics with numeric mean values were found.",
                foreground="red",
            )

    def _init_sash(self, pane):
        """Place the divider so the file tree opens at a readable fixed width
        rather than claiming half the window."""
        try:
            if pane.winfo_exists() and pane.winfo_width() > 1:
                pane.sashpos(0, 300)
        except Exception:
            pass

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
            'Peak Amp', 'Peak Rel Amp', 'Peak Apex Offset', 'Peak Apex',
            'Peak Baseline', 'Peak Width', 'Peak Area',
            'Rise Duration', 'Fall Duration', 'Rise minus Fall Duration',
            'Rising Slope', 'Falling Slope', 'Max Rising Slope', 'Max Falling Slope',
            'Rising/Falling Slope Ratio',
            'CCF % Phase Shift', 'CCF Shift', 'Peak-Apex Shift',
            'Rising-Edge Shift', 'Falling-Edge Shift',
            'Rise-Apex Shift Diff', 'Fall-Apex Shift Diff',
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
                add_stats=self.scatter_stats_var.get(),
            )
            out_dir = os.path.join(self.results_path, "group_scatter_graphs")
            os.makedirs(out_dir, exist_ok=True)
            suffix = " with stats" if self.scatter_stats_var.get() else ""
            output_path = os.path.join(out_dir, hf.sanitize_filename(f"{x_col} vs {y_col}{suffix}.png"))
            fig.savefig(output_path)
            self.scatter_status.configure(text=f"Saved: {os.path.basename(output_path)}", foreground="green")
            self._show_generated_plot(output_path)
        except Exception as e:
            self.scatter_status.configure(text=f"Could not create plot: {e}", foreground="red")

    def _make_group_comparison(self):
        try:
            import pandas as pd
            import waveanalysis.plotting as pt
            import waveanalysis.housekeeping.housekeeping_functions as hf

            csv_files = glob.glob(os.path.join(self.results_path, "!*_summary.csv"))
            if not csv_files:
                raise ValueError("No summary CSV found.")

            summary_df = pd.read_csv(sorted(csv_files)[-1])
            if 'Group Name' not in summary_df.columns:
                raise ValueError("Summary has no 'Group Name' column to compare.")

            present_groups = [
                g for g in dict.fromkeys(summary_df['Group Name'].dropna().tolist())
                if str(g).strip() != ""
            ]
            if len(present_groups) < 2:
                raise ValueError("At least two groups are required for a comparison.")

            dialog = _GroupOrderDialog(self, present_groups)
            if dialog.result is None:
                return
            order, labels = dialog.result

            stats_test = ('parametric' if self.compare_test_var.get().startswith('Parametric')
                          else 'nonparametric')
            log_params = {'Plotting errors': []}
            figs = pt.generate_group_comparison(
                summary_df=summary_df,
                log_params=log_params,
                dark_plots=self.default_dark_plots,
                group_order=order,
                group_labels=labels,
                add_stats=self.compare_stats_var.get(),
                stats_test=stats_test,
            )
            if not figs:
                raise ValueError("No metrics had data to compare.")

            out_dir = os.path.join(self.results_path, "group_comparison_graphs")
            os.makedirs(out_dir, exist_ok=True)
            hf.save_plots(figs, out_dir, group_by_metric=True)

            first_path = None
            for root, _dirs, files in os.walk(out_dir):
                for f in sorted(files, key=_natural_sort_key):
                    if f.lower().endswith(".png"):
                        first_path = os.path.join(root, f)
                        break
                if first_path:
                    break
            if first_path:
                self._show_generated_plot(first_path)
        except Exception as e:
            messagebox.showerror("Group Comparison", f"Could not create group comparison:\n{e}")

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

        _add_popup_header(self, parent, "Summary Table", "Results")

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


# ---------------------------------------------------------------------------
# Info / glossary panel
# ---------------------------------------------------------------------------

# Plain-language reference for the controls, plots, and metrics. Each section is
# a list of (title, body) entries rendered into a tab of the Info panel.
_HELP = {
    "Overview": [
        ("How the analysis works",
         "Wave Analysis measures oscillatory / excitable dynamics in time-lapse "
         "movies. Every mode follows the same pipeline: sample the image into small "
         "regions ('bins'), turn each bin into a mean-intensity trace over time, then "
         "quantify the timing and shape of that trace. The steps below show "
         "the pipeline on a single bin. The real output is these metrics pooled "
         "across every bin of every image."),
        ("1 · Sample the image into bins",
         "Each channel is tiled into small sampling regions: square boxes (standard), "
         "vertical lines (kymograph), or boxes within rolling time windows. Every bin "
         "is analyzed independently and produces one trace per channel.",
         "boxes"),
        ("2 · Period, from autocorrelation",
         "The top half of the graph shows the raw mean-intensity trace for one bin. "
         "A bin's mean intensity over time is the oscillatory signal. Correlating that "
         "signal with a time-shifted copy of itself (autocorrelation) reveals its "
         "dominant period, the first non-zero strong ACF peak (bottom).",
         "indvACF"),
        ("3 · Shift, from cross-correlation",
         "For multi-channel data, cross-correlating two channels measures the temporal "
         "shift (lead / lag) between them — the CCF Shift. It is also reported as a "
         "percentage of the period (CCF % Phase Shift), which normalizes across bins "
         "with different periods.",
         "indvCCF"),
        ("4 · Peak shape",
         "Individual peaks in the (smoothed) trace give amplitude, width, area, "
         "rise / fall durations and slopes. Savitzky–Golay smoothing suppresses "
         "spurious peaks in noisy data before this step.",
         "indvPeaks"),
        ("5 · Summarize across bins",
         "A single bin is just one sample. Pooling every bin of an image gives a "
         "distribution for each metric, and the per-image mean becomes one data point "
         "for group comparisons.",
         "groupComparison"),
        ("Which mode should I use?",
         "Standard: boxes on an en-face movie; best for a few clean wave periods. (Follows the pipeline above.) "
         "Kymograph: vertical lines on a kymograph image. A single medial slice gives "
         "finer temporal resolution than z-stacked en-face capture. Rolling: "
         "overlapping temporal sub-windows of a long movie; use when the recording "
         "spans tens to thousands of periods that change over time."),
        ("Kymograph sampling",
         "Instead of boxes, entire columns of the kymograph are segmented and measured. "
         "Produces the same metrics as the standard method, but uses a different bin creation workflow.",
         "lines"),
        ("Rolling analysis",
         "The movie is split into short, overlapping sub-movies; each is analyzed like "
         "a standard movie so you can track how wave properties drift over time.",
         "rolling_analysis"),
    ],
    "Controls": [
        ("Input folder / Browse",
         "The folder of .tif movies to analyze (all non-hidden .tif files are "
         "processed). You can also drag a folder onto the window. The last-used "
         "folder is remembered between sessions."),
        ("Group names",
         "Comma-separated labels for grouping files in the comparison graphs. A "
         "file joins a group when the group's text appears in its filename. Leave "
         "blank to skip grouping. Example: 'control,drug'."),
        ("Order & names…",
         "Set the left-to-right order of groups in the comparison graphs and give "
         "each a display name, without renaming any files."),
        ("Box size (px) — standard",
         "Side length of each square sampling box (a 'bin'). The image is tiled "
         "into boxes and each box produces one mean-intensity trace per channel "
         "that is analyzed independently."),
        ("Line width (px) — kymograph",
         "Width, in pixels, of each vertical sampling line across the kymograph. "
         " The width is the number of neighboring columns that are averaged together to produce "
         " each line's mean-intensity trace. Each line yields one trace per channel."),
        ("Box / Line shift (px)",
         "Step between neighboring bins. Setting it equal to the box size / line "
         "width tiles without overlap; a smaller value produces overlapping, "
         "denser bins. A larger value skips bins and may miss some wave events, but "
         "reduces the number of bins and speeds up analysis."),
        ("Subframe size / roll — rolling",
         "Rolling analysis splits the movie into temporal sub-windows. 'Subframe "
         "size' is the number of frames per window; 'Subframe roll' is how many "
         "frames the window advances each step."),
        ("Live Injection — rolling",
         "Optional overlay for experiments where something is added mid-recording "
         "(e.g. a drug injection). 'Injection frame' draws a vertical dashed line "
         "on every rolling plot at that time point — enter the frame number where "
         "the injection occurred (leave 0 to omit the line). The 'Overlay signal "
         "Ch1–Ch4' checkboxes additionally plot each selected channel's mean "
         "intensity over the movie, rescaled to the metric's y-range, so you can "
         "line up changes in a wave metric against when a channel's signal (e.g. a "
         "co-imaged injection marker) rises or falls. Only applies to rolling "
         "analysis."),
        ("ACF peak threshold",
         "Minimum relative prominence (0–1) a peak in the autocorrelation must "
         "have to count when detecting the period. Higher = stricter. Typical 0.1."),
        ("CCF peak threshold",
         "Minimum relative prominence (0–1) for a peak in the cross-correlation "
         "when measuring the temporal shift between two channels."),
        ("Peak prominence fraction",
         "Minimum prominence — as a fraction of the trace's intensity range — "
         "for a peak in the raw signal to be included in peak-shape analysis."),
        ("Small shifts correction",
         "The inter-channel shift is read from the cross-correlation peak nearest "
         "zero lag. Because the signals are periodic, a shift of, say, +0.9 period "
         "is indistinguishable from a −0.1 period lead, and noise or phase aliasing "
         "can push a genuinely small lead/lag out to a near-full-period value. When "
         "enabled, any measured shift whose magnitude exceeds 60% of the average "
         "period is wrapped back toward zero by ±1 period (subtract a period from a "
         "positive shift, add one to a negative shift), collapsing these aliased "
         "readings onto the small lead/lag they actually represent. "
         "Only use this for signals you expect to be closely matched (small true "
         "offset). If channels can genuinely be offset by more than half a period, "
         "leave it off — it would wrap real large shifts and bias the measurement."),
        ("Smoothing (Ch1–Ch4, CCF)",
         "Savitzky–Golay smoothing applied to each channel's trace (and the CCF) "
         "before analysis. 'win' is the window length in frames (odd number); "
         "'poly' is the polynomial order. 'Enable smoothing' is the master toggle."
         " Window length should be smaller than the shortest expected period, "
         "and should be the same across all channels if you want to compare their timing directly."),
        ("Channel Names (optional)",
         "Names for channels shown in plot labels only. Blank fields fall "
         "back to Ch1–Ch4. Does not affect the analysis or filenames."),
        ("Landmark Edge Height (Rise/fall height)",
         "The height — as a percentage of each peak's amplitude — at which the "
         "rising and falling 'landmarks' are placed. These landmarks define the "
         "rise/fall durations and edge-shift metrics. 50% ≈ full-width-half-max."
         " 100% = peak apex; 0% = baseline. The landmarks are drawn on the plots."
         # TODO: add a new image here showing the landmarks on a peak,
        ),
        ("Plot Options",
         "Choose which figure sets to generate (see the Plots tab). 'Summary' plots "
         "aggregate across all bins of an image; 'Indv' plots are one-per-bin. "
         "'Dark plots' renders figures on a dark background. See the Plots tab for a "
         "glossary of each figure type."),
        ("Test File",
         "Run the full analysis on a single chosen .tif (no settings are saved). A "
         "fast way to check parameters before processing a whole folder."),
        ("Preview Bins",
         "Overlay the current box/line bin grid on one image so you can see exactly "
         "where signals will be sampled before running."),
        ("Load Results",
         "Open an existing results folder (a 0_signalProcessing-… folder)"
         " to browse its plots and summary without re-running."),
        ("Rolling / Kymograph",
         "Switch the analysis mode."),
    ],
    # Grouped into sub-tabs (dict value) rather than one long scroll. Each key
    # becomes a sub-tab inside the Plots tab; see _build_section.
    "Plots": {
        "Summary": [
            ("About summary plots",
             "One figure per image, pooling every bin of that image into a "
             "distribution or mean. These are the day-to-day outputs."),
            ("Summary ACFs",
             "Per-channel mean autocorrelation across all bins of an image, used to "
             "read out the dominant period.",
             "summaryACF"),
            ("Summary CCFs",
             "Mean cross-correlation between each channel pair, summarizing the "
             "typical temporal shift between channels.",
             "summaryCCF"),
            ("Summary peaks",
             "Distribution of peak-shape metrics (amplitude, width, area, etc.) "
             "pooled over all bins of an image.",
             "summaryPeaks"),
            ("Summary slopes",
             "Distribution of the rising/falling-edge slope metrics across an "
             "image's bins — how steep and how symmetric the wave edges are.",
             "summarySlopes"),
            ("Summary edge times",
             "Distribution of the rise- and fall-edge timing (rise/fall durations) "
             "pooled over the bins of an image.",
             "summaryEdgeTimes"),
            ("Summary landmark shifts",
             "Per-channel-pair inter-channel timing offsets measured from matched "
             "peak landmarks (apex, rising edge, falling edge), pooled over the "
             "image's bins. A landmark-based alternative to the CCF shift.",
             "summaryLandmarkShifts"),
            ("Lag profile",
             "How the cross-correlation shift between two channels varies across the "
             "image's bins — a profile of lead/lag rather than a single summary "
             "value.",
             "lagProfile"),
            ("Metric correlations (per image)",
             "Spearman correlation heatmap between metrics computed across the bins "
             "of a single image — which measurements move together.",
             "summaryMetricCorr"),
        ],
        "Individual": [
            ("About individual-bin plots",
             "One figure per bin — the same analyses as the summary plots, drawn "
             "separately for each bin. Useful for QC but they produce many figures "
             "and significantly increase analysis time, so sample sparsely (large "
             "box / line shift) or leave them off for routine runs."),
            ("Indv ACFs",
             "The autocorrelation for a single bin, with the detected period marked.",
             "indvACF"),
            ("Indv CCFs",
             "The cross-correlation between two channels for a single bin, with the "
             "detected shift marked.",
             "indvCCF"),
            ("Indv peaks",
             "Peak-shape analysis for a single bin's trace: detected peaks with "
             "their amplitude, width, and baseline annotated.",
             "indvPeaks"),
            ("Indv landmark shift",
             "Per-bin inter-channel timing from matched peak landmarks, showing the "
             "apex / rising-edge / falling-edge offsets between two channels.",
             "indvLandmarkShift"),
            ("Fourier transforms",
             "FFT power spectrum of each bin's trace, annotated with the top three "
             "spectral peaks and a reference line at the ACF-detected period (with "
             "the nearest FFT peak highlighted).",
             "indvFFT"),
        ],
        "Spatial": [
            ("About spatial maps",
             "Values placed back onto the image at the location of each bin, so you "
             "can see where in the field a metric is high or low."),
            ("Heatmaps",
             "Each metric painted onto the image at the location of its bin.",
             "heatmap"),
            ("Bin reference chart",
             "A companion map that labels each bin with its ID, so heatmap features "
             "can be traced back to specific bins.",
             "heatmapBinRef"),
        ],
        "Group": [
            ("About group plots",
             "One point per image, compared across the folder using the group "
             "labels. These are the between-condition comparisons."),
            ("Group comparison",
             "Box-and-swarm plot per group for each metric, with the sample size in "
             "the axis label and an optional significance test. Use 'Group stats' to "
             "show or hide the test, and choose the test family: non-parametric "
             "(Mann–Whitney for two groups, Kruskal–Wallis for more — no "
             "distribution assumption) or parametric (t-test / one-way ANOVA, which "
             "assume roughly normal groups).",
             "groupComparison"),
            ("Group correlations",
             "Spearman cross-metric correlation heatmap computed separately for each "
             "group, where each row is one image's per-image mean.",
             "groupCorr"),
            ("Group scatter",
             "Scatter of any two summary metrics with one point per image, groups "
             "overlaid by color/marker, with an optional trend line and statistics.",
             "groupScatter"),
        ],
        "Quality": [
            ("About quality plots",
             "Diagnostics for how much you should trust the numbers — how many bins "
             "were usable and how variable they were."),
            ("Detection quality",
             "Per group, the percentage of bins where no period/peak/shift could be "
             "measured. High values mean the group's metrics rest on few usable "
             "bins; a caution band marks >50%.",
             "qualityDetection"),
            ("Coverage",
             "Number of usable bins per image ('Num Bins'), per group. Flags "
             "under-sampled images whose per-image means are statistically thin.",
             "qualityCoverage"),
            ("Per-image reliability",
             "Caterpillar plot: each image's per-image mean ± within-image "
             "variability, blocked by group. Long error bars flag images whose mean "
             "rests on highly variable bins.",
             "qualityReliability"),
        ],
    },
    "Metrics": [
        ("Units",
         "AU = arbitrary intensity units. s = seconds. Summary values are means "
         "across an image's bins."),
        ("Period (s)",
         "Dominant oscillation period of the trace, from the first peak of the "
         "autocorrelation."),
        ("Peak Amp (AU)",
         "Peak height above its local baseline, in intensity units."),
        ("Peak Rel Amp",
         "Peak amplitude relative to the baseline (dimensionless)."),
        ("Peak Width (s)",
         "Peak width at half maximum (FWHM)."),
        ("Peak Apex / Peak Baseline (AU)",
         "Apex (maximum) and baseline (minimum) intensity of the peak."),
        ("Peak Apex Offset (s)",
         "Offset of the peak apex from the midpoint between its rise and fall "
         "landmarks — a measure of asymmetry."),
        ("Peak Area (AU·s)",
         "Area under the peak above its local baseline, integrated over time."),
        ("Rise Duration (s)",
         "Time from the rising landmark (at the chosen edge height) up to the apex."),
        ("Fall Duration (s)",
         "Time from the apex down to the falling landmark."),
        ("Rise minus Fall Duration (s)",
         "Rise duration minus fall duration; sign indicates asymmetry direction."),
        ("Rising / Falling Slope",
         "Mean rate of change on the rising / falling edge (intensity per second)."),
        ("Max Rising / Falling Slope",
         "Steepest instantaneous slope on the rising / falling edge."),
        ("Rising/Falling Slope Ratio",
         "Ratio of rising to falling slope; 1 = symmetric edges."),
        ("CCF Shift (s)",
         "Temporal shift between two channels from the cross-correlation (CCF) peak. "
         "The sign indicates which channel leads."),
        ("CCF % Phase Shift",
         "CCF shift expressed as a percentage of the period (phase)."),
        ("Peak-Apex Shift (s)",
         "Inter-channel shift between the channels' peak-apex times (landmark-based, "
         "not from the CCF)."),
        ("Rising-Edge / Falling-Edge Shift (s)",
         "Inter-channel shift measured at the rising / falling edge landmarks."),
        ("Rise-Apex / Fall-Apex Shift Diff (s)",
         "Edge-landmark shift minus the peak-apex shift — whether the edges and apex "
         "shift between channels by the same amount."),
        ("Num Bins (quality)",
         "Number of usable bins contributing to an image's per-image means."),
        ("Pcnt No … (quality)",
         "Percentage of bins in an image where a given measurement (period, peak, "
         "or shift) could not be detected."),
    ],
    "Preparing Data": [
        ("Before you analyze",
         "A few minutes of pre-processing makes the measurements far more "
         "trustworthy. Wave Analysis measures the whole image exactly as given — it "
         "does no drift, crop, or bleach correction for you."),
        ("Correct drift first",
         "Any significant 2-D drift smears the wave dynamics. Register / stabilize "
         "the movie before analysis."),
        ("Crop out black / background regions",
         "Empty areas (from drift correction or true background) add meaningless "
         "bins. Crop them out, or crop each region of interest into its own file."),
        ("Bleaching & z-drift",
         "Both bias amplitude and width measurements. It is best to avoid them at "
         "acquisition, since bleach-correction algorithms can add their own "
         "artifacts. Correct only if necessary."),
        ("File format & metadata",
         "Files should be TIFFs in tzcyx order. The tool reads ImageJ metadata to "
         "identify the time, channel, and z axes and to get pixel size and frame "
         "interval. If the frame interval or pixel size is missing it defaults to 1 "
         "— so periods and shifts would come out in frames, not seconds."),
        ("Z-stacks",
         "Files with more than one z plane are max-projected along z before "
         "analysis."),
        ("One image = one region (for now)",
         "The tool analyzes the entire image. To isolate a cell or exclude "
         "background, crop it into a separate file. Mask-based sub-region analysis "
         "is planned."),
        ("Dataset character",
         "Standard and kymograph modes want a few consistent periods; rolling wants "
         "many periods that vary over time."),
        ("Picking a box / line size",
         "Bins should be large enough to average out noise but small enough not to "
         "span multiple structures. A good empirical check: in FIJI draw a box, open "
         "Image > Stacks > Plot Z-axis Profile, click Live, and resize until the "
         "trace captures the dynamics you expect."),
    ],
    "Output": [
        ("Results folder",
         "Each run creates a 0_signalProcessing-<timestamp> folder inside your "
         "source folder, so runs never overwrite each other."),
        ("Pooled summary spreadsheet",
         "!<timestamp>_summary.csv at the top of the results folder holds one row "
         "per image (the per-image means) for every metric — the file you take to "
         "statistics. The leading ! keeps it sorted to the top."),
        ("Per-image folders",
         "One subfolder per input file holds that image's summary plots, per-bin "
         "measurement CSVs, and any heatmaps / FFT / individual-bin plots you "
         "enabled."),
        ("Group comparison graphs",
         "If you set group names, group_comparison_graphs holds the box-and-swarm "
         "comparison plots plus the group correlation and scatter figures."),
        ("Quality assessment graphs",
         "quality_assessment_graphs holds the detection-quality, coverage, "
         "effective-N, and per-image reliability plots — check these before trusting "
         "the comparisons."),
        ("Mean parameter measurements",
         "mean_parameter_measurements holds the per-metric mean CSVs behind the "
         "group plots."),
        ("Run log",
         "!log-<timestamp>.txt records the exact parameters used, plus any files "
         "that didn't match a group."),
        ("Re-opening a run",
         "Use Load Results in the GUI to browse an existing 0_signalProcessing-… "
         "folder's plots and summary without re-running."),
    ],
    "Tips": [
        ("Test before you batch",
         "Run Test File on one movie (nothing is saved) and use Preview Bins to see "
         "exactly where signals will be sampled before processing a whole folder."),
        ("Most bins report nothing / low detection quality",
         "Usually the bins are too small, the movie too short, or a threshold too "
         "high. Enlarge the box / line, include more periods, or lower the ACF / "
         "peak thresholds. The quality plots show which images are affected."),
        ("A shift jumped to about one whole period",
         "Near-zero shifts can wrap to a full period. Enable 'Small shifts "
         "correction' — but only for closely matched signals (e.g. the same protein "
         "in two fluorophores), since it biases genuinely large shifts."),
        ("Peaks look spurious / over-counted",
         "Increase per-channel smoothing (larger window) or raise the peak "
         "prominence fraction so small wiggles are ignored."),
        ("Comparing channel timing",
         "Keep the smoothing window the same across channels, and smaller than the "
         "shortest expected period — unequal smoothing shifts the apparent timing "
         "between channels."),
        ("Individual-bin plots are slow / flood the folder",
         "They write one figure per bin. Sample sparsely (large box / line shift) "
         "when you turn them on, or leave them off for routine runs."),
        ("Period or shift is in frames, not seconds",
         "The frame interval wasn't in the metadata, so it defaulted to 1. Fix the "
         "TIFF metadata and re-run."),
    ],
}


# Example figures embedded in the Info panel live inside the installed package
# (see pyproject build include) so they ship with pip/uv installs, not just the
# git checkout. Each entry may name a figure "stem"; the light/dark variant is
# chosen to match the active theme.
_HELP_ASSET_DIR = os.path.join(os.path.dirname(__file__), "help_assets")

# Order of the Info panel tabs.
_HELP_SECTIONS = ("Overview", "Controls", "Plots", "Metrics",
                  "Preparing Data", "Output", "Tips")

# Width (px) example figures are scaled to inside the panel.
_HELP_IMAGE_WIDTH = 600


def _help_image_path(stem, theme):
    """Path to a bundled help figure for the given theme, or None if missing."""
    for ext in ("png", "jpg"):
        p = os.path.join(_HELP_ASSET_DIR, f"{stem}_{theme}.{ext}")
        if os.path.exists(p):
            return p
    return None


class _InfoPanel(tk.Toplevel):
    """Tabbed help panel explaining the pipeline, controls, plots, and metrics."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Wave Analysis — Info & Glossary")
        self.geometry("840x640")
        self.transient(parent)

        self._theme = getattr(parent, "_theme", "light")
        if self._theme not in ("light", "dark"):
            self._theme = "light"
        self._photo_refs = []  # keep PhotoImages alive for the panel's lifetime

        _add_popup_header(self, parent, "Info & Glossary", "Reference")
        ttk.Label(self, text="How the pipeline works, plus every control, plot, and metric.",
                  foreground=_RETRO["header_sub"]).pack(anchor="w", padx=12, pady=(6, 6))

        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 6))
        for section in _HELP_SECTIONS:
            frame = ttk.Frame(nb)
            nb.add(frame, text=section)
            self._build_section(frame, _HELP[section])

        ttk.Button(self, text="Close", command=self.destroy).pack(pady=(0, 10))

    def _load_photo(self, stem):
        """Return a theme-matched, panel-width PhotoImage for a figure, or None."""
        path = _help_image_path(stem, self._theme)
        if path is None:
            return None
        try:
            from PIL import Image, ImageTk
            image = Image.open(path)
            if image.width > _HELP_IMAGE_WIDTH:
                h = round(image.height * _HELP_IMAGE_WIDTH / image.width)
                image = image.resize((_HELP_IMAGE_WIDTH, h), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)
        except Exception:
            return None
        self._photo_refs.append(photo)
        return photo

    def _build_section(self, parent, content):
        # A section is either a flat list of entries, or a dict mapping
        # sub-tab name -> entries (rendered as a nested notebook to avoid one
        # very long scroll, e.g. the Plots tab).
        if isinstance(content, dict):
            sub_nb = ttk.Notebook(parent)
            sub_nb.pack(fill=tk.BOTH, expand=True)
            for name, entries in content.items():
                frame = ttk.Frame(sub_nb)
                sub_nb.add(frame, text=name)
                self._build_entries(frame, entries)
            return
        self._build_entries(parent, content)

    def _build_entries(self, parent, entries):
        txt = scrolledtext.ScrolledText(
            parent, wrap=tk.WORD, relief="sunken", borderwidth=2,
            highlightthickness=0, background=_RETRO["log_bg"], foreground=_RETRO["log_fg"],
            padx=12, pady=10,
        )
        txt.pack(fill=tk.BOTH, expand=True)
        fam = tkfont.nametofont("TkDefaultFont").actual("family")
        txt.tag_configure("h", font=(fam, 14, "bold"), foreground=_RETRO["select"],
                          spacing1=10, spacing3=2)
        txt.tag_configure("body", font=(fam, 13), spacing3=6, lmargin1=6, lmargin2=6)
        txt.tag_configure("img", spacing1=4, spacing3=8, lmargin1=6, lmargin2=6)
        for entry in entries:
            title, body = entry[0], entry[1]
            stem = entry[2] if len(entry) > 2 else None
            txt.insert(tk.END, title + "\n", "h")
            txt.insert(tk.END, body + "\n", "body")
            if stem:
                photo = self._load_photo(stem)
                if photo is not None:
                    txt.image_create(tk.END, image=photo)
                    txt.insert(tk.END, "\n", "img")
        txt.configure(state="disabled")
