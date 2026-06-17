import numpy as np
import matplotlib.pyplot as plt
from waveanalysis.housekeeping.housekeeping_functions import get_channel_name, get_channel_combo_name

# Per-channel metrics: key in img_metrics -> colorbar label
def _edge_percent(edge_height_fraction: float) -> str:
    return f'{int(round(float(edge_height_fraction) * 100))}%'

def _channel_metric_labels(edge_height_fraction: float) -> dict:
    pct = _edge_percent(edge_height_fraction)
    return {
    'Period':       'Detected period (s)',
    'Peak Amp':     'Peak amplitude (AU)',
    'Peak Rel Amp': 'Relative peak amplitude',
    'Peak Width':   'Peak width, FWHM (s)',
    'Peak Max':     'Peak maximum intensity (AU)',
    'Peak Min':     'Peak baseline/minimum intensity (AU)',
    'Peak Offset':  'Peak apex offset from midpoint (s)',
    'Peak Area':    'Peak area above local baseline (AU)',
        'Rise Time':    f'Rise duration, {pct} to apex (s)',
        'Fall Time':    f'Fall duration, apex to {pct} (s)',
        'Rise-Fall Time': 'Rise duration minus fall duration (s)',
    }

# Per channel-combo metrics: key in img_metrics -> colorbar label
def _combo_metric_labels(edge_height_fraction: float) -> dict:
    pct = _edge_percent(edge_height_fraction)
    return {
    'Shift':          'CCF shift (s)',
    '% Phase Shift':  'CCF shift as phase (% of period)',
    'Peak Shift':     'Peak-apex shift (s)',
        'Rise Shift':     f'{pct} rising-edge shift (s)',
        'Fall Shift':     f'{pct} falling-edge shift (s)',
    'Rise-Peak Diff': 'Rise shift minus peak shift (s)',
    'Fall-Peak Diff': 'Fall shift minus peak shift (s)',
    }


def plot_metric_heatmaps_workflow(
    img_metrics: dict,
    img_props: dict,
    image_array: np.ndarray,
    dark_plots: bool = False,
) -> dict:
    """
    Generate overlay heatmaps for every metric + a Bin ID Reference Map with grids.
    """
    num_channels   = img_props['num_channels']
    num_x_bins     = img_props['num_x_bins']
    num_y_bins     = img_props['num_y_bins']
    box_size       = img_props['box_size']
    step           = img_props['step']
    pixel_size     = img_props['pixel_size'][0]   # x-axis pixel size
    pixel_unit     = img_props['pixel_unit']
    channel_combos = img_props.get('channel_combos', [])
    channel_names  = img_props.get('channel_names')
    edge_height_fraction = img_props.get('edge_height_fraction', 0.5)

    # First frame of each channel as background
    bg_images = {ch: image_array[0, 0, ch, :, :] for ch in range(num_channels)}

    heatmap_figs = {}

    # --- 1. GENERATE BIN REFERENCE CHART ---
    total_bins = num_x_bins * num_y_bins
    bin_ids = np.arange(1, total_bins + 1, dtype=float)

    # Use Channel 1 as the background for the reference chart
    ref_panels = [(bin_ids, bg_images[0], 'Bin Reference Map (IDs)')]
    heatmap_figs['Bin Reference Chart'] = _return_metric_figure(
        panels=ref_panels,
        vmin=1,
        vmax=total_bins,
        num_x_bins=num_x_bins,
        num_y_bins=num_y_bins,
        box_size=box_size,
        step=step,
        metric_title='Bin Reference Index',
        cbar_label='Bin ID Number',
        pixel_size=pixel_size,
        pixel_unit=pixel_unit,
        dark_plots=dark_plots,
        show_bin_ids=True # Trigger text labels and grid lines
    )

    # --- 2. PER-CHANNEL METRICS ---
    for metric_name, cbar_label in _channel_metric_labels(edge_height_fraction).items():
        if metric_name not in img_metrics:
            continue
        data = img_metrics[metric_name]
        valid_all = data[np.isfinite(data)]
        vmin = float(np.min(valid_all)) if valid_all.size > 0 else 0.0
        vmax = float(np.max(valid_all)) if valid_all.size > 0 else 1.0

        panels = [
            (data[ch], bg_images[ch], get_channel_name(channel_names, ch))
            for ch in range(num_channels)
        ]
        heatmap_figs[f'{metric_name} Heatmap'] = _return_metric_figure(
            panels=panels,
            vmin=vmin,
            vmax=vmax,
            num_x_bins=num_x_bins,
            num_y_bins=num_y_bins,
            box_size=box_size,
            step=step,
            metric_title=metric_name,
            cbar_label=cbar_label,
            pixel_size=pixel_size,
            pixel_unit=pixel_unit,
            dark_plots=dark_plots,
            show_bin_ids=False # Ensure other metrics don't have lines
        )

    # --- 3. PER-COMBO METRICS ---
    for metric_name, cbar_label in _combo_metric_labels(edge_height_fraction).items():
        if metric_name not in img_metrics:
            continue
        for combo_idx, (ch1, ch2) in enumerate(channel_combos):
            combo_label = f'Ch{ch1 + 1}-Ch{ch2 + 1}'
            combo_display = get_channel_combo_name(channel_names, [ch1, ch2])
            values = img_metrics[metric_name][combo_idx]
            bg = (bg_images[ch1].astype(float) + bg_images[ch2].astype(float)) / 2
            valid = values[np.isfinite(values)]
            vmin = float(np.min(valid)) if valid.size > 0 else 0.0
            vmax = float(np.max(valid)) if valid.size > 0 else 1.0

            panels = [(values, bg, combo_display)]
            heatmap_figs[f'{combo_label} {metric_name} Heatmap'] = _return_metric_figure(
                panels=panels,
                vmin=vmin,
                vmax=vmax,
                num_x_bins=num_x_bins,
                num_y_bins=num_y_bins,
                box_size=box_size,
                step=step,
                metric_title=f'{combo_display} {metric_name}',
                cbar_label=cbar_label,
                pixel_size=pixel_size,
                pixel_unit=pixel_unit,
                dark_plots=dark_plots,
                show_bin_ids=False
            )

    return heatmap_figs


def plot_metric_heatmaps_kymo_workflow(
    img_metrics: dict,
    img_props: dict,
    image_array: np.ndarray,
    dark_plots: bool = False,
) -> dict:
    """
    Generate 1-D strip heatmaps for kymograph data.

    Each metric is visualised as a coloured strip along the spatial (column)
    axis, overlaid at the top of the kymograph image.
    """
    num_channels   = img_props['num_channels']
    num_bins       = img_props['num_bins']
    line_width     = img_props['line_width']
    step           = img_props['step']
    pixel_size     = img_props['pixel_size'][0]
    pixel_unit     = img_props['pixel_unit']
    channel_combos = img_props.get('channel_combos', [])
    channel_names  = img_props.get('channel_names')
    edge_height_fraction = img_props.get('edge_height_fraction', 0.5)

    heatmap_figs = {}

    # --- 1. BIN REFERENCE STRIP ---
    bin_ids = np.arange(1, num_bins + 1, dtype=float)
    ref_panels = [(bin_ids, image_array[0], 'Bin Reference Map (IDs)')]
    heatmap_figs['Bin Reference Chart'] = _return_kymo_metric_figure(
        panels=ref_panels,
        vmin=1, vmax=num_bins,
        num_bins=num_bins, line_width=line_width, step=step,
        metric_title='Bin Reference Index',
        cbar_label='Bin ID Number',
        pixel_size=pixel_size, pixel_unit=pixel_unit,
        dark_plots=dark_plots, show_bin_ids=True,
    )

    # --- 2. PER-CHANNEL METRICS ---
    for metric_name, cbar_label in _channel_metric_labels(edge_height_fraction).items():
        if metric_name not in img_metrics:
            continue
        data = img_metrics[metric_name]
        valid_all = data[np.isfinite(data)]
        vmin = float(np.min(valid_all)) if valid_all.size > 0 else 0.0
        vmax = float(np.max(valid_all)) if valid_all.size > 0 else 1.0

        panels = [
            (data[ch], image_array[ch], get_channel_name(channel_names, ch))
            for ch in range(num_channels)
        ]
        heatmap_figs[f'{metric_name} Heatmap'] = _return_kymo_metric_figure(
            panels=panels,
            vmin=vmin, vmax=vmax,
            num_bins=num_bins, line_width=line_width, step=step,
            metric_title=metric_name, cbar_label=cbar_label,
            pixel_size=pixel_size, pixel_unit=pixel_unit,
            dark_plots=dark_plots, show_bin_ids=False,
        )

    # --- 3. PER-COMBO METRICS ---
    for metric_name, cbar_label in _combo_metric_labels(edge_height_fraction).items():
        if metric_name not in img_metrics:
            continue
        for combo_idx, (ch1, ch2) in enumerate(channel_combos):
            combo_label = f'Ch{ch1 + 1}-Ch{ch2 + 1}'
            combo_display = get_channel_combo_name(channel_names, [ch1, ch2])
            values = img_metrics[metric_name][combo_idx]
            bg = (image_array[ch1].astype(float) + image_array[ch2].astype(float)) / 2
            valid = values[np.isfinite(values)]
            vmin = float(np.min(valid)) if valid.size > 0 else 0.0
            vmax = float(np.max(valid)) if valid.size > 0 else 1.0

            panels = [(values, bg, combo_display)]
            heatmap_figs[f'{combo_label} {metric_name} Heatmap'] = _return_kymo_metric_figure(
                panels=panels,
                vmin=vmin, vmax=vmax,
                num_bins=num_bins, line_width=line_width, step=step,
                metric_title=f'{combo_display} {metric_name}',
                cbar_label=cbar_label,
                pixel_size=pixel_size, pixel_unit=pixel_unit,
                dark_plots=dark_plots, show_bin_ids=False,
            )

    return heatmap_figs


def _return_kymo_metric_figure(
    panels: list,
    vmin: float,
    vmax: float,
    num_bins: int,
    line_width: int,
    step: int,
    metric_title: str,
    cbar_label: str,
    pixel_size: float,
    pixel_unit: str,
    dark_plots: bool,
    show_bin_ids: bool = False,
) -> plt.Figure:
    """Render a 1-D colour strip overlaid on kymograph background(s)."""
    n_panels = len(panels)
    height_px, width_px = panels[0][1].shape

    scale   = 6.0 / max(height_px, width_px)
    panel_w = width_px * scale
    panel_h = height_px * scale
    hist_h  = 1.4
    figsize = (n_panels * panel_w + 1.4, panel_h + hist_h)

    cmap = plt.cm.inferno.copy()
    cmap.set_bad(alpha=0)

    style = 'dark_background' if dark_plots else 'default'
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(
            2, n_panels,
            height_ratios=[panel_h, hist_h],
            hspace=0.12, wspace=0.08,
        )
        hmap_axes = [fig.add_subplot(gs[0, i]) for i in range(n_panels)]
        hist_axes = [fig.add_subplot(gs[1, i]) for i in range(n_panels)]

        last_im = None
        for ax_h, ax_d, (values, bg_image, panel_title) in zip(hmap_axes, hist_axes, panels):
            last_im = _draw_kymo_heatmap_panel(
                ax=ax_h, values=values, bg_image=bg_image,
                vmin=vmin, vmax=vmax,
                num_bins=num_bins, line_width=line_width, step=step,
                panel_title=panel_title, cmap=cmap,
                pixel_size=pixel_size, pixel_unit=pixel_unit,
                show_bin_ids=show_bin_ids,
            )
            _draw_hist_panel(ax=ax_d, values=values, vmin=vmin, vmax=vmax, cbar_label=cbar_label)

        cbar = fig.colorbar(last_im, ax=hmap_axes, fraction=0.03, pad=0.02)
        cbar.set_label(cbar_label)
        fig.suptitle(metric_title)
        fig.tight_layout()
        plt.close(fig)

    return fig


def _draw_kymo_heatmap_panel(
    ax,
    values: np.ndarray,
    bg_image: np.ndarray,
    vmin: float,
    vmax: float,
    num_bins: int,
    line_width: int,
    step: int,
    panel_title: str,
    cmap,
    pixel_size: float,
    pixel_unit: str,
    show_bin_ids: bool = False,
):
    """Draw a kymograph background with a 1-D colour strip overlay along the spatial axis."""
    height_px, width_px = bg_image.shape

    strip_height = max(int(height_px * 0.08), 6)
    ind = line_width // 2
    half_step = step / 2.0

    grid_1d = np.ma.masked_invalid(values)
    strip = np.tile(grid_1d, (strip_height, 1))

    col_start = ind - half_step
    col_end = ind + (num_bins - 1) * step + half_step
    row_start = 0
    row_end = strip_height

    ax.imshow(
        bg_image, cmap='gray', origin='upper', aspect='auto',
        vmin=np.percentile(bg_image, 1), vmax=np.percentile(bg_image, 99),
    )

    im = ax.imshow(
        strip, cmap=cmap, origin='upper', aspect='auto',
        interpolation='nearest',
        extent=[col_start, col_end, row_end, row_start],
        vmin=vmin, vmax=vmax, alpha=0.75,
    )

    if show_bin_ids:
        y_center = strip_height / 2
        for b in range(num_bins):
            x_center = ind + b * step
            ax.text(x_center, y_center, str(b + 1),
                    color='white', fontsize=5, ha='center', va='center', weight='bold')

    _add_scale_bar(ax, width_px, height_px, pixel_size, pixel_unit)
    ax.set_title(panel_title, fontsize=9)
    ax.set_axis_off()
    return im


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _return_metric_figure(
    panels: list,
    vmin: float,
    vmax: float,
    num_x_bins: int,
    num_y_bins: int,
    box_size: int,
    step: int,
    metric_title: str,
    cbar_label: str,
    pixel_size: float,
    pixel_unit: str,
    dark_plots: bool,
    show_bin_ids: bool = False,
) -> plt.Figure:
    n_panels  = len(panels)
    height_px, width_px = panels[0][1].shape

    scale    = 6.0 / max(height_px, width_px)
    panel_w  = width_px  * scale
    panel_h  = height_px * scale
    hist_h   = 1.4
    figsize  = (n_panels * panel_w + 1.4, panel_h + hist_h)

    cmap = plt.cm.inferno.copy()
    cmap.set_bad(alpha=0)

    style = 'dark_background' if dark_plots else 'default'
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        gs  = fig.add_gridspec(
            2, n_panels,
            height_ratios=[panel_h, hist_h],
            hspace=0.12,
            wspace=0.08,
        )
        hmap_axes = [fig.add_subplot(gs[0, i]) for i in range(n_panels)]
        hist_axes = [fig.add_subplot(gs[1, i]) for i in range(n_panels)]

        last_im = None
        for ax_h, ax_d, (values, bg_image, panel_title) in zip(hmap_axes, hist_axes, panels):
            last_im = _draw_heatmap_panel(
                ax=ax_h,
                values=values,
                bg_image=bg_image,
                vmin=vmin,
                vmax=vmax,
                num_x_bins=num_x_bins,
                num_y_bins=num_y_bins,
                box_size=box_size,
                step=step,
                panel_title=panel_title,
                cmap=cmap,
                pixel_size=pixel_size,
                pixel_unit=pixel_unit,
                show_bin_ids=show_bin_ids,
            )
            _draw_hist_panel(
                ax=ax_d,
                values=values,
                vmin=vmin,
                vmax=vmax,
                cbar_label=cbar_label,
            )

        cbar = fig.colorbar(last_im, ax=hmap_axes, fraction=0.03, pad=0.02)
        cbar.set_label(cbar_label)

        fig.suptitle(metric_title)
        fig.tight_layout()
        plt.close(fig)

    return fig


def _draw_heatmap_panel(
    ax,
    values: np.ndarray,
    bg_image: np.ndarray,
    vmin: float,
    vmax: float,
    num_x_bins: int,
    num_y_bins: int,
    box_size: int,
    step: int,
    panel_title: str,
    cmap,
    pixel_size: float,
    pixel_unit: str,
    show_bin_ids: bool = False,
):
    grid        = values.reshape(num_x_bins, num_y_bins)
    grid_masked = np.ma.masked_invalid(grid)

    ind       = box_size // 2
    half_step = step / 2.0
    col_start = ind - half_step
    col_end   = ind + (num_y_bins - 1) * step + half_step
    row_start = ind - half_step
    row_end   = ind + (num_x_bins - 1) * step + half_step
    extent    = [col_start, col_end, row_end, row_start]

    height_px, width_px = bg_image.shape

    # Drawing background
    ax.imshow(
        bg_image,
        cmap='gray',
        origin='upper',
        aspect='equal',
        vmin=np.percentile(bg_image, 1),
        vmax=np.percentile(bg_image, 99),
    )

    # Drawing overlay
    im = ax.imshow(
        grid_masked,
        cmap=cmap,
        origin='upper',
        aspect='equal',
        interpolation='nearest',
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        alpha=0.6,
    )

    # --- TEXT OVERLAY AND GRID LINES FOR REFERENCE CHART ---
    if show_bin_ids:
        # 1. Overlay Numbers
        for r in range(num_x_bins):
            for c in range(num_y_bins):
                bin_val = grid[r, c]
                if np.isfinite(bin_val):
                    # Data coordinates for text center
                    x_center = ind + (c * step)
                    y_center = ind + (r * step)
                    ax.text(
                        x_center, y_center, str(int(bin_val)),
                        color='white', fontsize=6, ha='center', va='center',
                        weight='bold'
                    )

        # 2. Draw Grid Lines
        line_style = {'color': 'white', 'linestyle': '-', 'linewidth': 0.5, 'alpha': 0.8}
        
        # Determine center points of bins
        x_centers = np.arange(num_y_bins) * step + ind
        y_centers = np.arange(num_x_bins) * step + ind
        
        # Draw vertical lines between columns
        if num_y_bins > 1:
            v_lines = x_centers[:-1] + step/2.0
            ax.vlines(v_lines, ymin=row_start, ymax=row_end, **line_style)
            
        # Draw horizontal lines between rows
        if num_x_bins > 1:
            h_lines = y_centers[:-1] + step/2.0
            ax.hlines(h_lines, xmin=col_start, xmax=col_end, **line_style)

    _add_scale_bar(ax, width_px, height_px, pixel_size, pixel_unit)

    ax.set_title(panel_title, fontsize=9)
    ax.set_axis_off()
    return im


def _draw_hist_panel(ax, values, vmin, vmax, cbar_label):
    valid   = values[np.isfinite(values)]
    valid_n = valid.size
    total_n = values.size
    ax.set_title(f'{valid_n}/{total_n} bins detected', fontsize=7, pad=3)
    if valid_n >= 5:
        n_bins = min(20, max(5, valid_n // 3))
        _, bin_edges, patches = ax.hist(valid, bins=n_bins, edgecolor='none')
        norm_range = vmax - vmin if vmax > vmin else 1.0
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        for patch, bc in zip(patches, bin_centers):
            patch.set_facecolor(plt.cm.inferno((bc - vmin) / norm_range))
    else:
        ax.text(0.5, 0.5, 'insufficient data', transform=ax.transAxes, ha='center', va='center', fontsize=7)
    ax.set_xlabel(cbar_label, fontsize=7)
    ax.set_ylabel('Count', fontsize=7)
    ax.tick_params(labelsize=6)
    for spine in ['top', 'right']: ax.spines[spine].set_visible(False)

def _add_scale_bar(ax, width_px, height_px, pixel_size, pixel_unit):
    if pixel_size <= 0: return
    target_physical = width_px * pixel_size / 5.0
    bar_physical = _nice_scale_length(target_physical)
    bar_px = bar_physical / pixel_size
    margin_x, margin_y = width_px * 0.04, height_px * 0.05
    x0, x1, y = margin_x, margin_x + bar_px, height_px - margin_y
    ax.plot([x0, x1], [y, y], color='white', linewidth=2, solid_capstyle='butt')
    ax.text((x0 + x1) / 2, y - height_px * 0.025, f'{bar_physical:g} {pixel_unit}',
            color='white', ha='center', va='bottom', fontsize=6)

def _nice_scale_length(target):
    if target <= 0: return 1.0
    magnitude = 10 ** np.floor(np.log10(target))
    for factor in [5, 2, 1]:
        if factor * magnitude <= target: return factor * magnitude
    return magnitude
