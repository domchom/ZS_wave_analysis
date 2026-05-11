import numpy as np
import matplotlib.pyplot as plt

# Per-channel metrics: key in img_metrics -> colorbar label
_CHANNEL_METRICS = {
    'Period':       'Period (s)',
    'Peak Amp':     'Amplitude (AU)',
    'Peak Rel Amp': 'Rel. Amplitude',
    'Peak Width':   'Width (s)',
    'Peak Max':     'Peak Max (AU)',
    'Peak Min':     'Peak Min (AU)',
    'Peak Offset':  'Offset (s)',
    'Peak Area':    'Area (AU)',
}

# Per channel-combo metrics: key in img_metrics -> colorbar label
_COMBO_METRICS = {
    'Shift':         'Shift (s)',
    '% Phase Shift': 'Phase Shift (%)',
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
    for metric_name, cbar_label in _CHANNEL_METRICS.items():
        if metric_name not in img_metrics:
            continue
        data = img_metrics[metric_name]
        valid_all = data[np.isfinite(data)]
        vmin = float(np.min(valid_all)) if valid_all.size > 0 else 0.0
        vmax = float(np.max(valid_all)) if valid_all.size > 0 else 1.0

        panels = [
            (data[ch], bg_images[ch], f'Ch {ch + 1}')
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
    for metric_name, cbar_label in _COMBO_METRICS.items():
        if metric_name not in img_metrics:
            continue
        for combo_idx, (ch1, ch2) in enumerate(channel_combos):
            combo_label = f'Ch{ch1 + 1}-Ch{ch2 + 1}'
            values = img_metrics[metric_name][combo_idx]
            bg = (bg_images[ch1].astype(float) + bg_images[ch2].astype(float)) / 2
            valid = values[np.isfinite(values)]
            vmin = float(np.min(valid)) if valid.size > 0 else 0.0
            vmax = float(np.max(valid)) if valid.size > 0 else 1.0

            panels = [(values, bg, combo_label)]
            heatmap_figs[f'{combo_label} {metric_name} Heatmap'] = _return_metric_figure(
                panels=panels,
                vmin=vmin,
                vmax=vmax,
                num_x_bins=num_x_bins,
                num_y_bins=num_y_bins,
                box_size=box_size,
                step=step,
                metric_title=f'{combo_label} {metric_name}',
                cbar_label=cbar_label,
                pixel_size=pixel_size,
                pixel_unit=pixel_unit,
                dark_plots=dark_plots,
                show_bin_ids=False
            )

    return heatmap_figs


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