"""
Shared utilities used by both combined_workflow and rolling_workflow.
Not part of the public API — import from data_workflows directly.
"""
import os
import timeit
import datetime
import numpy as np
from typing import Any

import waveanalysis.housekeeping.housekeeping_functions as hf
from waveanalysis.image_props.image_bin_calc import smooth_signal
from waveanalysis.image_props.image_properties import get_multi_frame_properties, get_single_frame_properties


def _setup_workflow(folder_path: str, test: bool):
    """
    Common workflow preamble: build file list, start timer, create output directory.

    Returns (file_names, start, now, main_save_path).
    """
    file_names = [
        f for f in os.listdir(folder_path)
        if f.endswith('.tif') and not f.startswith('.')
    ]
    start = timeit.default_timer()
    now = datetime.datetime.now()
    main_save_path = os.path.join(folder_path, f"0_signalProcessing-{now.strftime('%Y%m%d%H%M')}")
    os.makedirs(main_save_path, exist_ok=True) if not test else None
    return file_names, start, now, main_save_path


def _load_image_props(image_path: str, log_params: dict[str, Any], file_name: str,
                      bin_shift: int, box_size: int, acf_peak_thresh: float,
                      peak_prominence_fraction: float = 0.1,
                      analysis_type: str = 'standard',
                      extra_props: dict = None) -> dict:
    """
    Load image properties, check and log frame interval and pixel size,
    and populate the standard analysis keys (step, box_size, peak_thresh).

    Uses get_multi_frame_properties for 'standard'/'rolling', or
    get_single_frame_properties for 'kymograph'.

    extra_props are merged into img_props after the standard keys (e.g. line_width,
    analysis_type for combined_workflow, or num_submovies for rolling_workflow).

    Returns the populated img_props dict, or None if the image has fewer than 11 frames.
    """
    if analysis_type == 'kymograph':
        img_props = get_single_frame_properties(image_path=image_path)
    else:
        img_props = get_multi_frame_properties(image_path=image_path)

    frame_interval = hf.check_frame_interval(
        frame_interval=img_props['frame_interval'],
        log_params=log_params,
        file_name=file_name,
    )
    img_props['frame_interval'] = frame_interval

    img_props['step'] = bin_shift
    img_props['box_size'] = box_size
    img_props['peak_thresh'] = acf_peak_thresh
    img_props['peak_prominence_fraction'] = peak_prominence_fraction

    if extra_props:
        img_props.update(extra_props)

    log_params['Pixel Size'].append(f"{file_name}: {img_props['pixel_size']} {img_props['pixel_unit']}s")
    log_params['Frame Interval'].append(f"{file_name}: {img_props['frame_interval']} seconds")

    if img_props['num_frames'] < 11:
        print(
            f"****** ERROR ******",
            f"\n{file_name} has less than 11 frames. Movies must have more than 10 frames",
            "\n****** ERROR ******",
        )
        return None

    return img_props


def _smooth_bin_values_inplace(
    bin_values: np.ndarray,
    num_bins: int,
    num_channels: int,
    smoothing_params: dict,
) -> None:
    """
    Apply per-channel Savitzky-Golay smoothing in-place on 3-D bin_values
    (frames × channels × bins). Channels with no smoothing entry are left unchanged.
    """
    for channel in range(num_channels):
        ch_params = (smoothing_params or {}).get(f"Ch{channel + 1}")
        if ch_params is not None:
            for bin_idx in range(num_bins):
                bin_values[:, channel, bin_idx] = smooth_signal(
                    signal=bin_values[:, channel, bin_idx],
                    window=ch_params["window"],
                    poly_order=ch_params["poly_order"],
                )
