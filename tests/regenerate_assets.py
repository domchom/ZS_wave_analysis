"""
Regenerate known test asset files.

Run from the project root:
    python tests/regenerate_assets.py

Regenerates ALL test assets from the current TIFFs using fixed parameters.
Run this whenever an algorithm changes and the known outputs need updating.

======================================================================
FILES REGENERATED
======================================================================
  standard and kymo:
    *_raw_image.npy             raw numpy array from TIFF
    *_basic_image_props.json    frame_interval, num_channels, etc.

  standard only:
    *_bin_values_unsmoothed.npy bins before Savitzky-Golay (box_size=20, step=20)
    *_bin_values_smoothed.npy   bins after Savitzky-Golay (window=11, poly=3)
    *_img_props_unsmoothed.json img_props dict with unsmoothed bin_values embedded
    *_img_props_smoothed.json   img_props dict with smoothed bin_values embedded
    *_indv_acfs.pkl             output of calc_indv_ACF_workflow (unsmoothed input)
    *_indv_ccfs.pkl             output of calc_indv_CCF_workflow (smoothed input)
    *_peak_props.pkl            output of calc_indv_peak_props_workflow (smoothed input)
    *_periods.pkl               output of calc_indv_period_workflow
    *_shifts.pkl                output of calc_indv_shift_workflow
    *_img_parameters.pkl        per-image analysis results dict from combined_workflow
    known_standard_summary.csv  end-to-end workflow output

  kymo only:
    *_bin_values.npy            bins from kymograph (line_width=5, bin_shift=5)
    *_img_props.json            img_props dict
    known_kymograph_summary.csv end-to-end workflow output

  rolling only:
    known_1_Group2_summary.csv  rolling workflow output (first file processed)

======================================================================
PARAMETERS USED
======================================================================
  Standard:  box_size=20, bin_shift=20, smoothing window=11, poly=3
  Kymograph: line_width=5, bin_shift=5, smoothing window=11, poly=3
  Rolling:   box_size=20, box_shift=20, roll_size=50, roll_by=5
  All:       acf_peak_thresh=0.1, ccf_peak_thresh=0.1,
             small_shifts_correction=True
"""
import os
import json
import pickle
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter

# Ensure we can import waveanalysis from project root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from waveanalysis.image_props.image_to_np_arrays import (
    tiff_to_np_array_multi_frame,
    tiff_to_np_array_single_frame,
)
from waveanalysis.image_props.image_properties import (
    get_multi_frame_properties,
    get_single_frame_properties,
)
from waveanalysis.image_props.image_bin_calc import (
    create_multi_frame_bin_array,
    create_kymo_bin_array,
)
from waveanalysis.signal_processing.correlation_functions import (
    calc_indv_ACF_workflow,
    calc_indv_CCF_workflow,
    calc_indv_period_workflow,
    calc_indv_shift_workflow,
)
from waveanalysis.signal_processing.peak_properties import calc_indv_peak_props_workflow
from waveanalysis.housekeeping.housekeeping_functions import get_channel_combos
from waveanalysis.data_workflows.combined_workflow import combined_workflow
from waveanalysis.data_workflows.rolling_workflow import rolling_workflow

ASSETS = Path('tests/assets')
GROUPS = ['1_Group1', '1_Group2']

SMOOTHING = {'window': 11, 'poly_order': 3}
SMOOTHING_PARAMS = {ch: SMOOTHING for ch in ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'CCF']}

# -----------------------------------------------------------------------
# Shared log_params templates (mirrors what the GUI populates)
# -----------------------------------------------------------------------
def _base_log_params(base_dir, group_names):
    return {
        'Base Directory': str(base_dir),
        'Group Names': group_names,
        'ACF Peak Prominence': 0.1,
        'CCF Peak Prominence': 0.1,
        'Small Shifts Correction': True,
        'Smoothing': True,
        'smoothing_params': SMOOTHING_PARAMS,
        'plot_flags': {k: False for k in [
            'plot_summary_ACFs', 'plot_summary_CCFs', 'plot_summary_peaks',
            'plot_indv_ACFs', 'plot_indv_CCFs', 'plot_indv_peaks', 'plot_heatmaps', 'dark_plots',
        ]},
        'Files Processed': [],
        'Files Not Processed': [],
        'Errors': [],
        'Frame Interval': [],
        'Pixel Size': [],
    }

# -----------------------------------------------------------------------
# Standard assets
# -----------------------------------------------------------------------
def regenerate_standard():
    src = ASSETS / 'standard'
    npy_dir = src / 'numpy_arrays'
    pkl_dir = src / 'dicts_lists'

    for group in GROUPS:
        tiff = str(src / f'{group}.tif')
        print(f'  {group}...')

        # Raw image array and basic image properties
        raw_image = tiff_to_np_array_multi_frame(tiff)
        np.save(npy_dir / f'{group}_raw_image.npy', raw_image)

        basic_props = get_multi_frame_properties(tiff)
        with open(pkl_dir / f'{group}_basic_image_props.json', 'w') as f:
            json.dump(basic_props, f)

        # img_props dict (without bin_values — added by create_multi_frame_bin_array)
        img_props = {**basic_props,
                     'analysis_type': 'standard',
                     'box_size': 20,
                     'step': 20,
                     'line_width': None,
                     'peak_thresh': 0.1}

        # Unsmoothed bin_values
        bin_values_unsmoothed, num_bins, _, _ = create_multi_frame_bin_array(raw_image, img_props)
        np.save(npy_dir / f'{group}_bin_values_unsmoothed.npy', bin_values_unsmoothed)

        num_channels = basic_props['num_channels']
        channel_combos = get_channel_combos(num_channels)
        num_combos = len(channel_combos)

        img_props_unsmoothed = {**img_props,
                                 'num_bins': num_bins,
                                 'num_combos': num_combos,
                                 'channel_combos': channel_combos,
                                 'bin_values': bin_values_unsmoothed.tolist()}
        with open(pkl_dir / f'{group}_img_props_unsmoothed.json', 'w') as f:
            json.dump(img_props_unsmoothed, f)

        # Smoothed bin_values (in-place Savitzky-Golay per channel per bin)
        bin_values_smoothed = bin_values_unsmoothed.astype(float)
        num_channels = img_props_unsmoothed['num_channels']
        for ch in range(num_channels):
            for b in range(num_bins):
                signal = bin_values_smoothed[:, ch, b]
                bin_values_smoothed[:, ch, b] = savgol_filter(
                    signal, SMOOTHING['window'], SMOOTHING['poly_order']
                )
        np.save(npy_dir / f'{group}_bin_values_smoothed.npy', bin_values_smoothed.astype(np.uint16))

        img_props_smoothed = {**img_props_unsmoothed,
                               'bin_values': str(bin_values_smoothed.astype(np.uint16))}
        with open(pkl_dir / f'{group}_img_props_smoothed.json', 'w') as f:
            json.dump(img_props_smoothed, f)

        # Signal processing outputs (use unsmoothed for ACFs, smoothed for CCF/peaks)
        indv_acfs = calc_indv_ACF_workflow(bin_values_unsmoothed, img_props_unsmoothed)
        with open(pkl_dir / f'{group}_indv_acfs.pkl', 'wb') as f:
            pickle.dump(indv_acfs, f)

        indv_ccfs = calc_indv_CCF_workflow(
            bin_values_smoothed.astype(np.uint16), img_props_smoothed,
            ccf_smoothing=SMOOTHING
        )
        with open(pkl_dir / f'{group}_indv_ccfs.pkl', 'wb') as f:
            pickle.dump(indv_ccfs, f)

        _, _, _, _, peak_props, _ = calc_indv_peak_props_workflow(
            bin_values_smoothed.astype(np.uint16), img_props_smoothed
        )
        with open(pkl_dir / f'{group}_peak_props.pkl', 'wb') as f:
            pickle.dump(peak_props, f)

        # Periods and shifts — derived from the ACFs/CCFs above, same img_props
        periods = calc_indv_period_workflow(indv_acfs, img_props_unsmoothed)
        with open(pkl_dir / f'{group}_periods.pkl', 'wb') as f:
            pickle.dump(periods, f)

        shifts = calc_indv_shift_workflow(
            indv_ccfs, periods, img_props_unsmoothed,
            small_shifts_correction=True,
            ccf_peak_thresh=0.1,
        )
        with open(pkl_dir / f'{group}_shifts.pkl', 'wb') as f:
            pickle.dump(shifts, f)

    # End-to-end workflow summary CSV + img_parameters pickle
    # img_parameters is saved via the (currently commented-out) pickle block in
    # combined_workflow.py — uncomment those lines, run once, then re-comment.
    print('  Regenerating known_standard_summary.csv...')
    log_params = {**_base_log_params(src, ['Group1', 'Group2']),
                  'Box Size(px)': 20, 'Box Shift(px)': 20,
                  'Calc Wave Speeds': False, 'Plot Wave Speeds': False}
    summary = combined_workflow(
        folder_path=str(src),
        group_names=['Group1', 'Group2'],
        log_params=log_params,
        analysis_type='standard',
        box_size=20, bin_shift=20, line_width=None,
        acf_peak_thresh=0.1, ccf_peak_thresh=0.1,
        small_shifts_correction=True,
        plot_flags=log_params['plot_flags'],
        calc_wave_speeds=None, plot_wave_speeds=None,
        smoothing_params=SMOOTHING_PARAMS, smoothing=True,
        test=True,
    )
    summary.to_csv(src / 'known_standard_summary.csv', index=False)
    print('  Saved known_standard_summary.csv')


# -----------------------------------------------------------------------
# Kymograph assets
# -----------------------------------------------------------------------
def regenerate_kymo():
    src = ASSETS / 'kymo'
    npy_dir = src / 'numpy_arrays'
    pkl_dir = src / 'dicts_lists'

    for group in GROUPS:
        tiff = str(src / f'{group}.tif')
        print(f'  {group}...')

        raw_image = tiff_to_np_array_single_frame(tiff)
        np.save(npy_dir / f'{group}_raw_image.npy', raw_image)

        basic_props = get_single_frame_properties(tiff)
        with open(pkl_dir / f'{group}_basic_image_props.json', 'w') as f:
            json.dump(basic_props, f)

        img_props = {**basic_props,
                     'analysis_type': 'kymograph',
                     'line_width': 5,
                     'step': 5,
                     'box_size': None,
                     'peak_thresh': 0.1}

        bin_values, num_bins = create_kymo_bin_array(raw_image, img_props)
        np.save(npy_dir / f'{group}_bin_values.npy', bin_values)

        num_channels = basic_props['num_channels']
        channel_combos = get_channel_combos(num_channels)
        img_props_full = {**img_props,
                          'num_bins': num_bins,
                          'num_combos': len(channel_combos),
                          'channel_combos': channel_combos}
        with open(pkl_dir / f'{group}_img_props.json', 'w') as f:
            json.dump(img_props_full, f)

    print('  Regenerating known_kymograph_summary.csv...')
    log_params = {**_base_log_params(src, ['Group1', 'Group2']),
                  'Line Size(px)': 5, 'Line Shift(px)': 5,
                  'Calc Wave Speeds': False, 'Plot Wave Speeds': False}
    summary = combined_workflow(
        folder_path=str(src),
        group_names=['Group1', 'Group2'],
        log_params=log_params,
        analysis_type='kymograph',
        box_size=None, bin_shift=5, line_width=5,
        acf_peak_thresh=0.1, ccf_peak_thresh=0.1,
        small_shifts_correction=True,
        plot_flags=log_params['plot_flags'],
        calc_wave_speeds=False, plot_wave_speeds=False,
        smoothing_params=SMOOTHING_PARAMS, smoothing=True,
        test=True,
    )
    summary.to_csv(src / 'known_kymograph_summary.csv', index=False)
    print('  Saved known_kymograph_summary.csv')


# -----------------------------------------------------------------------
# Rolling assets
# -----------------------------------------------------------------------
def regenerate_rolling():
    src = ASSETS / 'rolling'
    print('  Regenerating known_1_Group2_summary.csv...')
    log_params = {
        **_base_log_params(src, []),
        'Box Size(px)': 20, 'Box Shift(px)': 20,
        'Submovies Used': [], 'Plotting errors': [],
        'Dark Plots': False,
    }
    summary = rolling_workflow(
        folder_path=str(src),
        log_params=log_params,
        box_size=20, bin_shift=20,
        subframe_size=50, subframe_roll=5,
        acf_peak_thresh=0.1, ccf_peak_thresh=0.1,
        small_shifts_correction=True,
        smoothing_params=SMOOTHING_PARAMS, smoothing=True,
        dark_plots=False,
        test=True,
    )
    summary.to_csv(src / 'known_1_Group2_summary.csv', index=False)
    print('  Saved known_1_Group2_summary.csv')


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
if __name__ == '__main__':
    print('Regenerating standard assets...')
    regenerate_standard()
    print('Regenerating kymograph assets...')
    regenerate_kymo()
    print('Regenerating rolling assets...')
    regenerate_rolling()
    print('\nDone. Run pytest to verify nothing broke.')
