"""
Tests for calc_indv_shift_workflow (correlation_functions.py).

Takes pre-computed CCF arrays and periods as inputs (reuses *_indv_ccfs.pkl
and *_periods.pkl). small_shifts_correction=True and ccf_peak_thresh=0.1 are
hardcoded to match the settings used when the known shifts were generated.
To regenerate: run tests/regenerate_assets.py
"""
import pickle
import json
import numpy as np
from waveanalysis.signal_processing.correlation_functions import calc_indv_shift_workflow

GROUPS = ['1_Group1', '1_Group2']

KNOWN_SHIFT_FILES = [
    f'tests/assets/standard/dicts_lists/{g}_shifts.pkl' for g in GROUPS
]
PERIOD_FILES = [
    f'tests/assets/standard/dicts_lists/{g}_periods.pkl' for g in GROUPS
]
CCF_FILES = [
    f'tests/assets/standard/dicts_lists/{g}_indv_ccfs.pkl' for g in GROUPS
]
IMG_PROPS_FILES = [
    f'tests/assets/standard/dicts_lists/{g}_img_props_unsmoothed.json' for g in GROUPS
]

# ── calc_indv_shift_workflow ──────────────────────────────────────────────────

def test_shift_calc():
    for period_file, ccf_file, shift_file, img_props_file in zip(PERIOD_FILES, CCF_FILES, KNOWN_SHIFT_FILES, IMG_PROPS_FILES):
        with open(period_file, 'rb') as f:
            periods = pickle.load(f)
        with open(ccf_file, 'rb') as f:
            ccfs = pickle.load(f)
        with open(img_props_file, 'r') as f:
            img_props = json.load(f)
        with open(shift_file, 'rb') as f:
            known_shifts = pickle.load(f)

        exp_shifts = calc_indv_shift_workflow(
            ccfs, periods, img_props,
            small_shifts_correction=True,
            ccf_peak_thresh=0.1,
        )

        np.testing.assert_allclose(known_shifts, exp_shifts, equal_nan=True, atol=1e-5)
