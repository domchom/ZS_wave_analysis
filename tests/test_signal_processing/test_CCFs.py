"""
Tests for calc_indv_CCF_workflow (correlation_functions.py).

Uses SMOOTHED bin_values (after Savitzky-Golay, window=11 poly=3) and img_props
that embed those smoothed bin_values. Known results were generated from the same data.
To regenerate: run tests/regenerate_assets.py
"""
import pickle
import json
import numpy as np
from waveanalysis.signal_processing.correlation_functions import calc_indv_CCF_workflow

GROUPS = ['1_Group1', '1_Group2']

KNOWN_CCF_FILES = [
    f'tests/assets/standard/dicts_lists/{g}_indv_ccfs.pkl' for g in GROUPS
]
BIN_VALUE_FILES = [
    f'tests/assets/standard/numpy_arrays/{g}_bin_values_smoothed.npy' for g in GROUPS
]
IMG_PROPS_FILES = [
    f'tests/assets/standard/dicts_lists/{g}_img_props_smoothed.json' for g in GROUPS
]

# ── calc_indv_CCF_workflow ────────────────────────────────────────────────────

def test_CCF_calc():
    for bin_values_file, ccf_file, img_props_file in zip(BIN_VALUE_FILES, KNOWN_CCF_FILES, IMG_PROPS_FILES):
        bin_values = np.load(bin_values_file)
        with open(ccf_file, 'rb') as f:
            known_ccfs = pickle.load(f)
        with open(img_props_file, 'r') as f:
            img_props = json.load(f)

        exp_ccfs = calc_indv_CCF_workflow(bin_values, img_props, ccf_smoothing={"window": 11, "poly_order": 3})

        np.testing.assert_allclose(known_ccfs, exp_ccfs, equal_nan=True, atol=1e-1)
