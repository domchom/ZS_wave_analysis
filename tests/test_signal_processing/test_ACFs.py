"""
Tests for calc_indv_ACF_workflow (correlation_functions.py).

Uses UNSMOOTHED bin_values (straight from pixel binning, before Savitzky-Golay)
and img_props that embed those same unsmoothed bin_values.
Known results were generated from the same unsmoothed data.
To regenerate: run tests/regenerate_assets.py
"""
import pickle
import json
import numpy as np
from waveanalysis.signal_processing.correlation_functions import calc_indv_ACF_workflow

GROUPS = ['1_Group1', '1_Group2']

KNOWN_ACF_FILES = [
    f'tests/assets/standard/dicts_lists/{g}_indv_acfs.pkl' for g in GROUPS
]
BIN_VALUE_FILES = [
    f'tests/assets/standard/numpy_arrays/{g}_bin_values_unsmoothed.npy' for g in GROUPS
]
IMG_PROPS_FILES = [
    f'tests/assets/standard/dicts_lists/{g}_img_props_unsmoothed.json' for g in GROUPS
]

# ── calc_indv_ACF_workflow ────────────────────────────────────────────────────

def test_ACF_calc():
    for bin_values_file, acf_file, img_props_file in zip(BIN_VALUE_FILES, KNOWN_ACF_FILES, IMG_PROPS_FILES):
        bin_values = np.load(bin_values_file)
        with open(acf_file, 'rb') as f:
            known_acfs = pickle.load(f)
        with open(img_props_file, 'r') as f:
            img_props = json.load(f)

        exp_acfs = calc_indv_ACF_workflow(bin_values, img_props)

        np.testing.assert_allclose(known_acfs, exp_acfs, equal_nan=True, atol=1e-1)
