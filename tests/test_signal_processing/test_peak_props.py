"""
Tests for calc_indv_peak_props_workflow (peak_properties.py).

Uses SMOOTHED bin_values (same data as test_CCFs.py) and img_props that embed
those smoothed bin_values. Known results are a dict of dicts:
  { bin_index: { property_name: np.ndarray } }
To regenerate: run tests/regenerate_assets.py
"""
import pickle
import json
import numpy as np
from waveanalysis.signal_processing.peak_properties import calc_indv_peak_props_workflow

GROUPS = ['1_Group1', '1_Group2']

KNOWN_PEAK_PROP_FILES = [
    f'tests/assets/standard/dicts_lists/{g}_peak_props.pkl' for g in GROUPS
]
BIN_VALUE_FILES = [
    f'tests/assets/standard/numpy_arrays/{g}_bin_values_smoothed.npy' for g in GROUPS
]
IMG_PROPS_FILES = [
    f'tests/assets/standard/dicts_lists/{g}_img_props_smoothed.json' for g in GROUPS
]

# ── calc_indv_peak_props_workflow ─────────────────────────────────────────────

def test_peak_props_calc():
    for bin_values_file, peak_props_file, img_props_file in zip(BIN_VALUE_FILES, KNOWN_PEAK_PROP_FILES, IMG_PROPS_FILES):
        bin_values = np.load(bin_values_file)
        with open(peak_props_file, 'rb') as f:
            known_peak_props = pickle.load(f)
        with open(img_props_file, 'r') as f:
            img_props = json.load(f)

        _, _, _, _, exp_peak_props, _, _, _, _, _ = calc_indv_peak_props_workflow(bin_values, img_props)

        for key, value in known_peak_props.items():
            for prop_name, known_arr in value.items():
                np.testing.assert_allclose(
                    known_arr,
                    exp_peak_props[key][prop_name],
                    equal_nan=True,
                    atol=1.01,
                )
