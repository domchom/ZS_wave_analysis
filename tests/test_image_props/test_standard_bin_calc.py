"""
Tests for create_multi_frame_bin_array (image_bin_calc.py).

Loads a raw image array (frames × channels × height × width) and the img_props
dict, then verifies that binning produces the expected unsmoothed bin_values
(frames × channels × bins). Known results are the unsmoothed bin_values .npy files.
To regenerate: run tests/regenerate_assets.py
"""
import json
import numpy as np
from waveanalysis.image_props.image_bin_calc import create_multi_frame_bin_array

GROUPS = ['1_Group1', '1_Group2']

RAW_IMAGE_FILES = [
    f'tests/assets/standard/numpy_arrays/{g}_raw_image.npy' for g in GROUPS
]
KNOWN_BIN_VALUE_FILES = [
    f'tests/assets/standard/numpy_arrays/{g}_bin_values_unsmoothed.npy' for g in GROUPS
]
IMG_PROPS_FILES = [
    f'tests/assets/standard/dicts_lists/{g}_img_props_unsmoothed.json' for g in GROUPS
]

# ── create_multi_frame_bin_array ─────────────────────────────────────────────

def test_standard_bin_calc():
    for raw_image_file, known_file, img_props_file in zip(RAW_IMAGE_FILES, KNOWN_BIN_VALUE_FILES, IMG_PROPS_FILES):
        raw_image = np.load(raw_image_file)
        known_bin_values = np.load(known_file)
        with open(img_props_file, 'r') as f:
            img_props = json.load(f)

        exp_bin_values, _, _, _ = create_multi_frame_bin_array(raw_image, img_props)

        assert np.array_equal(known_bin_values, exp_bin_values)
