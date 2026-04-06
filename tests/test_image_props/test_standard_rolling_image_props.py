"""
Tests for get_multi_frame_properties (image_properties.py).

Verifies that extracting metadata from a multi-frame TIFF returns the expected
dict (frame_interval, num_channels, num_frames, pixel_size, pixel_unit).
These basic_image_props files contain only those 5 keys — not the full
analysis img_props dict (which adds box_size, num_bins, etc. at runtime).
To regenerate: run tests/regenerate_assets.py
"""
import json
import numpy as np
from waveanalysis.image_props.image_properties import get_multi_frame_properties

TIFF_FILES = [
    'tests/assets/standard/1_Group1.tif',
    'tests/assets/standard/1_Group2.tif',
]
KNOWN_PROPS_FILES = [
    'tests/assets/standard/dicts_lists/1_Group1_basic_image_props.json',
    'tests/assets/standard/dicts_lists/1_Group2_basic_image_props.json',
]

def test_standard_rolling_image_properties():
    for tiff_file, known_file in zip(TIFF_FILES, KNOWN_PROPS_FILES):
        with open(known_file, 'r') as f:
            known_props = json.load(f)
        exp_props = get_multi_frame_properties(tiff_file)
        assert np.array_equal(known_props, exp_props)
