"""
Tests for get_single_frame_properties (image_properties.py).

Verifies that extracting metadata from a single-frame (kymograph) TIFF returns
the expected dict (frame_interval, num_channels, num_frames, pixel_size, pixel_unit).
These basic_image_props files contain only those 5 keys — not the full
analysis img_props dict (which adds line_width, num_bins, etc. at runtime).
To regenerate: run tests/regenerate_assets.py
"""
import json
from waveanalysis.image_props.image_properties import get_single_frame_properties

TIFF_FILES = [
    'tests/assets/kymo/1_Group1.tif',
    'tests/assets/kymo/1_Group2.tif',
]
KNOWN_PROPS_FILES = [
    'tests/assets/kymo/dicts_lists/1_Group1_basic_image_props.json',
    'tests/assets/kymo/dicts_lists/1_Group2_basic_image_props.json',
]

# ── get_single_frame_properties ──────────────────────────────────────────────

def test_kymo_image_properties():
    for tiff_file, known_file in zip(TIFF_FILES, KNOWN_PROPS_FILES):
        with open(known_file, 'r') as f:
            known_props = json.load(f)
        exp_props = get_single_frame_properties(tiff_file)
        assert known_props == exp_props
