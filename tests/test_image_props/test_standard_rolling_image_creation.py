"""
Tests for tiff_to_np_array_multi_frame (image_to_np_arrays.py).

Verifies that loading a multi-frame TIFF produces the expected numpy array.
The same raw_image.npy files are used by both standard and rolling modes
since they share the same TIFF loading function.
To regenerate: run tests/regenerate_assets.py
"""
import numpy as np
from waveanalysis.image_props.image_to_np_arrays import tiff_to_np_array_multi_frame

TIFF_FILES = [
    'tests/assets/standard/1_Group1.tif',
    'tests/assets/standard/1_Group2.tif',
]
KNOWN_ARRAY_FILES = [
    'tests/assets/standard/numpy_arrays/1_Group1_raw_image.npy',
    'tests/assets/standard/numpy_arrays/1_Group2_raw_image.npy',
]

# ── tiff_to_np_array_multi_frame ─────────────────────────────────────────────

def test_standard_rolling_image_creation():
    for tiff_file, known_file in zip(TIFF_FILES, KNOWN_ARRAY_FILES):
        known_array = np.load(known_file)
        exp_array = tiff_to_np_array_multi_frame(tiff_file)
        assert np.array_equal(known_array, exp_array)
