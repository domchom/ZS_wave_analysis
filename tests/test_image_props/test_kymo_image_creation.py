"""
Tests for tiff_to_np_array_single_frame (image_to_np_arrays.py).

Verifies that loading a single-frame (kymograph) TIFF produces the expected
numpy array (channels × height × width).
To regenerate: run tests/regenerate_assets.py
"""
import numpy as np
from waveanalysis.image_props.image_to_np_arrays import tiff_to_np_array_single_frame

TIFF_FILES = [
    'tests/assets/kymo/1_Group1.tif',
    'tests/assets/kymo/1_Group2.tif',
]
KNOWN_ARRAY_FILES = [
    'tests/assets/kymo/numpy_arrays/1_Group1_raw_image.npy',
    'tests/assets/kymo/numpy_arrays/1_Group2_raw_image.npy',
]

def test_kymo_image_creation():
    for tiff_file, known_file in zip(TIFF_FILES, KNOWN_ARRAY_FILES):
        known_array = np.load(known_file)
        exp_array = tiff_to_np_array_single_frame(tiff_file)
        assert np.array_equal(known_array, exp_array)
