"""
Tests for plot_mean_ACF_workflow (mean_plot_creation.py).

Verifies that the function runs without error and returns the expected number
of figures (one per channel). Comparing matplotlib Figure objects across
environments is not reliable, so we only check the count.
To regenerate inputs: run tests/regenerate_assets.py

Inputs:
  - img_parameters: per-bin analysis results dict (Period, Peak Amp, etc.)
  - img_props: image metadata + analysis config (unsmoothed variant)
  - acf_array: pre-computed ACF curves (channels × bins × frames)
"""
import pickle
import json
from waveanalysis.plotting.mean_plot_creation import plot_mean_ACF_workflow

GROUPS = ['1_Group1', '1_Group2']

IMG_PARAMETERS_FILES = [
    f'tests/assets/standard/dicts_lists/{g}_img_parameters.pkl' for g in GROUPS
]
IMG_PROPS_FILES = [
    f'tests/assets/standard/dicts_lists/{g}_img_props_unsmoothed.json' for g in GROUPS
]
ACF_ARRAY_FILES = [
    f'tests/assets/standard/dicts_lists/{g}_indv_acfs.pkl' for g in GROUPS
]

# Both test images have 2 channels → expect 2 figures each
EXPECTED_PLOT_COUNT = 2

def test_mean_ACF_plot():
    for img_params_file, img_props_file, acf_file in zip(IMG_PARAMETERS_FILES, IMG_PROPS_FILES, ACF_ARRAY_FILES):
        with open(img_params_file, 'rb') as f:
            img_params = pickle.load(f)
        with open(img_props_file, 'r') as f:
            img_props = json.load(f)
        with open(acf_file, 'rb') as f:
            acf_array = pickle.load(f)

        result = plot_mean_ACF_workflow(img_params, img_props, acf_array)

        assert len(result) == EXPECTED_PLOT_COUNT
