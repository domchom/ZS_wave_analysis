"""
Tests for calc_indv_period_workflow (correlation_functions.py).

Takes pre-computed ACF arrays as input (reuses *_indv_acfs.pkl from test_ACFs)
and returns the period for each channel/bin.
To regenerate: run tests/regenerate_assets.py
"""
import pickle
import json
import numpy as np
from waveanalysis.signal_processing.correlation_functions import calc_indv_period_workflow

GROUPS = ['1_Group1', '1_Group2']

KNOWN_PERIOD_FILES = [
    f'tests/assets/standard/dicts_lists/{g}_periods.pkl' for g in GROUPS
]
ACF_FILES = [
    f'tests/assets/standard/dicts_lists/{g}_indv_acfs.pkl' for g in GROUPS
]
IMG_PROPS_FILES = [
    f'tests/assets/standard/dicts_lists/{g}_img_props_unsmoothed.json' for g in GROUPS
]

def test_period_calc():
    for acf_file, period_file, img_props_file in zip(ACF_FILES, KNOWN_PERIOD_FILES, IMG_PROPS_FILES):
        with open(period_file, 'rb') as f:
            known_periods = pickle.load(f)
        with open(acf_file, 'rb') as f:
            acf_array = pickle.load(f)
        with open(img_props_file, 'r') as f:
            img_props = json.load(f)

        exp_periods = calc_indv_period_workflow(acf_array, img_props)

        assert np.array_equal(known_periods, exp_periods)
