"""
End-to-end test for the rolling (subframe, box-based) workflow.

Runs rolling_workflow on two 2-channel TIFFs and compares the summary
DataFrame against a known-good CSV. The workflow processes files in order
and returns early (after 1_Group2) when test=True, so only that file's
results are in the known CSV.

To regenerate known_1_Group2_summary.csv: run rolling_workflow with test=True
and save the returned DataFrame.
"""
import pytest
import pandas as pd
from pathlib import Path
from waveanalysis.data_workflows import rolling_workflow

@pytest.fixture
def default_log_params():
    return {
        'Box Size(px)': 20,
        'Box Shift(px)': 20,
        'Base Directory': 'tests/assets/rolling',
        'ACF Peak Prominence': 0.1,
        'Files Processed': [],
        'Files Not Processed': [],
        'Plotting errors': [],
        'Submovies Used' : [],
        'Errors': [],
        'Frame Interval': [],
        'Pixel Size': [],
        'Small Shifts Correction': True,
        'CCF Peak Prominence': 0.1,
        "Smoothing": True,
        "smoothing_params": {
            "Ch1": {"window": 11, "poly_order": 3},
            "Ch2": {"window": 11, "poly_order": 3},
            "Ch3": {"window": 11, "poly_order": 3},
            "Ch4": {"window": 11, "poly_order": 3},
            "CCF": {"window": 11, "poly_order": 3},
        },
        "Dark Plots": False,
        }


def test_rolling_workflow(default_log_params):
    known_results = pd.read_csv('tests/assets/rolling/known_1_Group2_summary.csv')
    exp_results = rolling_workflow(
        folder_path=str(Path('tests/assets/rolling/')),
        log_params=default_log_params,
        box_size=default_log_params['Box Size(px)'],
        bin_shift=default_log_params['Box Shift(px)'],
        subframe_size=50,
        subframe_roll=5,
        acf_peak_thresh=default_log_params['ACF Peak Prominence'],
        ccf_peak_thresh=default_log_params['CCF Peak Prominence'],
        small_shifts_correction=default_log_params['Small Shifts Correction'],
        smoothing_params=default_log_params['smoothing_params'],
        smoothing=default_log_params['Smoothing'],
        test=True,
        dark_plots=default_log_params['Dark Plots'],
    )
        
    pd.testing.assert_frame_equal(
        known_results.reset_index(drop=True),
        exp_results.reset_index(drop=True),
        atol=1e-0,
    )
