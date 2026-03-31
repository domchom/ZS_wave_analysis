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
        "Ch1 Window": 11,
        "Ch1 Poly Order": 3,
        "Ch2 Window": 11,
        "Ch2 Poly Order": 3,
        "Ch3 Window": 11,
        "Ch3 Poly Order": 3,
        "Ch4 Window": 11,
        "Ch4 Poly Order": 3,
        "CCF Window": 11,
        "CCF Poly Order": 3,
        "Smoothing": True,
        "Dark Plots": False,
        }


def test_rolling_workflow(default_log_params):
    # load csv
    known_results = pd.read_csv('tests/assets/rolling/known_1_Group2_summary.csv')
    assert isinstance(known_results, pd.DataFrame)
    exp_results = rolling_workflow(
        folder_path=str(Path('tests/assets/rolling/')),
        log_params=default_log_params,
        box_size=default_log_params['Box Size(px)'],
        box_shift=default_log_params['Box Shift(px)'],
        roll_size=50,
        roll_by=5,       
        acf_peak_thresh=default_log_params['ACF Peak Prominence'],
        ccf_peak_thresh=default_log_params['CCF Peak Prominence'],
        small_shifts_correction=default_log_params['Small Shifts Correction'],
        Ch1_window = default_log_params['Ch1 Window'],
        Ch1_poly_order = default_log_params['Ch1 Poly Order'],
        Ch2_window = default_log_params['Ch2 Window'],
        Ch2_poly_order = default_log_params['Ch2 Poly Order'],
        Ch3_window = default_log_params['Ch3 Window'],
        Ch3_poly_order = default_log_params['Ch3 Poly Order'],
        Ch4_window = default_log_params['Ch4 Window'],
        Ch4_poly_order = default_log_params['Ch4 Poly Order'],
        CCF_window = default_log_params['CCF Window'],
        CCF_poly_order = default_log_params['CCF Poly Order'],
        smoothing = default_log_params['Smoothing'],
        test=True,
        dark_plots=default_log_params['Dark Plots'],
    )
        
    pd.testing.assert_frame_equal(
        known_results.reset_index(drop=True),
        exp_results.reset_index(drop=True),
        atol=1e-0,
    )
