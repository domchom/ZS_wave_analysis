import pytest
import pandas as pd
from pathlib import Path
from waveanalysis.data_workflows.combined_workflow import combined_workflow

@pytest.fixture
def default_log_params():
    return {
        'Line Size(px)': 5,
        'Line Shift(px)': 5,
        'Base Directory': 'tests/assets/kymo',
        'ACF Peak Prominence': 0.1,
        'Group Names': ['Group1', 'Group2'],
        'Plot Summary ACFs': False,
        'Plot Summary CCFs': False,
        'Plot Summary Peaks': False,
        'Plot Individual ACFs': False,
        'Plot Individual CCFs': False,
        'Plot Individual Peaks': False,
        'Calc Wave Speeds': False,
        'Plot Wave Speeds': False, 
        'Files Processed': [],
        'Files Not Processed': [],
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

def test_kymo_workflow(default_log_params):
    # load csv
    known_results = pd.read_csv('tests/assets/kymo/known_kymograph_summary.csv')
    assert isinstance(known_results, pd.DataFrame)
    exp_results = combined_workflow(
        folder_path=str(Path('tests/assets/kymo')),
        group_names= default_log_params['Group Names'],
        log_params=default_log_params,
        analysis_type='kymograph',
        box_size=None, #type: ignore
        bin_shift=default_log_params['Line Shift(px)'],
        line_width=default_log_params['Line Size(px)'],         
        acf_peak_thresh=default_log_params['ACF Peak Prominence'],
        ccf_peak_thresh=default_log_params['CCF Peak Prominence'],
        small_shifts_correction=default_log_params['Small Shifts Correction'],
        plot_summary_ACFs=default_log_params['Plot Summary ACFs'],
        plot_summary_CCFs=default_log_params['Plot Summary CCFs'],
        plot_summary_peaks=default_log_params['Plot Summary Peaks'],
        plot_indv_ACFs=default_log_params['Plot Individual ACFs'],
        plot_indv_CCFs=default_log_params['Plot Individual CCFs'],
        plot_indv_peaks=default_log_params['Plot Individual Peaks'],
        calc_wave_speeds=default_log_params['Calc Wave Speeds'],
        plot_wave_speeds=default_log_params['Plot Wave Speeds'],
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
        atol=1e-1,
    )
