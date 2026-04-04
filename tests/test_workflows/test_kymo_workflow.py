"""
End-to-end test for the kymograph (single-frame line-scan) workflow.

Runs combined_workflow on two 2-channel kymograph TIFFs and compares the
summary DataFrame against a known-good CSV. Plotting is disabled because
test=True skips directory creation.

To regenerate known_kymograph_summary.csv: run combined_workflow with test=False
on the same TIFFs and copy the output CSV here.
"""
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
        'plot_flags': {
            'plot_summary_ACFs': False,
            'plot_summary_CCFs': False,
            'plot_summary_peaks': False,
            'plot_indv_ACFs': False,
            'plot_indv_CCFs': False,
            'plot_indv_peaks': False,
            'plot_heatmaps': False,
            'dark_plots': False,
        },
        'Calc Wave Speeds': False,
        'Plot Wave Speeds': False,
        'Files Processed': [],
        'Files Not Processed': [],
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
    }

def test_kymo_workflow(default_log_params):
    known_results = pd.read_csv('tests/assets/kymo/known_kymograph_summary.csv')
    assert isinstance(known_results, pd.DataFrame)
    exp_results = combined_workflow(
        folder_path=str(Path('tests/assets/kymo')),
        group_names=default_log_params['Group Names'],
        log_params=default_log_params,
        analysis_type='kymograph',
        box_size=None, #type: ignore
        bin_shift=default_log_params['Line Shift(px)'],
        line_width=default_log_params['Line Size(px)'],
        acf_peak_thresh=default_log_params['ACF Peak Prominence'],
        ccf_peak_thresh=default_log_params['CCF Peak Prominence'],
        small_shifts_correction=default_log_params['Small Shifts Correction'],
        plot_flags=default_log_params['plot_flags'],
        calc_wave_speeds=default_log_params['Calc Wave Speeds'],
        plot_wave_speeds=default_log_params['Plot Wave Speeds'],
        smoothing_params=default_log_params['smoothing_params'],
        smoothing=default_log_params['Smoothing'],
        test=True,
    )
    pd.testing.assert_frame_equal(
        known_results.reset_index(drop=True),
        exp_results.reset_index(drop=True),
        atol=1e-1,
    )
