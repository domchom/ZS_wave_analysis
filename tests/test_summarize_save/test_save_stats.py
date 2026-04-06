"""
Tests for save_stats.py.

Covers: get_mean_CCF_values, get_indv_CCF_values, save_parameter_means_to_csv.
All inputs are constructed inline — no external assets needed.
"""
import numpy as np
import pandas as pd
import pytest
from waveanalysis.summarize_save.save_stats import (
    get_mean_CCF_values,
    get_indv_CCF_values,
    save_parameter_means_to_csv,
)

CHANNEL_COMBOS = [[0, 1]]
NUM_BINS = 3
CCF_LEN = 9        # 2*num_frames - 1 for num_frames=5
FRAME_INTERVAL = 0.5

def _make_indv_ccfs(seed=0):
    # shape: (num_combos, num_bins, ccf_len)
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, (len(CHANNEL_COMBOS), NUM_BINS, CCF_LEN))

def _make_bin_values(seed=1):
    # shape: (num_frames, num_channels, num_bins) for standard analysis
    num_frames = 5
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 100, (num_frames, 2, NUM_BINS))

def _make_img_props():
    return {
        'frame_interval': FRAME_INTERVAL,
        'num_bins': NUM_BINS,
        'analysis_type': 'standard',
        'channel_combos': CHANNEL_COMBOS,
    }


# ── get_mean_CCF_values ───────────────────────────────────────────────────────

def test_get_mean_CCF_values_returns_dict():
    result = get_mean_CCF_values(CHANNEL_COMBOS, _make_indv_ccfs(), FRAME_INTERVAL)
    assert isinstance(result, dict)

def test_get_mean_CCF_values_key_format():
    result = get_mean_CCF_values(CHANNEL_COMBOS, _make_indv_ccfs(), FRAME_INTERVAL)
    assert 'Ch1-Ch2 Mean CCF values' in result

def test_get_mean_CCF_values_entry_length():
    result = get_mean_CCF_values(CHANNEL_COMBOS, _make_indv_ccfs(), FRAME_INTERVAL)
    entries = result['Ch1-Ch2 Mean CCF values']
    assert len(entries) == CCF_LEN

def test_get_mean_CCF_values_entry_is_3_tuple():
    result = get_mean_CCF_values(CHANNEL_COMBOS, _make_indv_ccfs(), FRAME_INTERVAL)
    time, mean, std = result['Ch1-Ch2 Mean CCF values'][0]
    assert time == pytest.approx(0.0)   # first time point is 0 * frame_interval

def test_get_mean_CCF_values_mean_correct():
    indv_ccfs = _make_indv_ccfs()
    result = get_mean_CCF_values(CHANNEL_COMBOS, indv_ccfs, FRAME_INTERVAL)
    entries = result['Ch1-Ch2 Mean CCF values']
    expected_mean_at_0 = np.nanmean(indv_ccfs[0, :, 0])
    _, mean_at_0, _ = entries[0]
    assert mean_at_0 == pytest.approx(expected_mean_at_0)


# ── get_indv_CCF_values ───────────────────────────────────────────────────────

def test_get_indv_CCF_values_returns_dict():
    result = get_indv_CCF_values(_make_indv_ccfs(), _make_bin_values(), _make_img_props())
    assert isinstance(result, dict)

def test_get_indv_CCF_values_key_count():
    # one key per combo per bin
    result = get_indv_CCF_values(_make_indv_ccfs(), _make_bin_values(), _make_img_props())
    assert len(result) == len(CHANNEL_COMBOS) * NUM_BINS

def test_get_indv_CCF_values_key_format():
    result = get_indv_CCF_values(_make_indv_ccfs(), _make_bin_values(), _make_img_props())
    assert 'Ch1-Ch2 Bin 1 CCF' in result

def test_get_indv_CCF_values_entry_is_4_tuple():
    result = get_indv_CCF_values(_make_indv_ccfs(), _make_bin_values(), _make_img_props())
    first_entry = result['Ch1-Ch2 Bin 1 CCF'][0]
    assert len(first_entry) == 4  # (time, ch1_value, ch2_value, ccf_value)

def test_get_indv_CCF_values_entry_length():
    result = get_indv_CCF_values(_make_indv_ccfs(), _make_bin_values(), _make_img_props())
    entries = result['Ch1-Ch2 Bin 1 CCF']
    assert len(entries) == CCF_LEN


# ── save_parameter_means_to_csv ───────────────────────────────────────────────

def _make_summary_df():
    return pd.DataFrame({
        'File Name': ['Group1_file1.tif', 'Group1_file2.tif', 'Group2_file1.tif'],
        'Group Name': ['Group1', 'Group1', 'Group2'],
        'Ch 1 Mean Period': [1.0, 2.0, 3.0],
        'Ch 1 Mean Peak Amp': [4.0, 5.0, 6.0],
    })

def test_save_parameter_means_returns_dict():
    result = save_parameter_means_to_csv(_make_summary_df(), ['Group1', 'Group2'])
    assert isinstance(result, dict)

def test_save_parameter_means_key_count():
    result = save_parameter_means_to_csv(_make_summary_df(), ['Group1', 'Group2'])
    # one entry per 'Mean' column
    assert len(result) == 2

def test_save_parameter_means_key_format():
    result = save_parameter_means_to_csv(_make_summary_df(), ['Group1', 'Group2'])
    assert 'ch_1_mean_period_means.csv' in result

def test_save_parameter_means_value_is_dataframe():
    result = save_parameter_means_to_csv(_make_summary_df(), ['Group1', 'Group2'])
    for df in result.values():
        assert isinstance(df, pd.DataFrame)

def test_save_parameter_means_columns_are_group_names():
    result = save_parameter_means_to_csv(_make_summary_df(), ['Group1', 'Group2'])
    df = result['ch_1_mean_period_means.csv']
    assert set(df.columns) == {'Group1', 'Group2'}
