"""
Tests for summarize_image and combine_stats_for_image_kymo_standard
(summarize_images.py).

All inputs are constructed with synthetic numpy arrays — no external assets
needed. Validates DataFrame shape, column names, and that statistics
(Mean, Median, StdDev, SEM) are computed correctly.
"""
import numpy as np
import pandas as pd
import pytest
from waveanalysis.summarize_save.summarize_images import (
    summarize_image,
    combine_stats_for_image_kymo_standard,
)

NUM_BINS = 4
NUM_CHANNELS = 2
CHANNEL_COMBOS = [[0, 1]]

PER_CHANNEL_METRICS = [
    'Period', 'Peak Amp', 'Peak Rel Amp', 'Peak Width',
    'Peak Max', 'Peak Min', 'Peak Offset', 'Peak Area',
]

def _make_img_props():
    return {
        'num_bins': NUM_BINS,
        'num_channels': NUM_CHANNELS,
        'channel_combos': CHANNEL_COMBOS,
    }

def _make_img_metrics(seed=42):
    rng = np.random.default_rng(seed)
    metrics = {
        name: rng.uniform(1, 10, (NUM_CHANNELS, NUM_BINS))
        for name in PER_CHANNEL_METRICS
    }
    metrics['Shift'] = rng.uniform(-1, 1, (1, NUM_BINS))
    metrics['% Phase Shift'] = rng.uniform(-0.5, 0.5, (1, NUM_BINS))
    return metrics


# ── summarize_image ───────────────────────────────────────────────────────────

def test_summarize_image_returns_dataframe():
    df, _ = summarize_image(_make_img_metrics(), _make_img_props())
    assert isinstance(df, pd.DataFrame)

def test_summarize_image_columns():
    df, _ = summarize_image(_make_img_metrics(), _make_img_props())
    expected_cols = ['Parameter', 'Mean', 'Median', 'StdDev', 'SEM'] + [f'Bin {i}' for i in range(1, NUM_BINS + 1)]
    assert list(df.columns) == expected_cols

def test_summarize_image_row_count():
    # 8 per-channel metrics × 2 channels + Shift × 1 combo + % Phase Shift × 1 combo
    expected_rows = len(PER_CHANNEL_METRICS) * NUM_CHANNELS + 2
    df, _ = summarize_image(_make_img_metrics(), _make_img_props())
    assert len(df) == expected_rows

def test_summarize_image_mean_correct():
    img_metrics = _make_img_metrics()
    df, _ = summarize_image(img_metrics, _make_img_props())
    row = df[df['Parameter'] == 'Ch 1 Period'].iloc[0]
    assert row['Mean'] == pytest.approx(np.nanmean(img_metrics['Period'][0]))

def test_summarize_image_median_correct():
    img_metrics = _make_img_metrics()
    df, _ = summarize_image(img_metrics, _make_img_props())
    row = df[df['Parameter'] == 'Ch 2 Period'].iloc[0]
    assert row['Median'] == pytest.approx(np.nanmedian(img_metrics['Period'][1]))

def test_summarize_image_std_correct():
    img_metrics = _make_img_metrics()
    df, _ = summarize_image(img_metrics, _make_img_props())
    row = df[df['Parameter'] == 'Ch 1 Peak Amp'].iloc[0]
    assert row['StdDev'] == pytest.approx(np.nanstd(img_metrics['Peak Amp'][0]))

def test_summarize_image_sem_correct():
    img_metrics = _make_img_metrics()
    df, _ = summarize_image(img_metrics, _make_img_props())
    row = df[df['Parameter'] == 'Ch 1 Period'].iloc[0]
    expected_sem = np.nanstd(img_metrics['Period'][0]) / np.sqrt(NUM_BINS)
    assert row['SEM'] == pytest.approx(expected_sem)

def test_summarize_image_nan_handling():
    img_metrics = _make_img_metrics()
    # introduce NaNs in channel 1 Period
    img_metrics['Period'][0, 0] = np.nan
    df, _ = summarize_image(img_metrics, _make_img_props())
    row = df[df['Parameter'] == 'Ch 1 Period'].iloc[0]
    assert row['Mean'] == pytest.approx(np.nanmean(img_metrics['Period'][0]))

def test_summarize_image_shift_row_present():
    df, _ = summarize_image(_make_img_metrics(), _make_img_props())
    assert 'Ch1-Ch2 CCF Shift' in df['Parameter'].values

def test_summarize_image_returns_stats_dict():
    _, stats = summarize_image(_make_img_metrics(), _make_img_props())
    assert isinstance(stats, dict)
    assert 'Period' in stats
    assert 'Shift' in stats


# ── combine_stats_for_image_kymo_standard ────────────────────────────────────

def test_combine_stats_returns_dict():
    img_props = _make_img_props()
    img_metrics = _make_img_metrics()
    _, stats = summarize_image(img_metrics, img_props)
    result = combine_stats_for_image_kymo_standard('f.tif', 'Group1', img_props, img_metrics, stats)
    assert isinstance(result, dict)

def test_combine_stats_file_and_group_name():
    img_props = _make_img_props()
    img_metrics = _make_img_metrics()
    _, stats = summarize_image(img_metrics, img_props)
    result = combine_stats_for_image_kymo_standard('f.tif', 'Group1', img_props, img_metrics, stats)
    assert result['File Name'] == 'f.tif'
    assert result['Group Name'] == 'Group1'

def test_combine_stats_has_period_keys():
    img_props = _make_img_props()
    img_metrics = _make_img_metrics()
    _, stats = summarize_image(img_metrics, img_props)
    result = combine_stats_for_image_kymo_standard('f.tif', 'Group1', img_props, img_metrics, stats)
    assert 'Ch 1 Mean Period' in result
    assert 'Ch 2 Mean Period' in result

def test_combine_stats_has_shift_keys():
    img_props = _make_img_props()
    img_metrics = _make_img_metrics()
    _, stats = summarize_image(img_metrics, img_props)
    result = combine_stats_for_image_kymo_standard('f.tif', 'Group1', img_props, img_metrics, stats)
    assert 'Ch1-Ch2 Mean CCF Shift' in result

def test_combine_stats_mean_period_value():
    img_props = _make_img_props()
    img_metrics = _make_img_metrics()
    _, stats = summarize_image(img_metrics, img_props)
    result = combine_stats_for_image_kymo_standard('f.tif', 'Group1', img_props, img_metrics, stats)
    assert result['Ch 1 Mean Period'] == pytest.approx(np.nanmean(img_metrics['Period'][0]))
