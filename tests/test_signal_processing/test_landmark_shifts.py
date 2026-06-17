"""
Tests for the landmark-based shift metrics (correlation_functions.py):
calc_indv_landmark_shifts and calc_indv_landmark_shift_workflow.

These measure the inter-channel offset separately at the peak apex, the
rising-edge crossing, and the falling-edge crossing at the configured edge
height. The rise/fall differences distinguish a pure phase shift (same waveform
shape) from channels whose waveforms rise or fall on different timescales.
"""
import numpy as np
from waveanalysis.signal_processing.correlation_functions import (
    calc_indv_landmark_shifts,
    calc_indv_landmark_shift_workflow,
    calc_indv_edge_times_workflow,
)


def _make_wave(apexes, rise_len, fall_len, length):
    """Piecewise-linear pulse train: each apex rises over rise_len frames to 1.0
    and falls over fall_len frames back to 0.0. At the default 50% edge height,
    crossings sit exactly rise_len/2 frames before and fall_len/2 frames after
    each apex."""
    s = np.zeros(length)
    for a in apexes:
        for k in range(rise_len + 1):
            idx = a - rise_len + k
            if 0 <= idx < length:
                s[idx] = max(s[idx], k / rise_len)
        for k in range(fall_len + 1):
            idx = a + k
            if 0 <= idx < length:
                s[idx] = max(s[idx], 1 - k / fall_len)
    return s


# ── calc_indv_landmark_shifts ─────────────────────────────────────────────────

def test_identical_signals_are_zero():
    signal = np.sin(np.linspace(0, 6 * np.pi, 300)) + 1
    peak_shift, rise_shift, fall_shift = calc_indv_landmark_shifts(
        signal, signal.copy(), period=100, peak_prominence_fraction=0.1
    )
    assert np.isclose(peak_shift, 0.0, atol=1e-6)
    assert np.isclose(rise_shift, 0.0, atol=1e-6)
    assert np.isclose(fall_shift, 0.0, atol=1e-6)


def test_pure_phase_shift_has_matching_landmarks():
    # signal1 is signal2 delayed by a fixed number of frames -> apex, rise, and
    # fall offsets are all equal, so the difference metrics are ~0.
    period = 100
    t = np.arange(600)
    shift_frames = 7
    signal2 = np.sin(2 * np.pi * t / period)
    signal1 = np.sin(2 * np.pi * (t - shift_frames) / period)

    peak_shift, rise_shift, fall_shift = calc_indv_landmark_shifts(
        signal1, signal2, period=period, peak_prominence_fraction=0.1
    )
    assert np.isclose(peak_shift, shift_frames, atol=1.0)
    assert np.isclose(rise_shift, shift_frames, atol=1.0)
    assert np.isclose(fall_shift, shift_frames, atol=1.0)
    assert abs(rise_shift - peak_shift) < 1.0
    assert abs(fall_shift - peak_shift) < 1.0


def test_differing_edge_kinetics_aligned_apexes():
    # Same apex positions in both channels (peak_shift ~ 0) but signal1 rises
    # twice as slowly and falls twice as fast, so its half-rise is earlier and
    # its half-fall is earlier too -> non-zero, distinct rise/fall shifts.
    apexes = [50, 150, 250]
    length = 300
    signal2 = _make_wave(apexes, rise_len=20, fall_len=20, length=length)  # half-rise apex-10, half-fall apex+10
    signal1 = _make_wave(apexes, rise_len=40, fall_len=10, length=length)  # half-rise apex-20, half-fall apex+5

    peak_shift, rise_shift, fall_shift = calc_indv_landmark_shifts(
        signal1, signal2, period=100, peak_prominence_fraction=0.1
    )
    assert np.isclose(peak_shift, 0.0, atol=1e-6)
    assert np.isclose(rise_shift, -10.0, atol=1e-6)   # (apex-20) - (apex-10)
    assert np.isclose(fall_shift, -5.0, atol=1e-6)    # (apex+5)  - (apex+10)
    assert np.isclose(rise_shift - peak_shift, -10.0, atol=1e-6)
    assert np.isclose(fall_shift - peak_shift, -5.0, atol=1e-6)


def test_differing_edge_kinetics_uses_configured_edge_height():
    apexes = [50, 150, 250]
    length = 300
    signal2 = _make_wave(apexes, rise_len=20, fall_len=20, length=length)
    signal1 = _make_wave(apexes, rise_len=40, fall_len=10, length=length)

    peak_shift, rise_shift, fall_shift = calc_indv_landmark_shifts(
        signal1, signal2, period=100, peak_prominence_fraction=0.1, edge_height_fraction=0.2
    )
    assert np.isclose(peak_shift, 0.0, atol=1e-6)
    assert np.isclose(rise_shift, -16.0, atol=1e-6)  # (apex-32) - (apex-16)
    assert np.isclose(fall_shift, -8.0, atol=1e-6)   # (apex+8)  - (apex+16)


def test_no_peaks_returns_nan():
    flat = np.zeros(300)
    sine = np.sin(np.linspace(0, 6 * np.pi, 300))
    peak_shift, rise_shift, fall_shift = calc_indv_landmark_shifts(
        flat, sine, period=100, peak_prominence_fraction=0.1
    )
    assert np.isnan(peak_shift)
    assert np.isnan(rise_shift)
    assert np.isnan(fall_shift)


# ── calc_indv_landmark_shift_workflow ─────────────────────────────────────────

def test_landmark_shift_workflow_shapes_and_values():
    apexes = [50, 150, 250]
    length = 300
    signal2 = _make_wave(apexes, rise_len=20, fall_len=20, length=length)
    signal1 = _make_wave(apexes, rise_len=40, fall_len=10, length=length)

    # kymograph layout: bin_values[channel, bin, frame]
    num_channels, num_bins = 2, 2
    bin_values = np.zeros((num_channels, num_bins, length))
    for b in range(num_bins):
        bin_values[0, b] = signal1
        bin_values[1, b] = signal2

    img_props = {
        'num_combos': 1,
        'num_bins': num_bins,
        'channel_combos': [(0, 1)],
        'analysis_type': 'kymograph',
        'peak_prominence_fraction': 0.1,
        'edge_height_fraction': 0.5,
    }
    indv_periods = np.full((num_channels, num_bins), 100.0)

    result = calc_indv_landmark_shift_workflow(
        bin_values=bin_values, indv_periods=indv_periods, img_props=img_props
    )

    expected = {
        'Peak Shift': 0.0,
        'Rise Shift': -10.0,
        'Fall Shift': -5.0,
        'Rise-Peak Diff': -10.0,
        'Fall-Peak Diff': -5.0,
    }
    assert set(result) == set(expected)
    for name, value in expected.items():
        assert result[name].shape == (1, num_bins)
        np.testing.assert_allclose(result[name], value, atol=1e-6)


# ── calc_indv_edge_times_workflow (per-channel, no comparison) ─────────────────

def test_edge_times_workflow_per_channel():
    # Ch1: slow rise (40) / fast fall (10) -> rise time 20, fall time 5
    # Ch2: symmetric (20/20)               -> rise time 10, fall time 10
    apexes = [50, 150, 250]
    length = 300
    ch1 = _make_wave(apexes, rise_len=40, fall_len=10, length=length)
    ch2 = _make_wave(apexes, rise_len=20, fall_len=20, length=length)

    num_channels, num_bins = 2, 2
    bin_values = np.zeros((num_channels, num_bins, length))
    for b in range(num_bins):
        bin_values[0, b] = ch1
        bin_values[1, b] = ch2

    img_props = {
        'num_channels': num_channels,
        'num_bins': num_bins,
        'analysis_type': 'kymograph',
        'peak_prominence_fraction': 0.1,
        'edge_height_fraction': 0.5,
    }

    result = calc_indv_edge_times_workflow(bin_values=bin_values, img_props=img_props)

    assert set(result) == {'Rise Time', 'Fall Time', 'Rise-Fall Time'}
    assert result['Rise Time'].shape == (num_channels, num_bins)
    np.testing.assert_allclose(result['Rise Time'][0], 20.0, atol=1e-6)  # Ch1
    np.testing.assert_allclose(result['Fall Time'][0], 5.0, atol=1e-6)
    np.testing.assert_allclose(result['Rise-Fall Time'][0], 15.0, atol=1e-6)
    np.testing.assert_allclose(result['Rise Time'][1], 10.0, atol=1e-6)  # Ch2
    np.testing.assert_allclose(result['Fall Time'][1], 10.0, atol=1e-6)
    np.testing.assert_allclose(result['Rise-Fall Time'][1], 0.0, atol=1e-6)
    # At the default 50% edge height, rise + fall equals the half-max width.
    np.testing.assert_allclose(result['Rise Time'] + result['Fall Time'],
                               [[25.0, 25.0], [20.0, 20.0]], atol=1e-6)


def test_edge_times_workflow_uses_configured_edge_height():
    apexes = [50, 150, 250]
    length = 300
    ch1 = _make_wave(apexes, rise_len=40, fall_len=10, length=length)
    ch2 = _make_wave(apexes, rise_len=20, fall_len=20, length=length)

    num_channels, num_bins = 2, 1
    bin_values = np.zeros((num_channels, num_bins, length))
    bin_values[0, 0] = ch1
    bin_values[1, 0] = ch2

    img_props = {
        'num_channels': num_channels,
        'num_bins': num_bins,
        'analysis_type': 'kymograph',
        'peak_prominence_fraction': 0.1,
        'edge_height_fraction': 0.2,
    }

    result = calc_indv_edge_times_workflow(bin_values=bin_values, img_props=img_props)

    np.testing.assert_allclose(result['Rise Time'][0], 32.0, atol=1e-6)
    np.testing.assert_allclose(result['Fall Time'][0], 8.0, atol=1e-6)
    np.testing.assert_allclose(result['Rise-Fall Time'][0], 24.0, atol=1e-6)
    np.testing.assert_allclose(result['Rise Time'][1], 16.0, atol=1e-6)
    np.testing.assert_allclose(result['Fall Time'][1], 16.0, atol=1e-6)
    np.testing.assert_allclose(result['Rise-Fall Time'][1], 0.0, atol=1e-6)


def test_edge_times_no_peaks_is_nan():
    length = 300
    flat = np.zeros(length)
    bin_values = flat.reshape(1, 1, length)
    img_props = {'num_channels': 1, 'num_bins': 1, 'analysis_type': 'kymograph',
                 'peak_prominence_fraction': 0.1}
    result = calc_indv_edge_times_workflow(bin_values=bin_values, img_props=img_props)
    assert np.isnan(result['Rise Time']).all()
    assert np.isnan(result['Fall Time']).all()
    assert np.isnan(result['Rise-Fall Time']).all()
