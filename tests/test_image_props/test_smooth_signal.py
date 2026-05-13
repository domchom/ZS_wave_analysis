import numpy as np

from waveanalysis.image_props.image_bin_calc import (
    _suppress_created_single_frame_extrema,
    smooth_signal,
)


def test_smooth_signal_suppresses_filter_created_single_frame_peak():
    raw_signal = np.array([0, 1, 2, 3, 4, 5, 6], dtype=float)
    smoothed_signal = np.array([0, 1, 2, 9, 4, 5, 6], dtype=float)

    corrected_signal = _suppress_created_single_frame_extrema(raw_signal, smoothed_signal)

    assert corrected_signal[3] <= raw_signal[2:5].max()
    assert corrected_signal[3] == 3


def test_smooth_signal_preserves_real_single_frame_peak():
    raw_signal = np.array([0, 0, 0, 10, 0, 0, 0], dtype=float)
    smoothed_signal = np.array([0, 0, 1, 8, 1, 0, 0], dtype=float)

    corrected_signal = _suppress_created_single_frame_extrema(raw_signal, smoothed_signal)

    assert np.array_equal(corrected_signal, smoothed_signal)


def test_smooth_signal_suppresses_filter_created_single_frame_trough():
    raw_signal = np.array([6, 5, 4, 3, 2, 1, 0], dtype=float)
    smoothed_signal = np.array([6, 5, 4, -3, 2, 1, 0], dtype=float)

    corrected_signal = _suppress_created_single_frame_extrema(raw_signal, smoothed_signal)

    assert corrected_signal[3] >= raw_signal[2:5].min()
    assert corrected_signal[3] == 3


def test_smooth_signal_keeps_signal_length():
    raw_signal = np.array([0, 1, 2, 3, 4, 5, 6], dtype=float)

    corrected_signal = smooth_signal(raw_signal, window=5, poly_order=2)

    assert corrected_signal.shape == raw_signal.shape
