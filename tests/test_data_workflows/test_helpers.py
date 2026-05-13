import numpy as np

from waveanalysis.data_workflows._helpers import _smooth_bin_values_inplace


def test_smooth_bin_values_promotes_uint16_to_float_without_trough_wraparound():
    signal = np.array([0, 0, 1000, 65000, 65000, 65000, 1000, 0, 0, 0, 0], dtype=np.uint16)
    bin_values = signal.reshape(11, 1, 1)

    smoothed = _smooth_bin_values_inplace(
        bin_values=bin_values,
        num_bins=1,
        num_channels=1,
        smoothing_params={"Ch1": {"window": 11, "poly_order": 2}},
    )

    assert np.issubdtype(smoothed.dtype, np.floating)
    assert smoothed.max() < np.iinfo(np.uint16).max
    assert not np.any(smoothed > signal.max())
