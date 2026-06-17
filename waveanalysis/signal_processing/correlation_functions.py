import numpy as np
from scipy import signal as sig

# Fraction of signal amplitude required as prominence to confirm both signals
# are oscillatory before computing CCF. Filters out flat or noisy signals.
# Not sure how much this specific value actually matters.
_CCF_SIGNAL_PEAK_PROMINENCE = 0.25

# Minimum prominence in the normalized CCF curve for a peak to be considered valid.
# This is not the user-configurable threshold for calculating the shift, 
# but rather a threshold to filter out noisy or flat CCF curves that do not have clear peaks.
_CCF_PEAK_PROMINENCE = 0.1

# If a detected shift exceeds this fraction of the period it is assumed to be a
# phase-aliased measurement and is corrected by ±1 period. 
# We chose 0.6 as the threshold to allow for some variability in the period while 
# still correcting for small shifts that are likely due to noise or phase aliasing. 
# This means that if the detected shift is greater than 60% of the average period, 
# it will be corrected by adding or subtracting the average period, depending on 
# the direction of the shift. This helps to improve the accuracy of the shift 
# measurements by reducing the impact of small, spurious shifts that may arise 
# from noise or other factors.
_SMALL_SHIFT_CORRECTION_THRESHOLD = 0.6

def _get_signal(bin_values: np.ndarray, channel: int, bin: int, analysis_type: str) -> np.ndarray:
    """Extract a single channel/bin signal from bin_values.

    standard analysis stores frames along axis 0: bin_values[frames, channels, bins]
    kymograph/rolling stores no frames axis: bin_values[channels, bins]
    """
    return bin_values[:, channel, bin] if analysis_type == 'standard' else bin_values[channel, bin]

def calc_indv_ACF_workflow(
    bin_values: np.ndarray,
    img_props: dict,
) -> np.ndarray: 
    '''
    Calculate individual Auto-Correlation Function (ACF) workflow.

    Args:
        bin_values (np.ndarray): The input bin values.
        img_props (dict): A dictionary containing image properties.

    Returns:
        np.ndarray: The calculated individual ACFs.
    '''
    # Extract image properties from the dictionary
    num_channels = img_props['num_channels']
    num_bins = img_props['num_bins']
    num_frames = img_props['num_frames']
    acf_peak_thresh = img_props['peak_thresh']
    analysis_type = img_props['analysis_type']

    # Initialize array to store the individual ACFs
    indv_acfs = np.zeros(shape=(num_channels, num_bins, num_frames * 2 - 1))

    # Loop through each channel and bin
    for channel in range(num_channels):
        for bin in range(num_bins):
            # Extract the bin values for the current channel and bin
            signal = _get_signal(bin_values, channel, bin, analysis_type)
            # Calculate and store the individual ACF for the current channel and bin
            acf_curve = calc_indv_ACF(signal=signal, num_frames=num_frames, peak_thresh=acf_peak_thresh)
            indv_acfs[channel, bin] = acf_curve

    return indv_acfs

def calc_indv_ACF(
    signal: np.ndarray,
    num_frames: int,
    peak_thresh: float,
) -> np.ndarray:
    '''
    Space saving function to calculate individual Auto-Correlation Function (ACF).
    '''
    # calc autocorrelation and normalize by the zero-lag value
    corr_signal = signal - np.mean(signal)
    acf_curve = np.correlate(corr_signal, corr_signal, mode='full')
    acf_curve = acf_curve / acf_curve[acf_curve.shape[0] // 2]

    # Find peaks in the autocorrelation curve. If less than two peaks found, return NaNs
    peaks, _ = sig.find_peaks(acf_curve, prominence=peak_thresh)
    if len(peaks) < 2:
        acf_curve = np.full((num_frames * 2 - 1), np.nan)

    return acf_curve

def calc_indv_period_workflow(
    acf_curve: np.ndarray,
    img_props: dict
) -> np.ndarray: 
    '''
    Calculate individual periods for each channel and bin based on the autocorrelation function curve.

    Parameters:
        acf_curve (np.ndarray): The autocorrelation function curve for each channel and bin.
        img_props (dict): A dictionary containing image properties, including the number of channels, number of bins, and peak threshold for the autocorrelation function.

    Returns:
        np.ndarray: An array of individual periods for each channel and bin.
    '''
    # Extract image properties from the dictionary
    num_channels = img_props['num_channels']
    num_bins = img_props['num_bins']
    acf_peak_thresh = img_props['peak_thresh']

    # Initialize array to store the individual periods
    indv_periods = np.zeros(shape=(num_channels, num_bins))

    # Loop through each channel and bin
    for channel in range(num_channels):
        for bin in range(num_bins):
            # Calculate and store the individual period for the current channel and bin
            period = calc_indv_period(acf_curve=acf_curve[channel, bin], peak_thresh=acf_peak_thresh)
            indv_periods[channel, bin] = period

    return indv_periods

def calc_indv_period(
    acf_curve: np.ndarray,
    peak_thresh: float,
) -> float:
    '''
    Space saving function to calculate individual periods for each channel and bin based on the autocorrelation function curve.
    '''
    center = acf_curve.shape[0] // 2
    peaks, _ = sig.find_peaks(acf_curve, prominence=peak_thresh)
    peaks_abs = np.abs(peaks - center)

    # Exclude the zero-lag peak, then pick the closest off-center peak.
    nonzero_mask = peaks_abs != 0
    if np.sum(nonzero_mask) < 1:
        return np.nan

    off_center_peaks = peaks[nonzero_mask]
    off_center_peaks_abs = peaks_abs[nonzero_mask]
    best_peak = off_center_peaks[np.argmin(off_center_peaks_abs)]
    return float(np.abs(best_peak - center))

def calc_indv_CCF_workflow(
    bin_values: np.ndarray,
    img_props: dict,
    ccf_smoothing: dict = None,
) -> np.ndarray:
    '''
    Calculate individual cross-correlation functions (CCFs) for each combination of channels and bins.

    Args:
        bin_values (np.ndarray): Array of bin values.
        img_props (dict): Dictionary containing image properties.

    Returns:
        np.ndarray: Array of individual CCFs.
    '''
    # Extract image properties from the dictionary
    num_combos = img_props['num_combos']
    num_bins = img_props['num_bins']
    num_frames = img_props['num_frames']
    channel_combos = img_props['channel_combos']
    analysis_type = img_props['analysis_type']

    # Initialize array to store the individual CCFs
    indv_ccfs = np.zeros(shape=(num_combos, num_bins, num_frames*2-1))
    
    # Loop through each combination of channels and bin
    for combo_number, combo in enumerate(channel_combos):
        for bin in range(num_bins):
            # Extract the bin values for the current channel and bin
            signal1 = _get_signal(bin_values, combo[0], bin, analysis_type)
            signal2 = _get_signal(bin_values, combo[1], bin, analysis_type)
            # Calculate and store the individual CCF for the current combination of channels and bin
            ccf = calc_indv_CCF(signal1=signal1, signal2=signal2, num_frames=num_frames, ccf_smoothing=ccf_smoothing)
            indv_ccfs[combo_number, bin] = ccf

    return indv_ccfs

def calc_indv_CCF(
    signal1: np.ndarray,
    signal2: np.ndarray,
    num_frames: int,
    ccf_smoothing: dict = None,
) -> np.ndarray:
    '''
    Space saving function to calculate individual cross-correlation functions (CCFs) for each combination of channels and bins.
    '''
    # Find peaks in the signals
    peaks1, _ = sig.find_peaks(signal1, prominence=(np.max(signal1)-np.min(signal1))*_CCF_SIGNAL_PEAK_PROMINENCE)
    peaks2, _ = sig.find_peaks(signal2, prominence=(np.max(signal2)-np.min(signal2))*_CCF_SIGNAL_PEAK_PROMINENCE)

    # If peaks are found in both signals
    if len(peaks1) > 0 and len(peaks2) > 0:
        # Subtract the mean from the signals
        corr_signal1 = signal1 - signal1.mean()
        corr_signal2 = signal2 - signal2.mean()
        # Calculate cross-correlation curve
        cc_curve = np.correlate(corr_signal1, corr_signal2, mode='full')

        # Normalize then optionally smooth the cross-correlation curve
        cc_curve = cc_curve / (num_frames * signal1.std() * signal2.std())
        if ccf_smoothing is not None:
            cc_curve = sig.savgol_filter(cc_curve, window_length=ccf_smoothing["window"], polyorder=ccf_smoothing["poly_order"])
        # Find peaks in the cross-correlation curve
        peaks, _ = sig.find_peaks(cc_curve, prominence=_CCF_PEAK_PROMINENCE)

        # If less than two peaks found, return NaNs
        if len(peaks) < 2:
            cc_curve = np.full((num_frames * 2 - 1), np.nan)

    else:
        # If no peaks found, return NaNs
        cc_curve = np.full((num_frames * 2 - 1), np.nan)

    return cc_curve

def calc_indv_shift_workflow(
    indv_ccfs: np.ndarray,
    indv_periods: np.ndarray,
    img_props: dict,
    small_shifts_correction: bool,
    ccf_peak_thresh: float
) -> np.ndarray:
    '''
    Calculate individual shifts for each channel combination and bin.

    Parameters:
        indv_ccfs (np.ndarray): Array of cross-correlation functions for each channel combination and bin.
        indv_periods (np.ndarray): Array of periods for each channel combination and bin.
        img_props (dict): Dictionary containing image properties.
        small_shifts_correction (bool): Flag to indicate if small shifts should be corrected.
        ccf_peak_thresh (float): Threshold for peak detection in the cross-correlation function.

    Returns:
        np.ndarray: Array of individual shifts for each channel combination and bin.
    '''
    # Extract image properties from the dictionary
    num_combos = img_props['num_combos']
    num_bins = img_props['num_bins']
    channel_combos = img_props['channel_combos']
    
    # Initialize array to store the individual shifts
    indv_shifts = np.zeros(shape=(num_combos, num_bins))

    # Loop through each combination of channels and bin
    for combo_number, combo in enumerate(channel_combos):
        for bin in range(num_bins):
            # Calculate and store the individual shift for the current combination of channels and bin
            shift = calc_indv_shift(cc_curve=indv_ccfs[combo_number, bin], ccf_peak_thresh=ccf_peak_thresh)
            if small_shifts_correction:
                average_period = np.mean(indv_periods[:, bin]) # If the shift is too small, correct it
                shift = correct_small_shifts(delay_frames=shift, average_period=average_period)
            indv_shifts[combo_number, bin] = shift

    return indv_shifts

def calc_indv_shift(cc_curve: np.ndarray,
                    ccf_peak_thresh: float) -> np.ndarray:
    '''
    Space saving function to calculate individual shifts for each channel combination and bin.
    '''
    # Find peaks in the cross-correlation curve
    peaks, _ = sig.find_peaks(cc_curve, prominence=ccf_peak_thresh)
    peaks_abs = abs(peaks - cc_curve.shape[0] // 2)

    # If multiple peaks found, select the one closest to the center
    if len(peaks) > 1:
        delay = np.argmin(peaks_abs[np.nonzero(peaks_abs)])
        delayIndex = peaks[delay]
        delay_frames = delayIndex - cc_curve.shape[0] // 2
    # Otherwise, return NaNs
    else:
        delay_frames = np.nan
    
    return delay_frames

def correct_small_shifts(
    delay_frames: float, 
    average_period: float
) -> float:
    '''
    Correct small shifts in the cross-correlation curve.
    '''
    # If the shift is larger than 60% of the average period, correct it by subtracting the average period
    if abs(delay_frames) > abs(average_period * _SMALL_SHIFT_CORRECTION_THRESHOLD):
        if delay_frames < 0:
            delay_frames = delay_frames + average_period
        elif delay_frames > 0:
            delay_frames = delay_frames - average_period

    return delay_frames


# Default fraction of signal amplitude (max - min) used as peak prominence when
# detecting the peaks that the landmark-shift offsets are measured from.
# Overridden by the user-supplied value passed through img_props.
_LANDMARK_SHIFT_PEAK_PROMINENCE_FRACTION = 0.1
_EDGE_HEIGHT_FRACTION = 0.5

# Matched peaks must fall within this fraction of the average period of each
# other, otherwise they are treated as belonging to different cycles and are not
# paired. Half a period is the largest gap that still unambiguously identifies
# the same oscillation in both channels.
_LANDMARK_SHIFT_MATCH_PERIOD_FRACTION = 0.5

def _peak_landmarks(
    signal: np.ndarray,
    peak_prominence_fraction: float,
    edge_height_fraction: float = _EDGE_HEIGHT_FRACTION,
) -> tuple:
    '''
    Locate, for every detected peak, three timing landmarks:
        - the peak apex (frame index of the maximum)
        - the rising-edge crossing at edge_height_fraction of that peak's prominence
        - the falling-edge crossing at edge_height_fraction of that peak's prominence

    Returns (apexes, rise_crossings, fall_crossings) as float arrays, or three
    empty arrays if no peaks are found.
    '''
    amplitude = np.max(signal) - np.min(signal)
    peaks, _ = sig.find_peaks(signal, prominence=amplitude * peak_prominence_fraction)
    if len(peaks) == 0:
        return np.array([]), np.array([]), np.array([])

    edge_height_fraction = float(np.clip(edge_height_fraction, 0.01, 0.99))
    # scipy's rel_height is measured downward from the peak prominence. A 20%
    # edge-height crossing is therefore rel_height=0.8; 50% remains 0.5.
    _, _, left_ips, right_ips = sig.peak_widths(signal, peaks, rel_height=1.0 - edge_height_fraction)
    return peaks.astype(float), left_ips, right_ips

def calc_indv_landmark_shifts(
    signal1: np.ndarray,
    signal2: np.ndarray,
    period: float,
    peak_prominence_fraction: float,
    edge_height_fraction: float = _EDGE_HEIGHT_FRACTION,
) -> tuple:
    '''
    Landmark-based inter-channel timing offsets from matched peaks.

    Where the CCF shift collapses the whole-waveform relationship into a single
    lag (dominated by the most correlated phase), this measures the offset
    separately at the peak apex, the rising edge, and the falling edge. When the
    two channels have different waveform shapes, these offsets disagree, and that
    disagreement is exactly the quantity of interest.

    Each signal1 peak is matched to the nearest signal2 peak (by apex), provided
    the two apexes fall within half a period of each other. For every matched
    pair the apex, rise, and fall offsets are computed as
    signal1 - signal2, then averaged over all matched pairs.

    Returns (peak_shift, rise_shift, fall_shift) in frames, or (nan, nan, nan) if
    no peaks match.
    '''
    apex1, rise1, fall1 = _peak_landmarks(signal1, peak_prominence_fraction, edge_height_fraction)
    apex2, rise2, fall2 = _peak_landmarks(signal2, peak_prominence_fraction, edge_height_fraction)
    if len(apex1) == 0 or len(apex2) == 0:
        return np.nan, np.nan, np.nan

    # Beyond half a period apart, two apexes belong to different cycles.
    tol = period * _LANDMARK_SHIFT_MATCH_PERIOD_FRACTION if np.isfinite(period) and period > 0 else np.inf

    peak_diffs, rise_diffs, fall_diffs = [], [], []
    for a1, r1, f1 in zip(apex1, rise1, fall1):
        j = int(np.argmin(np.abs(apex2 - a1)))
        if np.abs(apex2[j] - a1) <= tol:
            peak_diffs.append(a1 - apex2[j])
            rise_diffs.append(r1 - rise2[j])
            fall_diffs.append(f1 - fall2[j])

    if len(peak_diffs) == 0:
        return np.nan, np.nan, np.nan

    return (
        float(np.nanmean(peak_diffs)),
        float(np.nanmean(rise_diffs)),
        float(np.nanmean(fall_diffs)),
    )

def calc_indv_landmark_shift_workflow(
    bin_values: np.ndarray,
    indv_periods: np.ndarray,
    img_props: dict,
) -> dict:
    '''
    Calculate the landmark-based shift metrics for every channel combination and bin.

    For each combo/bin this measures the inter-channel offset (signal1 - signal2,
    in frames) at three waveform landmarks and forms two difference metrics:
        - 'Peak Shift':     offset at the peak apex (apex-to-apex)
        - 'Rise Shift':     offset at the selected rising-edge height crossing
        - 'Fall Shift':     offset at the selected falling-edge height crossing
        - 'Rise-Peak Diff': rise_shift - peak_shift
        - 'Fall-Peak Diff': fall_shift - peak_shift

    The difference metrics are the diagnostics: ~0 means the channels are a simple
    phase-shifted pair (same shape), while a non-zero value means their waveforms
    rise/fall on different timescales, so the apex shift alone misrepresents the
    lag at that edge.

    Args:
        bin_values (np.ndarray): Array of bin values.
        indv_periods (np.ndarray): Per-channel/bin periods in frames, used to set
            the peak-matching tolerance.
        img_props (dict): Dictionary containing image properties.

    Returns:
        dict: Maps each metric name above to an array shaped (num_combos, num_bins).
    '''
    num_combos = img_props['num_combos']
    num_bins = img_props['num_bins']
    channel_combos = img_props['channel_combos']
    analysis_type = img_props['analysis_type']
    peak_prominence_fraction = img_props.get('peak_prominence_fraction', _LANDMARK_SHIFT_PEAK_PROMINENCE_FRACTION)
    edge_height_fraction = img_props.get('edge_height_fraction', _EDGE_HEIGHT_FRACTION)

    peak_shifts = np.zeros(shape=(num_combos, num_bins))
    rise_shifts = np.zeros(shape=(num_combos, num_bins))
    fall_shifts = np.zeros(shape=(num_combos, num_bins))

    for combo_number, combo in enumerate(channel_combos):
        for bin in range(num_bins):
            signal1 = _get_signal(bin_values, combo[0], bin, analysis_type)
            signal2 = _get_signal(bin_values, combo[1], bin, analysis_type)
            # Average the two channels' periods for the matching tolerance.
            combo_period = np.nanmean(indv_periods[[combo[0], combo[1]], bin])
            peak_shift, rise_shift, fall_shift = calc_indv_landmark_shifts(
                signal1=signal1,
                signal2=signal2,
                period=combo_period,
                peak_prominence_fraction=peak_prominence_fraction,
                edge_height_fraction=edge_height_fraction,
            )
            peak_shifts[combo_number, bin] = peak_shift
            rise_shifts[combo_number, bin] = rise_shift
            fall_shifts[combo_number, bin] = fall_shift

    return {
        'Peak Shift': peak_shifts,
        'Rise Shift': rise_shifts,
        'Fall Shift': fall_shifts,
        'Rise-Peak Diff': rise_shifts - peak_shifts,
        'Fall-Peak Diff': fall_shifts - peak_shifts,
    }

# Amplitude fractions (of each peak's prominence) at which the rising- and
# falling-edge inter-channel lag is sampled for the lag-vs-threshold profile.
# 1.0 is the apex (= peak shift).
_LAG_PROFILE_FRACTIONS = (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)

def calc_indv_edge_lag_profile(
    signal1: np.ndarray,
    signal2: np.ndarray,
    period: float,
    peak_prominence_fraction: float,
    fractions: tuple = _LAG_PROFILE_FRACTIONS,
) -> tuple:
    '''
    Inter-channel lag (signal1 - signal2, in frames) sampled at several amplitude
    fractions along the rising and falling edges of matched peaks.

    For each fraction f the crossing on the rising edge is found via
    scipy.peak_widths at rel_height = 1 - f (so f = 1.0 is the apex itself), and
    likewise for the falling edge. Lags are averaged over all matched peak pairs.

    A flat profile across fractions means a pure phase shift; a sloped profile
    means the channels' edges move at different rates -- the same diagnostic as
    the rise/fall-peak differences, resolved across the whole edge.

    Returns (fractions, rise_lags, fall_lags) with the lag arrays aligned to
    `fractions`; lag entries are NaN where no peaks match.
    '''
    fractions = np.asarray(fractions, dtype=float)
    rise_lags = np.full(fractions.shape, np.nan)
    fall_lags = np.full(fractions.shape, np.nan)

    amp1 = np.max(signal1) - np.min(signal1)
    amp2 = np.max(signal2) - np.min(signal2)
    peaks1, _ = sig.find_peaks(signal1, prominence=amp1 * peak_prominence_fraction)
    peaks2, _ = sig.find_peaks(signal2, prominence=amp2 * peak_prominence_fraction)
    if len(peaks1) == 0 or len(peaks2) == 0:
        return fractions, rise_lags, fall_lags

    # Match each signal1 peak to the nearest signal2 peak (same rule as the
    # landmark shifts) so every fraction uses the same peak pairing.
    tol = period * _LANDMARK_SHIFT_MATCH_PERIOD_FRACTION if np.isfinite(period) and period > 0 else np.inf
    pairs = []
    for i, p1 in enumerate(peaks1):
        j = int(np.argmin(np.abs(peaks2 - p1)))
        if np.abs(peaks2[j] - p1) <= tol:
            pairs.append((i, j))
    if not pairs:
        return fractions, rise_lags, fall_lags

    peaks1_f = peaks1.astype(float)
    peaks2_f = peaks2.astype(float)
    for k, f in enumerate(fractions):
        rel = 1.0 - f
        if rel <= 0:  # apex: both edges collapse onto the peak index
            left1 = right1 = peaks1_f
            left2 = right2 = peaks2_f
        else:
            _, _, left1, right1 = sig.peak_widths(signal1, peaks1, rel_height=rel)
            _, _, left2, right2 = sig.peak_widths(signal2, peaks2, rel_height=rel)
        rise_lags[k] = np.nanmean([left1[i] - left2[j] for i, j in pairs])
        fall_lags[k] = np.nanmean([right1[i] - right2[j] for i, j in pairs])

    return fractions, rise_lags, fall_lags

def calc_indv_edge_times_workflow(
    bin_values: np.ndarray,
    img_props: dict,
) -> dict:
    '''
    Per-channel within-peak edge durations (no inter-channel comparison).

    For each channel/bin this averages, over that signal's peaks:
        'Rise Time': apex - rising-edge selected-height crossing (upstroke duration)
        'Fall Time': falling-edge selected-height crossing - apex (decay duration)
        'Rise-Fall Time': rise time minus fall time for each eligible peak

    These describe the shape of a single channel's own waveform; their sum is the
    selected-height peak width, while Rise-Fall Time reports waveform asymmetry
    (positive = slower rise than fall; negative = faster rise than fall). Returns
    a dict mapping each name to an array shaped (num_channels, num_bins), in frames.

    Args:
        bin_values (np.ndarray): Array of bin values.
        img_props (dict): Dictionary containing image properties.
    '''
    num_channels = img_props['num_channels']
    num_bins = img_props['num_bins']
    analysis_type = img_props['analysis_type']
    peak_prominence_fraction = img_props.get('peak_prominence_fraction', _LANDMARK_SHIFT_PEAK_PROMINENCE_FRACTION)
    edge_height_fraction = img_props.get('edge_height_fraction', _EDGE_HEIGHT_FRACTION)

    rise_times = np.full((num_channels, num_bins), np.nan)
    fall_times = np.full((num_channels, num_bins), np.nan)
    rise_fall_times = np.full((num_channels, num_bins), np.nan)

    for channel in range(num_channels):
        for bin in range(num_bins):
            signal = _get_signal(bin_values, channel, bin, analysis_type)
            apexes, rises, falls = _peak_landmarks(signal, peak_prominence_fraction, edge_height_fraction)
            if len(apexes) == 0:
                continue
            peak_rise_times = apexes - rises
            peak_fall_times = falls - apexes
            rise_times[channel, bin] = np.nanmean(peak_rise_times)
            fall_times[channel, bin] = np.nanmean(peak_fall_times)
            rise_fall_times[channel, bin] = np.nanmean(peak_rise_times - peak_fall_times)

    return {'Rise Time': rise_times, 'Fall Time': fall_times, 'Rise-Fall Time': rise_fall_times}


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    '''
    Normalize a signal to the range [0, 1].
    '''
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
