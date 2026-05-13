import numpy as np
import scipy.ndimage as nd


def _suppress_created_single_frame_extrema(
    raw_signal: np.ndarray,
    smoothed_signal: np.ndarray,
) -> np.ndarray:
    """
    Remove isolated extrema that are introduced by smoothing rather than present
    in the raw trace.
    """
    corrected_signal = smoothed_signal.copy()
    raw_float = np.asarray(raw_signal, dtype=float)
    smoothed_float = np.asarray(smoothed_signal, dtype=float)
    raw_range = np.nanmax(raw_float) - np.nanmin(raw_float)
    min_excursion = raw_range * 0.2

    for i in range(1, len(smoothed_float) - 1):
        left = smoothed_float[i - 1]
        center = smoothed_float[i]
        right = smoothed_float[i + 1]
        raw_local_min = np.nanmin(raw_float[i - 1:i + 2])
        raw_local_max = np.nanmax(raw_float[i - 1:i + 2])

        if center > left and center > right and center > raw_local_max + min_excursion:
            corrected_signal[i] = min((left + right) / 2, raw_local_max)
        elif center < left and center < right and center < raw_local_min - min_excursion:
            corrected_signal[i] = max((left + right) / 2, raw_local_min)

    return corrected_signal

def create_kymo_bin_array(
    image: np.ndarray,
    img_props: dict
) -> (np.ndarray, int): # type: ignore
    """
    Create a binary array for kymograph analysis.

    Args:
        image (np.ndarray): The input image array.
        img_props (dict): A dictionary containing image properties.

    Returns:
        np.ndarray: The line values array.
        int: The number of bins.

    Raises:
        ValueError: If the line width is less than 1.
    """
    # Get the image properties
    line_width = img_props["line_width"]
    step = img_props["step"]
    num_channels = img_props["num_channels"]
    num_frames = img_props["num_frames"]
    num_columns = img_props["num_columns"]

    if line_width < 1:
        raise ValueError("Line width must be at least 1")
    
    # Calculate the total amount of bins based on the step size
    num_bins = (num_columns // step) 

    # Initialize array to store line values
    line_values = np.full(shape=(num_channels, num_bins, num_frames), fill_value=np.nan)

    for channel in range(num_channels):
        # Loop through the columns of the image
        for col_num in range(0, num_columns, step):
            # Calculate the end column for the slice
            end_col = col_num + line_width
            # Check if the end column is within the image boundaries
            if end_col <= num_columns:
                # Extract the signal slice from the image
                signal_slice = image[channel, :, col_num:end_col]
                # Check if the signal slice has the correct shape
                if signal_slice.shape == (num_frames, line_width):
                    # Calculate the mean signal over the slice
                    signal = np.mean(signal_slice, axis=1)
                    # Calculate the index for the current bin
                    idx = col_num // step
                    if idx < num_bins:
                        # Store the signal in the line values array
                        line_values[channel, idx] = signal

    return line_values, num_bins

def create_multi_frame_bin_array(
    image: np.ndarray,
    img_props: dict
) -> (np.ndarray, int, int, int): # type: ignore
    """
    Create a multi-frame binary array based on the given image and image properties.

    Args:
        image (np.ndarray): The input image.
        img_props (dict): A dictionary containing image properties.

    Returns:
        np.ndarray: The multi-frame binary array.
    """
    # Get the image properties
    box_size = img_props["box_size"]
    step = img_props["step"]
    num_channels = img_props["num_channels"]
    num_frames = img_props["num_frames"]
    
    # Calculate the index for the center of the kernel
    ind = box_size // 2
    
    # Apply uniform filter to calculate mean signal over specified box size
    box_values = nd.uniform_filter(image[:, 0, :, :, :], size=(1, 1, box_size, box_size))[:, :, ind:-ind:step, ind:-ind:step]

    # Get the dimensions of the resulting mean image
    num_x_bins, num_y_bins = box_values.shape[-2:]
    num_bins = num_x_bins * num_y_bins
    box_values = box_values.reshape(num_frames, num_channels, num_bins)

    return box_values, num_bins, num_x_bins, num_y_bins

def smooth_signal(
    signal: np.ndarray,
    window: int,
    poly_order: int
) -> np.ndarray:
    """
    Smooth the input signal using Savitzky-Golay filter.

    Args:
        signal (np.ndarray): The input signal to be smoothed.
        window (int): The length of the filter window (must be a positive odd integer).
        poly_order (int): The order of the polynomial used to fit the samples (must be less than window).

    Returns:
        np.ndarray: The smoothed signal.
    """
    from scipy.signal import savgol_filter

    # Ensure the window size is odd and at least 3
    if window % 2 == 0:
        window += 1
    if window < 3:
        window = 3

    # Apply Savitzky-Golay filter to smooth the signal
    smoothed_signal = savgol_filter(signal, window_length=window, polyorder=poly_order)

    return _suppress_created_single_frame_extrema(signal, smoothed_signal)
