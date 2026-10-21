
"""
CardioSense - Custom ECG Signal Filtering
Implements zero-phase Butterworth bandpass and notch filters.
"""

import numpy as np
from scipy import signal


def get_bandpass_filter(lowcut=0.5, highcut=40.0, fs=100.0, order=3):
    """Create Butterworth bandpass filter coefficients (b, a)."""
    try:
        fs = float(fs)
        lowcut = float(lowcut)
        highcut = float(highcut)
        order_value = float(order)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "fs, lowcut, highcut, and order must be numeric."
        ) from exc

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sampling frequency fs must be finite and positive.")

    if not (
        np.isfinite(lowcut)
        and np.isfinite(highcut)
        and 0 < lowcut < highcut < fs / 2
    ):
        raise ValueError(
            "Cutoffs must satisfy 0 < lowcut < highcut < fs / 2."
        )

    if (
        not np.isfinite(order_value)
        or order_value < 1
        or not order_value.is_integer()
    ):
        raise ValueError("Filter order must be a positive integer.")

    order = int(order_value)
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    return signal.butter(order, [low, high], btype="bandpass")


def get_notch_filter(notch_freq=50.0, fs=100.0, quality_factor=30.0):
    """Create IIR notch filter coefficients (b, a)."""
    try:
        fs = float(fs)
        notch_freq = float(notch_freq)
        quality_factor = float(quality_factor)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "fs, notch_freq, and quality_factor must be numeric."
        ) from exc

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sampling frequency fs must be finite and positive.")

    if not (
        np.isfinite(notch_freq)
        and 0 < notch_freq < fs / 2
    ):
        raise ValueError("notch_freq must be between 0 and fs / 2.")

    if not np.isfinite(quality_factor) or quality_factor <= 0:
        raise ValueError("quality_factor must be finite and positive.")

    return signal.iirnotch(notch_freq, quality_factor, fs=fs)


def filter_ecg(raw_signals, fs=100.0, lowcut=0.5, highcut=40.0):
    """
    Filter a 1D ECG signal or a 2D (samples, leads) array.

    Applies zero-phase Butterworth bandpass filtering along axis 0.
    The notch filter remains a separate helper, as in the original code.
    """
    try:
        signals = np.asarray(raw_signals)
    except (TypeError, ValueError) as exc:
        raise ValueError("ECG input must be a numeric array.") from exc

    if not np.issubdtype(signals.dtype, np.number):
        raise ValueError("ECG input must contain numeric values.")

    if np.iscomplexobj(signals):
        raise ValueError("ECG input must contain real-valued samples.")

    if signals.ndim not in (1, 2):
        raise ValueError("ECG input must be 1D or 2D (samples, leads).")

    if signals.shape[0] == 0:
        raise ValueError("ECG input must contain at least one sample.")

    if signals.ndim == 2 and signals.shape[1] == 0:
        raise ValueError("ECG input must contain at least one lead.")

    if not np.all(np.isfinite(signals)):
        raise ValueError("ECG input contains NaN or infinite values.")

    b, a = get_bandpass_filter(
        lowcut=lowcut,
        highcut=highcut,
        fs=fs,
        order=3,
    )

    padlen = 3 * max(len(a), len(b))

    if signals.shape[0] <= padlen:
        raise ValueError(
            f"ECG input has {signals.shape[0]} samples; "
            f"at least {padlen + 1} are required for this filter."
        )
    
    return signal.filtfilt(b, a, signals, axis=0)

