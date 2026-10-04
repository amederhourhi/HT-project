"""
CardioSense - Custom ECG Signal Filtering
Implements zero-phase Butterworth bandpass and notch filters
to remove baseline wander, powerline interference, and muscle tremor.
"""

import numpy as np
from scipy import signal


# ---------------------------------------------------------
# Block 1: Design a Butterworth Bandpass Filter (0.5 - 40 Hz)
# ---------------------------------------------------------
def get_bandpass_filter(lowcut=0.5, highcut=40.0, fs=100.0, order=3):
    """
    Creates filter coefficients (b, a) for a Butterworth bandpass filter.
    - lowcut (0.5 Hz): eliminates respiratory baseline drift.
    - highcut (40.0 Hz): eliminates high-frequency EMG muscle jitter.
    - fs: sampling frequency (100 Hz for our PTB-XL records).
    - order: filter steepness (3rd order provides clean roll-off without instability).
    """
    nyquist = 0.5 * fs                    # Nyquist frequency is half the sampling rate
    low = lowcut / nyquist                # Normalize cutoff to Nyquist frequency
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype="bandpass")
    return b, a


# ---------------------------------------------------------
# Block 2: Design a Notch Filter (50 Hz / 60 Hz Powerline)
# ---------------------------------------------------------
def get_notch_filter(notch_freq=50.0, fs=100.0, quality_factor=30.0):
    """
    Creates filter coefficients (b, a) for an IIR notch filter
    to remove specific 50 Hz or 60 Hz electrical mains noise.
    """
    b, a = signal.iirnotch(notch_freq, quality_factor, fs=fs)
    return b, a


# ---------------------------------------------------------
# Block 3: Zero-Phase Multi-Lead ECG Filter
# ---------------------------------------------------------
def filter_ecg(raw_signals, fs=100.0, lowcut=0.5, highcut=40.0):
    """
    Cleans a 1D or 2D (time x leads) ECG signal array.
    Uses 'filtfilt' (forward-backward filtering) to guarantee ZERO phase distortion,
    preserving exact landmark timings (P-wave, QRS-complex, ST-segment).
    """
    # Get Butterworth bandpass coefficients
    b, a = get_bandpass_filter(lowcut=lowcut, highcut=highcut, fs=fs)

    # Apply zero-phase filter along the time axis (axis 0)
    cleaned_signals = signal.filtfilt(b, a, raw_signals, axis=0)
    
    return cleaned_signals