
"""Load WFDB ECG recordings using CardioSense's input contract."""

from pathlib import Path

import wfdb

from src.dsp.input_validation import validate_ecg_input


def load_standardized_ecg(record_path):
    """Read a WFDB record and return a standardized ECG.

    Parameters
    ----------
    record_path:
        Record base path without the .hea or .dat extension.

    Returns
    -------
    signals_mv:
        ECG array with shape (samples, 12), in millivolts,
        ordered according to CardioSense's standard lead order.
    fs:
        Sampling frequency in Hz.
    lead_names:
        The standardized lead names.
    """

    # WFDB reads both waveform samples and header metadata.
    signals, fields = wfdb.rdsamp(str(Path(record_path)))

    # We cannot validate or convert an ECG without these fields.
    required = ("fs", "sig_name", "units")

    missing = [
        name for name in required
        if name not in fields
    ]

    if missing:
        raise ValueError(
            f"WFDB record is missing metadata: {missing}"
        )

    # Reuse our single validation function.
    # This verifies and standardizes the actual waveform data.
    signals_mv, lead_names = validate_ecg_input(
        raw_signals=signals,
        fs=fields["fs"],
        lead_names=fields["sig_name"],
        units=fields["units"],
    )

    # Preserve the original sample rate.
    # We do not resample or filter in this loader.
    fs = float(fields["fs"])

    return signals_mv, fs, lead_names
