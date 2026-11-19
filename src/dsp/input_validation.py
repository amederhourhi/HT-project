
"""Technical input checks for CardioSense's 12-lead ECG pipeline.

This module checks array structure and metadata. It does NOT decide whether a
recording has usable signal quality or whether it is clinically normal.
"""

from numbers import Real

import numpy as np


# Every ECG entering the existing models must follow this channel order.
STANDARD_LEADS = (
    "I", "II", "III", "AVR", "AVL", "AVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
)

# Convert declared voltage units into millivolts (mV).
# Never guess the units from the size of the waveform.
UNIT_TO_MV = {
    "mV": 1.0,
    "uV": 0.001,
    "µV": 0.001,
    "μV": 0.001,
    "V": 1000.0,
}


def validate_ecg_input(raw_signals, fs, lead_names, units):
    """Return (signals_mv, standard_lead_names) in the model's channel order.

    raw_signals:
        ECG matrix with shape (samples, 12).

    fs:
        Sampling frequency in Hz.

    lead_names:
        Lead labels matching the ORIGINAL waveform column order.

    units:
        Either one voltage unit for the entire ECG or a sequence of
        12 units, one per original channel (as provided by WFDB).

    Returns:
        A new float64 ECG array in millivolts and standard lead order.
        The original input is never modified.
    """

    # -------------------------------------------------------------
    # 1. Validate sampling frequency.
    # -------------------------------------------------------------
    # The existing filter uses a 40 Hz cutoff.
    # Nyquist frequency (fs / 2) must be greater than 40 Hz.
    if isinstance(fs, bool) or not isinstance(fs, Real):
        raise ValueError(
            "fs must be a numeric sampling frequency in Hz"
        )

    if not np.isfinite(fs) or fs <= 80:
        raise ValueError(
            "fs must be finite and greater than 80 Hz"
        )

    # -------------------------------------------------------------
    # 2. Validate the ECG lead labels.
    # -------------------------------------------------------------
    # We require explicit labels rather than silently assuming
    # that the incoming channels are already in the correct order.
    if lead_names is None or isinstance(lead_names, str):
        raise ValueError(
            "lead_names must explicitly identify all 12 channels"
        )

    try:
        names = list(lead_names)
    except TypeError as exc:
        raise ValueError(
            "lead_names must be a sequence"
        ) from exc

    if len(names) != 12 or any(
        not isinstance(name, str) for name in names
    ):
        raise ValueError(
            "lead_names must contain 12 strings"
        )

    # Normalize common naming differences: aVR -> AVR.
    names = [name.strip().upper() for name in names]

    if len(set(names)) != 12:
        raise ValueError("Duplicate ECG lead names")

    if set(names) != set(STANDARD_LEADS):
        missing = sorted(set(STANDARD_LEADS) - set(names))
        extra = sorted(set(names) - set(STANDARD_LEADS))

        raise ValueError(
            f"Unexpected lead set; missing={missing}, extra={extra}"
        )

    # -------------------------------------------------------------
    # 3. Validate the recording's amplitude units.
    # -------------------------------------------------------------
    # WFDB reports physical units per channel. This function accepts
    # both that format and a single unit applying to all channels.
    if isinstance(units, str):
        channel_units = [units] * 12
    else:
        try:
            channel_units = list(units)
        except TypeError as exc:
            raise ValueError(
                "units must be a string or 12 channel units"
            ) from exc

    if len(channel_units) != 12 or any(
        not isinstance(unit, str) or unit not in UNIT_TO_MV
        for unit in channel_units
    ):
        raise ValueError(
            "Each channel needs units: mV, uV, µV, μV, or V"
        )

    # -------------------------------------------------------------
    # 4. Validate the actual ECG waveform.
    # -------------------------------------------------------------
    try:
        signals = np.asarray(raw_signals)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "ECG must be a rectangular numeric array"
        ) from exc

    # We require 12 simultaneous channels.
    # This minimum length is structural only; downstream filtering
    # and SQI can impose stricter recording-length requirements.
    if (
        signals.ndim != 2
        or signals.shape[1] != 12
        or signals.shape[0] < 2
    ):
        raise ValueError(
            "ECG must have shape (at least 2 samples, 12 leads)"
        )

    # Reject strings, Boolean arrays, and complex-valued samples.
    if not (
        np.issubdtype(signals.dtype, np.number)
        and not np.issubdtype(
            signals.dtype, np.complexfloating
        )
        and not np.issubdtype(
            signals.dtype, np.bool_
        )
    ):
        raise ValueError(
            "ECG samples must be real numbers"
        )

    # Work with a copy so the original recording stays unchanged.
    signals = signals.astype(np.float64, copy=True)

    if not np.isfinite(signals).all():
        raise ValueError(
            "ECG samples contain NaN or infinity"
        )

    # -------------------------------------------------------------
    # 5. Convert all channels to millivolts.
    # -------------------------------------------------------------
    # Each multiplier belongs to its ORIGINAL waveform column.
    factors = np.array([
        UNIT_TO_MV[unit] for unit in channel_units
    ])

    with np.errstate(over="ignore", invalid="ignore"):
        signals_mv = signals * factors[np.newaxis, :]

    if not np.isfinite(signals_mv).all():
        raise ValueError(
            "Unit conversion resulted in non-finite ECG samples"
        )

    # -------------------------------------------------------------
    # 6. Reorder the waveforms into CardioSense's standard order.
    # -------------------------------------------------------------
    # Find the original column index for each required lead.
    # This moves the actual waveform data, not just the labels.
    indices = [
        names.index(lead) for lead in STANDARD_LEADS
    ]

    ordered_mv = signals_mv[:, indices].copy()

    return ordered_mv, list(STANDARD_LEADS)
