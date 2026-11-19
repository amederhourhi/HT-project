
"""Tests for paired ECG alignment and SQI feature isolation."""

import numpy as np
import pandas as pd
import pytest

from scripts.audit_sqi_rf_ablation import (
    SHARED_FEATURES,
    pair_partition,
    verify_shared_features,
)
from src.features.extract_features import FEATURE_COLUMNS


def make_frame(ids, labels):
    frame = pd.DataFrame({
        "ecg_id": ids,
        "patient_id": ids,
        "target_label": labels,
        "strat_fold": [9] * len(ids),
    })

    for name in FEATURE_COLUMNS:
        frame[name] = 0.0

    return frame


def test_pairing_uses_same_ids():
    v1 = make_frame([3, 1, 2], [0, 0, 1])
    v3 = make_frame([1, 2, 3], [0, 1, 0])

    left, right = pair_partition(v1, v3)

    assert list(left.index) == [1, 2, 3]
    assert list(right.index) == [1, 2, 3]


def test_sqi_difference_is_allowed():
    v1 = make_frame([1, 2], [0, 1]).set_index("ecg_id")
    v3 = v1.copy()

    v3["overall_sqi"] = 0.95

    assert verify_shared_features(v1, v3)


def test_other_feature_difference_is_rejected():
    v1 = make_frame([1, 2], [0, 1]).set_index("ecg_id")
    v3 = v1.copy()

    v3.loc[1, "st_V3"] = 0.2

    with pytest.raises(ValueError, match="non-SQI"):
        verify_shared_features(v1, v3)


def test_wrong_labels_are_rejected():
    v1 = make_frame([1, 2], [0, 1])
    v3 = make_frame([1, 2], [1, 0])

    with pytest.raises(ValueError, match="target_label"):
        pair_partition(v1, v3)


def test_wrong_patient_ids_are_rejected():
    v1 = make_frame([1, 2], [0, 1])
    v3 = make_frame([1, 2], [0, 1])
    v3.loc[0, "patient_id"] = 999

    with pytest.raises(ValueError, match="patient IDs"):
        pair_partition(v1, v3)


def test_only_sqi_is_removed():
    assert len(FEATURE_COLUMNS) == 23
    assert len(SHARED_FEATURES) == 22
    assert "overall_sqi" not in SHARED_FEATURES
