
"""Tests for V1-V3 feature-difference calculations."""

import math

import pytest

from scripts.audit_v3_feature_shift import compare_features
from src.features.extract_features import FEATURE_COLUMNS


def feature_row():
    """Create one complete 23-feature test vector."""
    return {name: 0.0 for name in FEATURE_COLUMNS}


def test_identical_features():
    old = feature_row()
    new = feature_row()

    assert compare_features(old, new, 1) == []


def test_changed_feature_is_reported():
    old = feature_row()
    new = feature_row()
    new["overall_sqi"] = 0.9

    changes = compare_features(old, new, 42)

    assert len(changes) == 1
    assert changes[0]["ecg_id"] == 42
    assert changes[0]["feature"] == "overall_sqi"
    assert changes[0]["absolute_difference"] == pytest.approx(0.9)


def test_tiny_rounding_difference_is_ignored():
    old = feature_row()
    new = feature_row()
    new["st_V3"] = 1e-11

    assert compare_features(old, new, 1) == []


def test_missing_feature_is_rejected():
    old = feature_row()
    new = feature_row()
    del new["st_V3"]

    with pytest.raises(ValueError, match="Missing classifier feature"):
        compare_features(old, new, 1)


@pytest.mark.parametrize("invalid", [
    float("nan"),
    float("inf"),
    -float("inf"),
])
def test_nonfinite_value_is_rejected(invalid):
    old = feature_row()
    new = feature_row()
    new["st_V3"] = invalid

    with pytest.raises(ValueError, match="Non-finite"):
        compare_features(old, new, 1)


def test_inputs_are_not_modified():
    old = feature_row()
    new = feature_row()
    new["overall_sqi"] = 0.75

    old_copy = old.copy()
    new_copy = new.copy()

    compare_features(old, new, 5)

    assert old == old_copy
    assert new == new_copy
