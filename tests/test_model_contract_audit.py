
"""Unit tests for CardioSense's checkpoint audit helpers."""

import hashlib

import numpy as np

from scripts.audit_model_contracts import (
    inspect_cnn_weights,
    inspect_rf_schema,
    sha256_file,
)
from src.features.extract_features import FEATURE_COLUMNS
from src.models.ecg_cnn import CardioNet1D


class FakeRF:
    """Minimal model metadata matching the existing classifier."""

    feature_names_in_ = np.array(FEATURE_COLUMNS)
    n_features_in_ = len(FEATURE_COLUMNS)
    classes_ = np.array([0, 1])


def test_file_hash_matches_sha256(tmp_path):
    path = tmp_path / "sample.bin"
    content = b"CardioSense reproducibility"

    path.write_bytes(content)

    assert sha256_file(path) == hashlib.sha256(
        content
    ).hexdigest()


def test_rf_matching_schema_passes():
    result = inspect_rf_schema(FakeRF())

    assert result["structural_check"] == "pass"
    assert result["feature_order_matches"] is True
    assert result["class_order_matches"] is True


def test_rf_wrong_feature_order_fails():
    model = FakeRF()
    model.feature_names_in_ = np.array(
        list(reversed(FEATURE_COLUMNS))
    )

    result = inspect_rf_schema(model)

    assert result["structural_check"] == "fail"
    assert result["feature_order_matches"] is False


def test_rf_wrong_class_order_fails():
    model = FakeRF()
    model.classes_ = np.array([1, 0])

    result = inspect_rf_schema(model)

    assert result["structural_check"] == "fail"


def test_cnn_matching_weights_pass():
    model = CardioNet1D(in_channels=12, num_classes=2)
    expected = model.state_dict()

    result = inspect_cnn_weights(expected, expected)

    assert result["structural_check"] == "pass"
    assert not result["missing_keys"]
    assert not result["shape_mismatches"]


def test_cnn_missing_tensor_fails():
    model = CardioNet1D(in_channels=12, num_classes=2)
    expected = model.state_dict()

    altered = dict(expected)
    altered.pop(next(iter(altered)))

    result = inspect_cnn_weights(altered, expected)

    assert result["structural_check"] == "fail"
    assert len(result["missing_keys"]) == 1
