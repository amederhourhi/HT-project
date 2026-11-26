
"""Offline guardrail tests for the protected LUDB evaluation runner.

Tests use synthetic files and mocked Git state.
No real ECG recordings are opened.
"""

import hashlib
import json

import pytest

import scripts.run_rpeak_independent as runner


SHA = "a" * 40


def mock_clean_git(monkeypatch):
    """Simulate a clean, frozen main commit."""

    responses = {
        ("branch", "--show-current"): "main",
        ("rev-parse", "HEAD"): SHA,
        (
            "status",
            "--porcelain",
            "--untracked-files=all",
        ): "",
    }

    monkeypatch.setattr(
        runner,
        "git_output",
        lambda *args: responses[args],
    )


def test_development_cohort_is_fixed():
    assert runner.check_cohort(
        tuple(range(1, 31)),
        runner.DEV_IDS,
    ) == runner.DEV_IDS


def test_changed_cohort_rejected():
    with pytest.raises(
        ValueError,
        match="predefined evaluation cohort",
    ):
        runner.check_cohort(
            (101, 102),
            runner.HOLDOUT_IDS,
        )


def test_development_cannot_process_reserved_id():
    with pytest.raises(
        ValueError,
        match="outside permitted cohort",
    ):
        runner.evaluate_local_record(
            101,
            permitted_ids=runner.DEV_IDS,
        )


def test_missing_record_files_rejected(tmp_path):
    with pytest.raises(
        FileNotFoundError,
        match="required files are missing",
    ):
        runner.preflight_files(
            (101, 102),
            data_dir=tmp_path,
        )


def test_complete_file_preflight(tmp_path):
    for extension in ("hea", "dat", "ii"):
        (tmp_path / f"101.{extension}").write_bytes(
            b"synthetic"
        )

    runner.preflight_files(
        (101,),
        data_dir=tmp_path,
    )


def test_dataset_manifest_is_deterministic(tmp_path):
    for extension in ("hea", "dat", "ii"):
        (tmp_path / f"101.{extension}").write_bytes(
            extension.encode("ascii")
        )

    first = runner.dataset_manifest(
        (101,),
        data_dir=tmp_path,
    )

    second = runner.dataset_manifest(
        (101,),
        data_dir=tmp_path,
    )

    assert first == second
    assert len(first["files"]) == 3

    expected = hashlib.sha256(
        b"hea"
    ).hexdigest()

    assert first["files"][0]["sha256"] == expected


def test_holdout_requires_confirmation(monkeypatch):
    mock_clean_git(monkeypatch)

    with pytest.raises(
        PermissionError,
        match="confirmation",
    ):
        runner.authorize_holdout(
            "WRONG",
            SHA,
        )


def test_holdout_rejects_wrong_commit(monkeypatch):
    mock_clean_git(monkeypatch)

    with pytest.raises(
        PermissionError,
        match="does not match HEAD",
    ):
        runner.authorize_holdout(
            runner.CONFIRMATION,
            "b" * 40,
        )


def test_holdout_requires_main(monkeypatch):
    monkeypatch.setattr(
        runner,
        "git_output",
        lambda *args: (
            "research/test"
            if args == ("branch", "--show-current")
            else SHA
            if args == ("rev-parse", "HEAD")
            else ""
        ),
    )

    with pytest.raises(
        PermissionError,
        match="requires main",
    ):
        runner.authorize_holdout(
            runner.CONFIRMATION,
            SHA,
        )


def test_holdout_requires_clean_tree(monkeypatch):
    monkeypatch.setattr(
        runner,
        "git_output",
        lambda *args: (
            "main"
            if args == ("branch", "--show-current")
            else SHA
            if args == ("rev-parse", "HEAD")
            else "?? unexpected_file.py"
        ),
    )

    with pytest.raises(
        PermissionError,
        match="completely clean",
    ):
        runner.authorize_holdout(
            runner.CONFIRMATION,
            SHA,
        )


def test_valid_authorization(monkeypatch):
    mock_clean_git(monkeypatch)

    assert runner.authorize_holdout(
        runner.CONFIRMATION,
        SHA,
    ) == SHA


def test_attempt_marker_is_exclusive(tmp_path):
    marker = tmp_path / "attempt.json"

    runner.create_attempt_marker(
        marker,
        commit_sha=SHA,
    )

    metadata = json.loads(
        marker.read_text(encoding="utf-8")
    )

    assert metadata["status"] == "started"
    assert metadata["commit_sha"] == SHA

    with pytest.raises(FileExistsError):
        runner.create_attempt_marker(
            marker,
            commit_sha=SHA,
        )


def test_reserved_access_requires_authorization_first(
    monkeypatch,
):
    """Invalid authorization must stop before file inspection."""

    def forbidden_preflight(*args, **kwargs):
        raise AssertionError(
            "Reserved data must not be inspected"
        )

    monkeypatch.setattr(
        runner,
        "preflight_files",
        forbidden_preflight,
    )

    with pytest.raises(PermissionError):
        runner.run_holdout(
            confirmation="WRONG",
            expected_sha=SHA,
        )