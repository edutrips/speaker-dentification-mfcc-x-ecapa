"""Local model installation verifies bytes and never loads or overwrites a model."""

import json
from pathlib import Path

import pytest
from scripts.install_inference_models import ROOT, install_models, sha256


def test_install_exact_models_is_idempotent_and_dataset_free(tmp_path):
    freeze = json.loads((ROOT / "docs/FINAL_OPEN_SET_A_FREEZE.json").read_text())
    for name in ("mfcc", "ecapa"):
        source = ROOT / freeze["artifacts_root"] / name / "model/model.joblib"
        if not source.is_file():
            pytest.skip("Install the trusted frozen artifacts locally first")
    first = install_models(ROOT, tmp_path)
    assert install_models(ROOT, tmp_path) == first
    assert len(first) == 2 and not (tmp_path / "data").exists()
    for row in first:
        assert row["verification"] == "PASS"
        assert sha256(Path(row["destination"])) == row["sha256"]


def test_bad_or_missing_source_never_creates_destination(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    with pytest.raises(ValueError, match="missing"):
        install_models(source, destination)
    bad = source / "artifacts/final_open_set_a/mfcc/model/model.joblib"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"untrusted model")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        install_models(source, destination)
    assert not destination.exists()


def test_different_destination_is_not_overwritten(tmp_path):
    if not (ROOT / "artifacts/final_open_set_a/mfcc/model/model.joblib").is_file():
        pytest.skip("Install the trusted frozen artifacts locally first")
    destination = tmp_path / "artifacts/final_open_set_a/mfcc/model/model.joblib"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"existing bytes")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        install_models(ROOT, tmp_path)
    assert destination.read_bytes() == b"existing bytes"
