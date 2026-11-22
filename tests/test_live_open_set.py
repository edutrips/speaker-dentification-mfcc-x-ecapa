"""Hardware-free tests for the read-only open-set CLI."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
import scripts.live_open_set as live
import soundfile as sf

from speaker_id.audio import AudioData, InputDevice, RecordedAudio


def _audio(seconds: int = 6) -> AudioData:
    samples = np.zeros(seconds * 16_000, dtype=np.float32)
    return AudioData(samples, 16_000, 16_000, 1, float(seconds))


def _freeze(threshold: float = 3.0) -> dict:
    return {
        "protocol": {"class_order": list(live.KNOWN_CLASSES)},
        "configuration": {"mfcc": {"dimension": 120}},
        "representations": {"mfcc": {"calibration": {"threshold": threshold}}},
    }


class FixedScores:
    classes_ = np.asarray(live.KNOWN_CLASSES)

    def __init__(self, scores: list[list[float]]) -> None:
        self.scores = np.asarray(scores, dtype=np.float64)
        self.fit_calls = 0

    def fit(self, *_args: object) -> None:
        self.fit_calls += 1
        pytest.fail("live inference must never fit")

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        assert features.shape == (3, 120)
        return self.scores


@pytest.mark.parametrize(
    ("scores", "threshold", "prediction", "accepted"),
    [
        ([[3.2, 1, 0, -1], [2.8, 1.2, 0, -1], [3.3, 1, 0, -1]], 3.0, "V01", True),
        ([[2.8, 1, 0, -1], [2.8, 1.2, 0, -1], [2.8, 1, 0, -1]], 3.0, "UNKNOWN", False),
        ([[3.0, 1, 0, -1], [3.0, 1.2, 0, -1], [3.0, 1, 0, -1]], 3.0, "V01", True),
    ],
)
def test_frozen_mean_scores_and_decision_without_fit(
    monkeypatch: pytest.MonkeyPatch,
    scores: list[list[float]],
    threshold: float,
    prediction: str,
    accepted: bool,
) -> None:
    monkeypatch.setattr(
        live, "extract_mfcc_vector", lambda _samples, _config: np.ones(120, dtype=np.float32)
    )
    model = FixedScores(scores)

    result = live.infer_audio(_audio(), "mfcc", _freeze(threshold), model)

    assert result.decision.prediction == prediction
    assert result.decision.accepted is accepted
    assert result.decision.threshold == threshold
    assert result.segments == 3
    assert result.scores["V01"] == pytest.approx(np.mean(np.asarray(scores)[:, 0]))
    assert result.runner_up == "V02"
    assert result.margin == pytest.approx(result.scores["V01"] - result.scores["V02"])
    assert model.fit_calls == 0


def _frozen_tree(tmp_path: Path) -> tuple[dict, Path]:
    freeze = json.loads(
        (live.ROOT / "docs/FINAL_OPEN_SET_A_FREEZE.json").read_text(encoding="utf-8")
    )
    config_path = tmp_path / "configs/mfcc.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_bytes((live.ROOT / "configs/mfcc.yaml").read_bytes())
    model_path = tmp_path / freeze["artifacts_root"] / "mfcc/model/model.joblib"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"test model bytes")
    (tmp_path / "docs").mkdir()
    return freeze, model_path


def _write_freeze(tmp_path: Path, freeze: dict) -> None:
    (tmp_path / "docs/FINAL_OPEN_SET_A_FREEZE.json").write_text(
        json.dumps(freeze), encoding="utf-8"
    )


def test_model_hash_mismatch_aborts_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze, _model_path = _frozen_tree(tmp_path)
    _write_freeze(tmp_path, freeze)
    monkeypatch.setattr(live.joblib, "load", lambda _path: pytest.fail("must not load model"))

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        live.load_frozen_assets("mfcc", tmp_path)


def test_invalid_model_classes_abort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    freeze, model_path = _frozen_tree(tmp_path)
    freeze["representations"]["mfcc"]["model_sha256"] = hashlib.sha256(
        model_path.read_bytes()
    ).hexdigest()
    _write_freeze(tmp_path, freeze)
    model = joblib.load(live.ROOT / freeze["artifacts_root"] / "mfcc/model/model.joblib")
    model.named_steps["svm"].classes_ = np.asarray(["V01", "V02", "V03", "V05"])
    monkeypatch.setattr(live.joblib, "load", lambda _path: model)

    with pytest.raises(ValueError, match="classes differ"):
        live.load_frozen_assets("mfcc", tmp_path)


def test_wav_mode_does_not_open_microphone_or_write_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    wav = tmp_path / "sample.wav"
    sf.write(wav, np.zeros(6 * 16_000, dtype=np.float32), 16_000, subtype="FLOAT")
    monkeypatch.setattr(live, "load_frozen_assets", lambda _representation: (_freeze(), object()))
    monkeypatch.setattr(
        live, "capture_audio", lambda *_args: pytest.fail("microphone must not open in WAV mode")
    )
    monkeypatch.setattr(
        live, "extract_mfcc_vector", lambda _samples, _config: np.ones(120, dtype=np.float32)
    )
    monkeypatch.setattr(
        live, "mean_recording_scores",
        lambda _model, _x: ("V01", dict.fromkeys(live.KNOWN_CLASSES, 0.0) | {"V01": 3.2}),
    )

    assert live.main(["--representation", "mfcc", "--wav", str(wav)]) == 0

    output = capsys.readouterr().out
    assert "Prediction     : V01" in output
    assert "not a probability" in output
    assert not (tmp_path / "data").exists()


def test_live_capture_uses_selected_device_and_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = InputDevice(7, "fake microphone", 1, 44_100)
    monkeypatch.setattr(live, "get_input_device", lambda index: selected if index == 7 else None)
    monkeypatch.setattr(
        live,
        "get_default_input_device",
        lambda: pytest.fail("explicit device should be used"),
    )
    calls: list[dict] = []

    def fake_record(**kwargs: object) -> RecordedAudio:
        calls.append(kwargs)
        samples = np.zeros(6 * 44_100, dtype=np.float32)
        return RecordedAudio(samples, 44_100, 1, 6.0)

    monkeypatch.setattr(live, "record_until_stop", fake_record)
    monkeypatch.setattr(live.tempfile, "tempdir", str(tmp_path))

    audio = live.capture_audio(6.0, 7)

    assert calls == [
        {"device": 7, "sample_rate": 44_100, "channels": 1, "max_duration_seconds": 6.0}
    ]
    assert audio.sample_rate == 16_000
    assert audio.samples.dtype == np.float32
    assert audio.duration_seconds == pytest.approx(6.0)
    assert list(tmp_path.iterdir()) == []
    assert not (tmp_path / "data").exists()


def test_live_mode_calls_capture_not_wav_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[float, int | None]] = []
    monkeypatch.setattr(live, "load_frozen_assets", lambda _representation: (_freeze(), object()))
    monkeypatch.setattr(live, "load_audio", lambda _path: pytest.fail("must not load a WAV"))

    def capture(duration: float, device: int | None) -> AudioData:
        seen.append((duration, device))
        return _audio()

    monkeypatch.setattr(live, "capture_audio", capture)
    monkeypatch.setattr(live, "infer_audio", lambda *_args: object())
    monkeypatch.setattr(live, "print_result", lambda *_args: None)

    assert live.main(["--representation", "mfcc", "--duration", "6", "--device", "7"]) == 0
    assert seen == [(6.0, 7)]


@pytest.mark.parametrize("duration", ["2.9", "nan", "inf"])
def test_invalid_duration_aborts_before_inference(duration: str) -> None:
    with pytest.raises(SystemExit) as error:
        live.main(["--duration", duration])
    assert error.value.code == 2


def test_short_audio_aborts_before_feature_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(live, "extract_mfcc_vector", lambda *_args: pytest.fail("too short"))
    with pytest.raises(ValueError, match="at least 3 seconds"):
        live.infer_audio(_audio(2), "mfcc", _freeze(), FixedScores([]))


def test_nonfinite_scores_abort(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        live, "extract_mfcc_vector", lambda _samples, _config: np.ones(120, dtype=np.float32)
    )
    model = FixedScores([[float("nan"), 1, 0, -1]] * 3)
    with pytest.raises(ValueError, match="invalid"):
        live.infer_audio(_audio(), "mfcc", _freeze(), model)


@pytest.mark.parametrize("representation,dimension", [("mfcc", 120), ("ecapa", 192)])
def test_exact_frozen_model_loads_without_historical_files(tmp_path, representation, dimension):
    freeze = json.loads((live.ROOT / "docs/FINAL_OPEN_SET_A_FREEZE.json").read_text())
    relative = Path(freeze["artifacts_root"]) / representation / "model/model.joblib"
    if not (live.ROOT / relative).is_file():
        pytest.skip("Frozen artifacts must be installed locally")
    (tmp_path / "docs").mkdir()
    shutil.copyfile(live.ROOT / "docs/FINAL_OPEN_SET_A_FREEZE.json",
                    tmp_path / "docs/FINAL_OPEN_SET_A_FREEZE.json")
    (tmp_path / "configs").mkdir()
    shutil.copyfile(live.ROOT / "configs/mfcc.yaml", tmp_path / "configs/mfcc.yaml")
    (tmp_path / relative).parent.mkdir(parents=True)
    shutil.copyfile(live.ROOT / relative, tmp_path / relative)

    loaded_freeze, model = live.load_frozen_assets(representation, tmp_path)

    assert model.n_features_in_ == dimension
    assert tuple(model.classes_) == live.KNOWN_CLASSES
    assert loaded_freeze["representations"][representation]["calibration"]["threshold"] == {
        "mfcc": 3.1804021896256245, "ecapa": 3.1886667008008707
    }[representation]
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / freeze["artifacts_root"] / "final_test").exists()


def test_missing_model_aborts_before_deserialization(tmp_path, monkeypatch):
    freeze, model_path = _frozen_tree(tmp_path)
    _write_freeze(tmp_path, freeze)
    model_path.unlink()
    monkeypatch.setattr(live.joblib, "load", lambda *_args: pytest.fail("must not deserialize"))
    with pytest.raises(ValueError, match="artifact not found"):
        live.load_frozen_assets("mfcc", tmp_path)


def test_ecapa_uses_only_local_encoder_and_ecapa_threshold(monkeypatch):
    calls = []

    class Encoder:
        def encode(self, samples, sample_rate):
            assert samples.shape == (48000,) and sample_rate == 16000
            calls.append("encode")
            return np.ones(192, dtype=np.float32)

    class Model:
        classes_ = np.asarray(live.KNOWN_CLASSES)

        def decision_function(self, features):
            assert features.shape == (3, 192)
            return np.tile([3.185, 1, 0, -1], (3, 1))

    monkeypatch.setattr(live, "load_local_encoder", lambda path: Encoder())
    monkeypatch.setattr(live, "extract_mfcc_vector", lambda *_args: pytest.fail("MFCC called"))
    freeze = _freeze(3.1804021896256245)
    freeze["configuration"]["ecapa"] = {"dimension": 192}
    freeze["representations"]["ecapa"] = {"calibration": {"threshold": 3.1886667008008707}}
    result = live.infer_audio(_audio(), "ecapa", freeze, Model(), ecapa_dir=Path("local"))
    assert calls == ["encode"] * 3
    assert result.decision.prediction == "UNKNOWN"
    assert result.decision.threshold == 3.1886667008008707


def test_mfcc_inference_does_not_import_neural_dependencies():
    code = """
import importlib.abc, sys
class RejectNeural(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchaudio', 'speechbrain'}:
            raise AssertionError('MFCC imported neural dependency: ' + fullname)
sys.meta_path.insert(0, RejectNeural())
import numpy as np
import scripts.live_open_set as live
freeze, model = live.load_frozen_assets('mfcc')
audio = live.AudioData(np.zeros(48000, dtype=np.float32), 16000, 16000, 1, 3.0)
result = live.infer_audio(audio, 'mfcc', freeze, model)
assert result.segments == 1
assert not any(name in sys.modules for name in ('torch', 'torchaudio', 'speechbrain'))
print('MFCC without neural dependencies PASS')
"""
    if not (live.ROOT / "artifacts/final_open_set_a/mfcc/model/model.joblib").is_file():
        pytest.skip("Frozen artifacts must be installed locally")
    environment = dict(os.environ, PYTHONPATH=str(live.ROOT / "src"),
                       PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=live.ROOT,
                            env=environment, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
