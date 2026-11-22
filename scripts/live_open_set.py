#!/usr/bin/env python3
"""Read-only microphone/WAV inference with the frozen open-set experiment."""

import argparse
import hashlib
import json
import math
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

from speaker_id.audio import AudioData, load_audio
from speaker_id.audio.devices import (
    get_default_input_device,
    get_input_device,
    list_input_devices,
)
from speaker_id.audio.recorder import record_until_stop
from speaker_id.features.config import load_mfcc_config
from speaker_id.features.mfcc import extract_mfcc_vector
from speaker_id.features.segmentation import segment_audio
from speaker_id.models import mean_recording_scores
from speaker_id.open_set import OpenSetDecision, open_set_decision

ROOT = Path(__file__).resolve().parents[1]
KNOWN_CLASSES = ("V01", "V02", "V03", "V04")


@dataclass(frozen=True)
class InferenceResult:
    decision: OpenSetDecision
    scores: dict[str, float]
    runner_up: str
    margin: float
    segments: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live open-set speaker identification (ECAPA recommended for demo).",
        epilog=(
            "Examples: live_open_set.py --list-devices | "
            "live_open_set.py --representation ecapa | "
            "live_open_set.py --representation mfcc --duration 8 | "
            "live_open_set.py --representation ecapa --wav sample.wav. "
            "SVM decision scores are not probabilities; UNKNOWN uses the frozen threshold."
        ),
    )
    parser.add_argument("--representation", choices=("ecapa", "mfcc"), default="ecapa")
    parser.add_argument("--list-devices", action="store_true", help="list input devices and exit")
    parser.add_argument("--device", type=int, help="input device index, as in record_session.py")
    parser.add_argument(
        "--duration", type=float, default=6.0, help="capture duration in seconds (minimum 3)"
    )
    parser.add_argument("--wav", type=Path, help="infer from a WAV instead of opening a microphone")
    parser.add_argument(
        "--ecapa-dir", type=Path,
        help="trusted local ECAPA resource directory; verified offline, never downloads",
    )
    parser.add_argument("--verbose", action="store_true", help="show all mean SVM decision scores")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_assets(representation: str, root: Path = ROOT) -> tuple[dict, Pipeline]:
    """Validate provenance and load only the model named by the freeze manifest."""

    freeze_path = root / "docs/FINAL_OPEN_SET_A_FREEZE.json"
    if not freeze_path.is_file():
        raise ValueError(f"Freeze manifest not found: {freeze_path}")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze["experiment_id"] != "FINAL-OPEN-SET-A"
        or freeze["status"] != "FROZEN_VALIDATION_ONLY"
    ):
        raise ValueError("Unexpected freeze manifest identity or status")
    if tuple(freeze["protocol"]["class_order"]) != KNOWN_CLASSES:
        raise ValueError("Frozen classes must be V01-V04 in order")
    if freeze["protocol"]["acceptance_inequality"] != "top_score >= threshold":
        raise ValueError("Unexpected frozen acceptance rule")
    if representation not in ("mfcc", "ecapa"):
        raise ValueError(f"Unsupported representation: {representation}")
    config = load_mfcc_config(root / "configs/mfcc.yaml")
    if _sha256(root / "configs/mfcc.yaml") != freeze["configuration"]["mfcc_config_sha256"]:
        raise ValueError("MFCC configuration hash differs from freeze")
    segment = freeze["configuration"]["segmentation"]
    if segment != {
        "sample_rate_hz": config.sample_rate_hz,
        "segment_seconds": config.segment_duration_seconds,
        "hop_seconds": config.segment_hop_seconds,
        "minimum_padded_tail_seconds": config.minimum_final_segment_seconds,
    }:
        raise ValueError("Segmentation configuration differs from freeze")
    classifier = freeze["configuration"]["classifier"]
    if classifier != {
        "pipeline": ["StandardScaler", "SVC"],
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "class_weight": "balanced",
        "decision_function_shape": "ovr",
        "probability": False,
    }:
        raise ValueError("Classifier configuration differs from frozen protocol")
    if freeze["configuration"]["ecapa"] != {
        "source": "speechbrain/spkrec-ecapa-voxceleb",
        "pretrained": True,
        "frozen": True,
        "fine_tuned": False,
        "dimension": 192,
    }:
        raise ValueError("ECAPA configuration differs from frozen protocol")
    mfcc = freeze["configuration"]["mfcc"]
    if mfcc != {
        "frame_ms": config.frame_length_ms,
        "hop_ms": config.frame_hop_ms,
        "n_fft": round(config.frame_length_ms * config.sample_rate_hz / 1000),
        "hop_length": round(config.frame_hop_ms * config.sample_rate_hz / 1000),
        "center": False,
        "mel_filters": config.n_mels,
        "coefficients": config.n_mfcc,
        "delta_orders": list(config.delta_orders),
        "temporal_aggregation": ["mean", "std"],
        "dimension": config.feature_dimension,
    }:
        raise ValueError("MFCC configuration differs from frozen protocol")
    artifact = freeze["representations"][representation]
    threshold = artifact["calibration"]["threshold"]
    if not isinstance(threshold, (float, int)) or not math.isfinite(threshold):
        raise ValueError("Frozen threshold must be finite")
    model_path = root / freeze["artifacts_root"] / representation / "model/model.joblib"
    if not model_path.is_file():
        raise ValueError(f"Frozen model artifact not found: {model_path}")
    actual_hash = _sha256(model_path)
    if actual_hash != artifact["model_sha256"]:
        raise ValueError(f"Frozen model SHA256 mismatch at {model_path}")
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        model = joblib.load(model_path)
    if not isinstance(model, Pipeline) or tuple(name for name, _ in model.steps) != (
        "scaler", "svm"
    ):
        raise ValueError("Frozen model must be a scaler + SVM pipeline")
    scaler, svm = model.named_steps["scaler"], model.named_steps["svm"]
    if not isinstance(scaler, StandardScaler) or not isinstance(svm, SVC):
        raise ValueError("Frozen model must contain StandardScaler and SVC")
    check_is_fitted(scaler)
    check_is_fitted(svm)
    if not scaler.with_mean or not scaler.with_std:
        raise ValueError("Frozen scaler must use mean and standard deviation")
    if any(getattr(svm, key) != classifier[key] for key in (
        "kernel", "C", "gamma", "class_weight", "decision_function_shape", "probability"
    )):
        raise ValueError("Frozen SVC parameters differ from freeze")
    if tuple(str(label) for label in model.classes_) != KNOWN_CLASSES:
        raise ValueError(f"Frozen model classes differ from V01-V04: {model.classes_}")
    if model.n_features_in_ != freeze["configuration"][representation]["dimension"]:
        raise ValueError("Frozen model input dimension differs from freeze")
    return freeze, model


def load_local_encoder(directory: Path | None = None):
    """Load only the explicitly fingerprinted local resources, without network fallback."""

    from speaker_id.features.ecapa import EcapaEncoder

    resources = json.loads((ROOT / "configs/ecapa_resources.json").read_text(encoding="utf-8"))
    if directory is None:
        bundled = ROOT / "artifacts/ecapa_encoder"
        directory = bundled if bundled.exists() else (
            Path.home() / ".cache/speechbrain/spkrec-ecapa-voxceleb"
        )
    return EcapaEncoder.from_local_resources(directory, resources["sha256"])


def capture_audio(duration: float, device_index: int | None) -> AudioData:
    """Capture in memory, then use the canonical WAV loader via a temporary FLOAT WAV."""

    device = (
        get_input_device(device_index)
        if device_index is not None
        else get_default_input_device()
    )
    print(f"Input device: [{device.index}] {device.name}")
    print("Prepare-se para falar...")
    print(f"Recording for {duration:.1f} seconds...")
    captured = record_until_stop(
        device=device.index,
        sample_rate=device.default_samplerate,
        channels=1,
        max_duration_seconds=duration,
    )
    if captured.samples.ndim != 1 or not np.isfinite(captured.samples).all():
        raise ValueError("Captured audio must be finite and mono")
    with tempfile.TemporaryDirectory(prefix="live-open-set-") as temporary:
        wav_path = Path(temporary) / "capture.wav"
        sf.write(wav_path, captured.samples, captured.sample_rate, subtype="FLOAT")
        return load_audio(wav_path)


def infer_audio(
    audio: AudioData, representation: str, freeze: dict, model: Pipeline,
    *, ecapa_dir: Path | None = None,
) -> InferenceResult:
    """Apply frozen segmentation, representation, score aggregation and rejection."""

    config = load_mfcc_config(ROOT / "configs/mfcc.yaml")
    if audio.sample_rate != config.sample_rate_hz or audio.samples.dtype != np.float32:
        raise ValueError("Inference audio must be canonical mono 16 kHz float32")
    if audio.samples.ndim != 1 or not np.isfinite(audio.samples).all():
        raise ValueError("Inference audio must be finite and mono")
    if audio.samples.size < round(config.segment_duration_seconds * config.sample_rate_hz):
        raise ValueError("Audio must be at least 3 seconds long")
    segments = segment_audio(audio.samples, None, config)
    if not segments:
        raise ValueError("No valid segments could be produced")
    if representation == "mfcc":
        vectors = [extract_mfcc_vector(segment.samples, config) for segment in segments]
    else:
        encoder = load_local_encoder(ecapa_dir)
        vectors = [
            encoder.encode(segment.samples, sample_rate=audio.sample_rate)
            for segment in segments
        ]
    features = np.stack(vectors)
    expected_dimension = freeze["configuration"][representation]["dimension"]
    if features.shape != (len(segments), expected_dimension) or not np.isfinite(features).all():
        raise ValueError("Inference features have invalid dimension or non-finite values")
    top_class, scores = mean_recording_scores(model, features)
    classes = list(freeze["protocol"]["class_order"])
    if tuple(scores) != tuple(classes):
        raise ValueError("Model score class order differs from freeze")
    threshold = freeze["representations"][representation]["calibration"]["threshold"]
    decision = open_set_decision([scores[label] for label in classes], classes, threshold)
    if decision.top_class != top_class:
        raise ValueError("Aggregated top class differs from open-set decision")
    runner_up = sorted(classes, key=lambda label: scores[label], reverse=True)[1]
    return InferenceResult(
        decision=decision,
        scores=scores,
        runner_up=runner_up,
        margin=decision.top_score - scores[runner_up],
        segments=len(segments),
    )


def print_result(result: InferenceResult, representation: str, verbose: bool) -> None:
    decision = result.decision
    print("========================================")
    print("OPEN-SET SPEAKER IDENTIFICATION")
    print("========================================")
    print(f"Representation : {representation.upper()}")
    print(f"Segments       : {result.segments}")
    print(f"{'Top known' if decision.accepted else 'Closest known':15}: {decision.top_class}")
    print(f"Top score      : {decision.top_score:.4f}")
    print(f"Runner-up      : {result.runner_up}")
    print(f"Margin         : {result.margin:.4f}")
    print(f"Threshold      : {decision.threshold:.4f}")
    if verbose:
        print("SVM decision scores (mean per class, not probabilities):")
        for label, score in result.scores.items():
            print(f"  {label}: {score:.6f}")
    print(f"Decision       : {'KNOWN' if decision.accepted else 'UNKNOWN'}")
    print(f"Prediction     : {decision.prediction}")
    print("========================================")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.list_devices:
            devices = list_input_devices()
            default = get_default_input_device() if devices else None
            if not devices:
                print("No input devices found.")
            for device in devices:
                marker = " (default)" if default and device.index == default.index else ""
                print(
                    f"[{device.index}] {device.name} | input channels: "
                    f"{device.max_input_channels} | default sample rate: "
                    f"{device.default_samplerate:g} Hz{marker}"
                )
            return 0
        if args.wav is not None and args.device is not None:
            parser.error("--wav and --device cannot be used together")
        if not math.isfinite(args.duration) or args.duration < 3.0:
            parser.error("--duration must be finite and at least 3 seconds")
        freeze, model = load_frozen_assets(args.representation)
        print("Live output is a model decision score, not a probability.")
        audio = load_audio(args.wav) if args.wav is not None else capture_audio(
            args.duration, args.device
        )
        if args.representation == "ecapa":
            result = infer_audio(
                audio, args.representation, freeze, model, ecapa_dir=args.ecapa_dir
            )
        else:
            result = infer_audio(audio, args.representation, freeze, model)
        print_result(result, args.representation, args.verbose)
        return 0
    except KeyboardInterrupt:
        print("\nCapture interrupted; no dataset files were written.")
        return 130
    except Exception as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
