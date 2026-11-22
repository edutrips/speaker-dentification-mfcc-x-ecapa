"""Offline unit tests for the reusable ECAPA inference adapter."""

import numpy as np
import pytest
import torch

from speaker_id.features import DEFAULT_ECAPA_SOURCE, EcapaEncoder


class FakeClassifier:
    def __init__(self, output: torch.Tensor | None = None) -> None:
        self.output = output
        self.waveforms: list[torch.Tensor] = []
        self.inference_modes: list[bool] = []
        self.gradient_modes: list[bool] = []

    def encode_batch(self, waveform: torch.Tensor) -> torch.Tensor:
        self.waveforms.append(waveform.clone())
        self.inference_modes.append(torch.is_inference_mode_enabled())
        self.gradient_modes.append(torch.is_grad_enabled())
        if self.output is not None:
            return self.output
        values = torch.arange(4, dtype=torch.float32, device=waveform.device)
        return values.reshape(1, 1, 4) + waveform.mean()


def make_encoder(output: torch.Tensor | None = None) -> tuple[EcapaEncoder, FakeClassifier]:
    classifier = FakeClassifier(output)
    return EcapaEncoder(classifier), classifier


def test_valid_waveform_returns_finite_one_dimensional_embedding() -> None:
    encoder, _classifier = make_encoder()
    samples = np.linspace(-1, 1, 32, dtype=np.float32)

    embedding = encoder.encode(samples)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (4,)
    assert embedding.dtype == np.float32
    assert np.isfinite(embedding).all()


def test_embedding_is_deterministic_and_dimension_is_not_hardcoded() -> None:
    output = torch.arange(7, dtype=torch.float32).reshape(1, 1, 7)
    encoder, _classifier = make_encoder(output)
    samples = np.ones(5, dtype=np.float32)

    first = encoder.encode(samples)
    second = encoder.encode(samples)

    assert first.shape == (7,)
    np.testing.assert_array_equal(first, second)


def test_classifier_receives_unit_batch_float32_on_configured_device() -> None:
    encoder, classifier = make_encoder()
    encoder.encode(np.ones(13, dtype=np.float32))

    waveform = classifier.waveforms[0]
    assert waveform.shape == (1, 13)
    assert waveform.dtype == torch.float32
    assert waveform.device.type == "cpu"


@pytest.mark.parametrize(
    ("samples", "sample_rate", "message"),
    [
        ([0.0, 1.0], 16_000, "NumPy array"),
        (np.ones((2, 2), dtype=np.float32), 16_000, "one-dimensional"),
        (np.array([], dtype=np.float32), 16_000, "must not be empty"),
        (np.array([np.nan], dtype=np.float32), 16_000, "finite"),
        (np.array([np.inf], dtype=np.float32), 16_000, "finite"),
        (np.ones(2, dtype=np.float64), 16_000, "dtype float32"),
        (np.ones(2, dtype=np.float32), 8_000, "16000 Hz"),
    ],
)
def test_invalid_inputs_are_rejected(samples, sample_rate: int, message: str) -> None:
    encoder, _classifier = make_encoder()

    with pytest.raises(ValueError, match=message):
        encoder.encode(samples, sample_rate=sample_rate)


@pytest.mark.parametrize(
    "output",
    [
        torch.empty((1, 1, 0)),
        torch.empty((4,)),
        torch.empty((2, 1, 4)),
        torch.tensor([[[1.0, float("nan")]]]),
        torch.tensor([[[1.0, float("inf")]]]),
    ],
)
def test_invalid_model_outputs_are_rejected(output: torch.Tensor) -> None:
    encoder, _classifier = make_encoder(output)

    with pytest.raises(ValueError):
        encoder.encode(np.ones(8, dtype=np.float32))


def test_encoder_runs_with_inference_mode_and_gradients_disabled() -> None:
    encoder, classifier = make_encoder()

    encoder.encode(np.ones(8, dtype=np.float32))

    assert classifier.inference_modes == [True]
    assert classifier.gradient_modes == [False]


def test_model_defaults_and_properties() -> None:
    encoder, _classifier = make_encoder()

    assert encoder.source == DEFAULT_ECAPA_SOURCE
    assert encoder.device == torch.device("cpu")


def test_local_encoder_rejects_missing_or_changed_resources_before_loading(tmp_path):
    import hashlib

    names = ("hyperparams.yaml", "embedding_model.ckpt", "classifier.ckpt",
             "mean_var_norm_emb.ckpt", "label_encoder.txt")
    expected = {name: hashlib.sha256(b"trusted").hexdigest() for name in names}
    with pytest.raises(ValueError, match="resource missing"):
        EcapaEncoder.from_local_resources(tmp_path, expected)
    (tmp_path / "hyperparams.yaml").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        EcapaEncoder.from_local_resources(tmp_path, expected)
