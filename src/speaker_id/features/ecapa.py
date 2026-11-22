"""Inference-only adapter for the pretrained ECAPA speaker encoder."""

import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .ecapa_constants import DEFAULT_ECAPA_SOURCE

CANONICAL_SAMPLE_RATE_HZ = 16_000


class EcapaEncoder:
    """Encode canonical mono 16 kHz waveforms using a pretrained ECAPA model."""

    def __init__(
        self,
        classifier: Any,
        *,
        source: str = DEFAULT_ECAPA_SOURCE,
        device: str | torch.device = "cpu",
    ) -> None:
        if not callable(getattr(classifier, "encode_batch", None)):
            raise ValueError("classifier must provide a callable encode_batch() method")
        self._classifier = classifier
        self._source = source
        self._device = torch.device(device)

    @classmethod
    def from_pretrained(
        cls,
        source: str = DEFAULT_ECAPA_SOURCE,
        *,
        device: str | torch.device = "cpu",
        savedir: str | Path | None = None,
    ) -> "EcapaEncoder":
        """Load the pretrained SpeechBrain classifier once for reuse."""

        from speechbrain.inference.speaker import EncoderClassifier

        if savedir is None:
            cache_name = (
                "spkrec-ecapa-voxceleb"
                if source == DEFAULT_ECAPA_SOURCE
                else source.replace("/", "--")
            )
            savedir_path = Path.home() / ".cache" / "speechbrain" / cache_name
        else:
            savedir_path = Path(savedir).expanduser()
        savedir_path = savedir_path.resolve()
        repository_root = Path(__file__).resolve().parents[3]
        if savedir_path.is_relative_to(repository_root):
            raise ValueError("ECAPA model cache must be outside the repository")

        options: dict[str, object] = {
            "source": source,
            "savedir": str(savedir_path),
            "run_opts": {"device": str(torch.device(device))},
            "freeze_params": True,
        }
        classifier = EncoderClassifier.from_hparams(**options)
        return cls(classifier, source=source, device=device)

    @property
    def source(self) -> str:
        """Pretrained model source associated with this encoder."""

        return self._source

    @classmethod
    def from_local_resources(
        cls, directory: Path, expected_sha256: dict[str, str]
    ) -> "EcapaEncoder":
        """Load fingerprinted local resources on CPU; never consult or update a remote model.

        Temporary copies keep SpeechBrain's collection step away from the original cache.
        Only file locations are overridden; architecture and weights are unchanged.
        """

        required = {
            "hyperparams.yaml", "embedding_model.ckpt", "classifier.ckpt",
            "mean_var_norm_emb.ckpt", "label_encoder.txt",
        }
        if set(expected_sha256) != required:
            raise ValueError("ECAPA resource manifest must describe exactly the required files")
        verified = {}
        for name, expected in expected_sha256.items():
            path = directory / name
            if name == "label_encoder.txt" and not path.exists():
                path = directory / "label_encoder.ckpt"
            if not path.is_file():
                raise ValueError(
                    f"Local ECAPA resource missing: {path}; see docs/PORTABLE_INFERENCE.md"
                )
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise ValueError(f"Local ECAPA resource SHA256 mismatch: {path}")
            verified[name] = path

        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import FetchConfig, LocalStrategy

        with tempfile.TemporaryDirectory(prefix="ecapa-offline-") as scratch:
            source = Path(scratch) / "source"
            source.mkdir()
            for name, path in verified.items():
                shutil.copyfile(path, source / name)
                with (source / name).open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != expected_sha256[name]:
                        raise ValueError(f"ECAPA resource changed while copying: {name}")
            classifier = EncoderClassifier.from_hparams(
                source=str(source), savedir=str(Path(scratch) / "loaded"),
                overrides={"pretrained_path": str(source)}, run_opts={"device": "cpu"},
                freeze_params=True, local_strategy=LocalStrategy.COPY,
                fetch_config=FetchConfig(allow_network=False),
            )
        return cls(classifier)

    @property
    def device(self) -> torch.device:
        """Device used for inference."""

        return self._device

    def encode(
        self,
        samples: np.ndarray,
        sample_rate: int = CANONICAL_SAMPLE_RATE_HZ,
    ) -> np.ndarray:
        """Return one finite, one-dimensional embedding for canonical audio."""

        if sample_rate != CANONICAL_SAMPLE_RATE_HZ:
            raise ValueError(
                f"sample_rate must be {CANONICAL_SAMPLE_RATE_HZ} Hz, got {sample_rate}"
            )
        if not isinstance(samples, np.ndarray):
            raise ValueError("samples must be a NumPy array")
        if samples.ndim != 1:
            raise ValueError(f"samples must be one-dimensional, got shape {samples.shape}")
        if samples.dtype != np.float32:
            raise ValueError(f"samples must have dtype float32, got {samples.dtype}")
        if samples.size == 0:
            raise ValueError("samples must not be empty")
        if not np.isfinite(samples).all():
            raise ValueError("samples must contain only finite values")

        waveform = torch.from_numpy(np.ascontiguousarray(samples)).unsqueeze(0)
        waveform = waveform.to(device=self._device, dtype=torch.float32)
        with torch.inference_mode():
            output = self._classifier.encode_batch(waveform)

        if not isinstance(output, torch.Tensor):
            raise ValueError("encode_batch() must return a torch.Tensor")
        if output.ndim < 2 or output.shape[0] != 1:
            raise ValueError(
                "encode_batch() must return a batched tensor with batch size 1; "
                f"got shape {tuple(output.shape)}"
            )
        if output.numel() == 0:
            raise ValueError("encode_batch() returned an empty embedding")
        if not output.is_floating_point():
            raise ValueError("encode_batch() must return a floating-point tensor")
        if not torch.isfinite(output).all().item():
            raise ValueError("encode_batch() returned non-finite embedding values")

        embedding = output.detach()[0].reshape(-1).to(device="cpu", dtype=torch.float32).numpy()
        if embedding.size == 0 or not np.isfinite(embedding).all():
            raise ValueError("encoder output is empty or non-finite after conversion")
        return embedding
