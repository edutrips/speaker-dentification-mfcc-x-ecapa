"""Install exact frozen model bytes from trusted local files, without deserialization."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def install_models(
    source_root: Path, destination_root: Path, *, ecapa_source: Path | None = None
) -> list[dict[str, str]]:
    """Preflight all inputs before copying; refuse to overwrite a different artifact."""

    freeze = json.loads((ROOT / "docs/FINAL_OPEN_SET_A_FREEZE.json").read_text(encoding="utf-8"))
    files = []
    for name in ("mfcc", "ecapa"):
        relative = Path(freeze["artifacts_root"]) / name / "model/model.joblib"
        files.append((source_root / relative, destination_root / relative,
                      freeze["representations"][name]["model_sha256"]))
    if ecapa_source is not None:
        resources = json.loads((ROOT / "configs/ecapa_resources.json").read_text(encoding="utf-8"))
        for name, expected in resources["sha256"].items():
            source = ecapa_source / name
            if name == "label_encoder.txt" and not source.exists():
                source = ecapa_source / "label_encoder.ckpt"
            files.append((source, destination_root / "artifacts/ecapa_encoder" / name, expected))
    for source, destination, expected in files:
        if not source.is_file():
            raise ValueError(f"Trusted source file missing: {source}")
        if sha256(source) != expected:
            raise ValueError(f"Source SHA256 mismatch: {source}")
        if destination.exists() and (not destination.is_file() or sha256(destination) != expected):
            raise ValueError(f"Refusing to overwrite a different destination: {destination}")

    report = []
    for source, destination, expected in files:
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
                staged = Path(temporary.name)
            try:
                shutil.copyfile(source, staged)
                if sha256(staged) != expected:
                    raise ValueError(f"Copied SHA256 mismatch: {destination}")
                # Publish a complete file atomically; never replace a concurrent destination.
                os.link(staged, destination)
            finally:
                staged.unlink(missing_ok=True)
        if sha256(destination) != expected:
            raise ValueError(f"Installed SHA256 mismatch: {destination}")
        report.append({"destination": str(destination), "sha256": expected, "verification": "PASS"})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="trusted local tree containing artifacts/final_open_set_a")
    parser.add_argument("--destination-root", type=Path, default=ROOT)
    parser.add_argument("--ecapa-source", type=Path,
                        help="optional trusted local encoder cache/snapshot; never downloads")
    args = parser.parse_args()
    try:
        report = install_models(
            args.source_root, args.destination_root, ecapa_source=args.ecapa_source
        )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(json.dumps(report, indent=2))
    print("Model bytes verified; load compatibility must be checked by the inference CLI.")


if __name__ == "__main__":
    main()
