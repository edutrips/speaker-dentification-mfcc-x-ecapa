"""Train the frozen MFCC and ECAPA open-set known-speaker classifiers."""

import json
from pathlib import Path

from speaker_id.training.open_set_models import train_open_set_models

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    result = train_open_set_models(
        ROOT / "artifacts/open_set/mfcc/training",
        ROOT / "artifacts/open_set/ecapa/training",
        ROOT / "artifacts/open_set/training_protocol.json",
        ROOT / "artifacts/open_set",
    )
    printable = {
        "training_rows": result["summary"]["rows"],
        "training_recordings": result["summary"]["recordings"],
        "class_distribution": result["summary"]["class_counts"],
        "mfcc": {key: value for key, value in result["mfcc"].items() if key != "config"},
        "ecapa": {key: value for key, value in result["ecapa"].items() if key != "config"},
        "no_threshold_calibrated": result["no_threshold_calibrated"],
        "no_open_set_evaluation": result["no_open_set_evaluation"],
    }
    print(json.dumps(printable, indent=2, default=str))


if __name__ == "__main__":
    main()
