"""Representation-agnostic open-set decisions and validation threshold calibration.

Scores are raw SVM decision-function scores, not probabilities. False accepts,
false rejects, and known-to-known misidentifications are distinct outcomes:
UNKNOWN→known is a false accept, known→UNKNOWN is a false reject, and a known
speaker predicted as another known speaker is a misidentification. The latter
does not enter the FAR/FRR calibration objective.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

UNKNOWN_LABEL = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class OpenSetDecision:
    """One recording-level open-set classification result."""

    prediction: str
    top_class: str
    top_score: float
    threshold: float
    accepted: bool


@dataclass(frozen=True, slots=True)
class ThresholdCandidate:
    """FAR, FRR, and objective at one candidate threshold."""

    threshold: float
    far: float
    frr: float
    objective: float


@dataclass(frozen=True, slots=True)
class ThresholdCalibration:
    """Selected validation threshold and its candidate-curve audit."""

    threshold: float
    far: float
    frr: float
    objective: float
    known_count: int
    unknown_count: int
    candidates: tuple[ThresholdCandidate, ...]

    @property
    def candidate_count(self) -> int:
        """Number of distinct threshold candidates evaluated."""

        return len(self.candidates)


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _finite_score_list(values: Iterable[float], name: str) -> list[float]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a non-empty sequence of scores")
    try:
        scores = [_finite_float(value, f"{name} score") for value in values]
    except TypeError as exc:
        raise ValueError(f"{name} must be a non-empty sequence of scores") from exc
    if not scores:
        raise ValueError(f"{name} must contain at least one score")
    return scores


def _validate_classes_and_scores(
    class_scores: Sequence[float], classes: Sequence[str]
) -> tuple[list[float], list[str]]:
    if isinstance(classes, (str, bytes)):
        raise ValueError("classes must be a non-empty sequence of known labels")
    class_list = list(classes)
    if not class_list:
        raise ValueError("classes must not be empty")
    if any(not isinstance(label, str) or not label for label in class_list):
        raise ValueError("every class must be a non-empty string")
    if len(set(class_list)) != len(class_list):
        raise ValueError("classes must not contain duplicates")
    if UNKNOWN_LABEL in class_list:
        raise ValueError(f"{UNKNOWN_LABEL} is an output label, not a trained class")
    if isinstance(class_scores, (str, bytes)):
        raise ValueError("class_scores must be a non-empty sequence")
    try:
        scores = [_finite_float(value, "class score") for value in class_scores]
    except TypeError as exc:
        raise ValueError("class_scores must be a non-empty sequence") from exc
    if not scores:
        raise ValueError("class_scores must not be empty")
    if len(scores) != len(class_list):
        raise ValueError("the number of class scores must match the number of classes")
    return scores, class_list


def open_set_decision(
    class_scores: Sequence[float], classes: Sequence[str], threshold: float
) -> OpenSetDecision:
    """Accept the top known class at ``score >= threshold``; otherwise return UNKNOWN."""

    scores, class_list = _validate_classes_and_scores(class_scores, classes)
    cutoff = _finite_float(threshold, "threshold")
    top_index = max(range(len(scores)), key=scores.__getitem__)
    top_score = scores[top_index]
    top_class = class_list[top_index]
    accepted = top_score >= cutoff
    return OpenSetDecision(
        prediction=top_class if accepted else UNKNOWN_LABEL,
        top_class=top_class,
        top_score=top_score,
        threshold=cutoff,
        accepted=accepted,
    )


def candidate_thresholds(
    known_top_scores: Iterable[float], unknown_top_scores: Iterable[float]
) -> tuple[float, ...]:
    """Return every distinct score boundary plus a finite reject-all boundary."""

    known = _finite_score_list(known_top_scores, "known_top_scores")
    unknown = _finite_score_list(unknown_top_scores, "unknown_top_scores")
    observed = set(known + unknown)
    reject_all = math.nextafter(max(observed), math.inf)
    if not math.isfinite(reject_all):
        raise ValueError("cannot represent a finite reject-all threshold above the maximum score")
    return tuple(sorted((*observed, reject_all)))


def threshold_curve(
    known_top_scores: Iterable[float], unknown_top_scores: Iterable[float]
) -> tuple[ThresholdCandidate, ...]:
    """Evaluate recording-level FAR, FRR, and their equally weighted objective."""

    known = _finite_score_list(known_top_scores, "known_top_scores")
    unknown = _finite_score_list(unknown_top_scores, "unknown_top_scores")
    candidates = candidate_thresholds(known, unknown)
    curve = []
    for threshold in candidates:
        far = sum(score >= threshold for score in unknown) / len(unknown)
        frr = sum(score < threshold for score in known) / len(known)
        curve.append(
            ThresholdCandidate(
                threshold=threshold,
                far=far,
                frr=frr,
                objective=0.5 * far + 0.5 * frr,
            )
        )
    return tuple(curve)


def calibrate_threshold(
    known_top_scores: Iterable[float], unknown_top_scores: Iterable[float]
) -> ThresholdCalibration:
    """Select the minimum-objective validation threshold, breaking ties upward.

    Inputs are one top decision score per known or unknown validation recording;
    this function does not aggregate segment scores or know the representation.
    """

    known = _finite_score_list(known_top_scores, "known_top_scores")
    unknown = _finite_score_list(unknown_top_scores, "unknown_top_scores")
    curve = threshold_curve(known, unknown)
    selected = min(curve, key=lambda point: (point.objective, -point.threshold))
    return ThresholdCalibration(
        threshold=selected.threshold,
        far=selected.far,
        frr=selected.frr,
        objective=selected.objective,
        known_count=len(known),
        unknown_count=len(unknown),
        candidates=curve,
    )
