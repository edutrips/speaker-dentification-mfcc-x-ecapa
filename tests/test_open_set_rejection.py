"""Synthetic tests for recording-level rejection and threshold calibration."""

import math

import numpy as np
import pytest

from speaker_id.models import mean_recording_scores
from speaker_id.open_set import (
    UNKNOWN_LABEL,
    calibrate_threshold,
    candidate_thresholds,
    open_set_decision,
    threshold_curve,
)

CLASSES = ("V01", "V02", "V03", "V04")


def test_accepts_top_known_class_above_threshold():
    decision = open_set_decision([3.0, 1.0, 0.2, -0.1], CLASSES, 2.5)
    assert decision.prediction == "V01"
    assert decision.top_class == "V01"
    assert decision.top_score == 3.0
    assert decision.threshold == 2.5
    assert decision.accepted is True


def test_rejects_top_score_below_threshold_but_reports_top_class():
    decision = open_set_decision([2.0, 1.9, 1.8, 1.7], CLASSES, 2.5)
    assert decision.prediction == UNKNOWN_LABEL
    assert decision.top_class == "V01"
    assert decision.top_score == 2.0
    assert decision.accepted is False


def test_score_equal_to_threshold_is_accepted():
    assert open_set_decision([2.5, 1.0, 0.0, -1.0], CLASSES, 2.5).accepted


@pytest.mark.parametrize(
    ("scores", "classes", "threshold", "message"),
    [
        ([1.0], [], 0.0, "classes must not be empty"),
        ([], ["V01"], 0.0, "class_scores must not be empty"),
        ([1.0], ["V01", "V02"], 0.0, "match"),
        ([1.0, 2.0], ["V01", "V01"], 0.0, "duplicates"),
        ([1.0, 2.0], ["V01", UNKNOWN_LABEL], 0.0, "output label"),
        ([float("nan")], ["V01"], 0.0, "finite"),
        ([float("inf")], ["V01"], 0.0, "finite"),
        ([float("-inf")], ["V01"], 0.0, "finite"),
        ([1.0], ["V01"], float("nan"), "finite"),
        ([1.0], ["V01"], float("inf"), "finite"),
        ([1.0], ["V01"], float("-inf"), "finite"),
    ],
)
def test_open_set_decision_rejects_invalid_inputs(scores, classes, threshold, message):
    with pytest.raises(ValueError, match=message):
        open_set_decision(scores, classes, threshold)


def test_candidate_thresholds_are_sorted_unique_and_include_reject_all():
    candidates = candidate_thresholds([3.0, 2.0], [2.0, 1.0])
    assert candidates == (1.0, 2.0, 3.0, math.nextafter(3.0, math.inf))
    assert all(left < right for left, right in zip(candidates, candidates[1:]))
    assert all(
        not open_set_decision([score], ["V01"], candidates[-1]).accepted
        for score in (1.0, 2.0, 3.0)
    )


def test_calibration_curve_calculates_far_frr_and_objective():
    result = calibrate_threshold([2.0, 1.0], [2.0, 0.0])
    point = next(candidate for candidate in result.candidates if candidate.threshold == 2.0)
    assert point.far == pytest.approx(0.5)
    assert point.frr == pytest.approx(0.5)
    assert point.objective == pytest.approx(0.5)
    assert result.known_count == 2
    assert result.unknown_count == 2


def test_highest_threshold_breaks_exact_objective_ties():
    result = calibrate_threshold([1.0, 2.0], [2.0, 1.0])
    tied = [point for point in result.candidates if point.objective == result.objective]
    assert len(tied) > 1
    assert result.threshold == max(point.threshold for point in tied)


def test_perfect_separation_selects_a_zero_error_boundary_deterministically():
    known = [3.0, 2.8, 2.7, 2.5]
    unknown = [1.1, 1.3, 1.8]
    first = calibrate_threshold(known, unknown)
    second = calibrate_threshold(known, unknown)
    assert first == second
    assert first.threshold == 2.5
    assert first.far == first.frr == first.objective == 0.0


def test_overlapping_scores_have_correct_rates_and_nonzero_best_error():
    result = calibrate_threshold([3.0, 2.1, 1.4], [2.3, 1.8, 1.0])
    selected = next(point for point in result.candidates if point.threshold == result.threshold)
    assert selected.far == pytest.approx(0.0)
    assert selected.frr == pytest.approx(2 / 3)
    assert selected.objective == pytest.approx(1 / 3)
    assert result.objective > 0


def test_equal_scores_one_record_each_and_empty_inputs():
    result = calibrate_threshold([1.0], [1.0])
    assert result.known_count == result.unknown_count == 1
    assert result.far == 0.0
    assert result.frr == 1.0
    assert result.threshold == math.nextafter(1.0, math.inf)
    with pytest.raises(ValueError, match="known_top_scores"):
        calibrate_threshold([], [1.0])
    with pytest.raises(ValueError, match="unknown_top_scores"):
        calibrate_threshold([1.0], [])


def test_reject_all_boundary_must_remain_finite():
    largest_float = float.fromhex("0x1.fffffffffffffp+1023")
    with pytest.raises(ValueError, match="finite reject-all"):
        candidate_thresholds([largest_float], [0.0])


@pytest.mark.parametrize("bad_score", [float("nan"), float("inf"), float("-inf")])
def test_calibration_rejects_non_finite_recording_scores(bad_score):
    with pytest.raises(ValueError, match="finite"):
        calibrate_threshold([1.0], [bad_score])


def test_recording_aggregation_helper_feeds_open_set_decision_without_reaggregation():
    class FixedDecisionModel:
        classes_ = np.asarray(CLASSES)

        @staticmethod
        def decision_function(x):
            return x

    segment_scores = np.asarray(
        [[3.2, 1.0, 0.2, -0.1], [2.8, 1.2, 0.0, -0.3]]
    )
    top_class, recording_scores = mean_recording_scores(FixedDecisionModel(), segment_scores)
    decision = open_set_decision(
        [recording_scores[label] for label in CLASSES], CLASSES, 2.5
    )
    assert top_class == decision.top_class == decision.prediction == "V01"
    assert decision.top_score == pytest.approx(3.0)


def test_calibration_is_independent_of_representation_name():
    known = [3.0, 2.1, 1.4]
    unknown = [2.3, 1.8, 1.0]
    mfcc_result = calibrate_threshold(known, unknown)
    ecapa_result = calibrate_threshold(known, unknown)
    assert mfcc_result == ecapa_result
    assert not hasattr(mfcc_result, "representation")


def test_threshold_curve_contains_each_partition_boundary():
    curve = threshold_curve([2.0, 1.0], [3.0, 1.0])
    assert tuple(point.threshold for point in curve) == candidate_thresholds(
        [2.0, 1.0], [3.0, 1.0]
    )
