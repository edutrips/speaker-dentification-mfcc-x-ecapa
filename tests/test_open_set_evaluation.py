"""Synthetic recording-level open-set evaluation tests."""

import pytest

from speaker_id.open_set import (
    UNKNOWN_LABEL,
    evaluate_open_set,
    evaluation_label_for_speaker,
    open_set_decision,
)

KNOWN = ("V01", "V02", "V03", "V04")
CLASS_ORDER = (*KNOWN, UNKNOWN_LABEL)


def test_perfect_synthetic_case():
    labels = ["V01", "V02", "V03", "V04", UNKNOWN_LABEL, UNKNOWN_LABEL]
    result = evaluate_open_set(labels, labels, KNOWN)

    assert result.recording_count == 6
    assert result.known_count == 4
    assert result.unknown_count == 2
    assert result.known_identification_accuracy == 1.0
    assert result.unknown_rejection_rate == 1.0
    assert result.far == result.frr == 0.0
    assert result.known_to_known_misidentification_count == 0
    assert result.known_to_known_misidentification_rate == 0.0
    assert result.overall_accuracy == result.macro_f1 == 1.0
    assert result.class_order == CLASS_ORDER
    assert result.confusion_matrix == (
        (1, 0, 0, 0, 0),
        (0, 1, 0, 0, 0),
        (0, 0, 1, 0, 0),
        (0, 0, 0, 1, 0),
        (0, 0, 0, 0, 2),
    )


def test_mixed_synthetic_case_derives_all_metrics_and_matrix():
    truth = ["V01", "V02", "V03", UNKNOWN_LABEL, UNKNOWN_LABEL]
    prediction = ["V01", UNKNOWN_LABEL, "V04", UNKNOWN_LABEL, "V02"]
    result = evaluate_open_set(truth, prediction, KNOWN)

    assert result.known_identification_accuracy == pytest.approx(1 / 3)
    assert result.unknown_rejection_rate == pytest.approx(1 / 2)
    assert result.far == pytest.approx(1 / 2)
    assert result.frr == pytest.approx(1 / 3)
    assert result.known_to_known_misidentification_count == 1
    assert result.known_to_known_misidentification_rate == pytest.approx(1 / 3)
    assert result.overall_accuracy == pytest.approx(2 / 5)
    assert result.macro_f1 == pytest.approx(0.3)
    assert result.class_order == CLASS_ORDER
    assert result.confusion_matrix == (
        (1, 0, 0, 0, 0),
        (0, 0, 0, 0, 1),
        (0, 0, 0, 1, 0),
        (0, 0, 0, 0, 0),
        (0, 1, 0, 0, 1),
    )


def test_wrong_known_identity_is_misidentification_not_false_reject():
    result = evaluate_open_set(["V01", UNKNOWN_LABEL], ["V02", UNKNOWN_LABEL], KNOWN)
    assert result.known_to_known_misidentification_count == 1
    assert result.known_to_known_misidentification_rate == 1.0
    assert result.frr == 0.0
    assert result.known_identification_accuracy == 0.0


@pytest.mark.parametrize("predicted", ["V01", "V02", "V03", "V04"])
def test_unknown_predicted_as_any_known_identity_counts_as_false_accept(predicted):
    result = evaluate_open_set(["V01", UNKNOWN_LABEL], ["V01", predicted], KNOWN)
    assert result.far == 1.0
    assert result.unknown_rejection_rate == 0.0


def test_known_rejection_and_correct_unknown_rejection_counts():
    result = evaluate_open_set(
        ["V01", "V02", UNKNOWN_LABEL], [UNKNOWN_LABEL, "V02", UNKNOWN_LABEL], KNOWN
    )
    assert result.frr == pytest.approx(1 / 2)
    assert result.known_identification_accuracy == pytest.approx(1 / 2)
    assert result.unknown_rejection_rate == 1.0
    assert result.far == 0.0


@pytest.mark.parametrize(
    ("truth", "prediction", "known", "message"),
    [
        ([], [], KNOWN, "y_true must not be empty"),
        (["V01", UNKNOWN_LABEL], ["V01"], KNOWN, "equal lengths"),
        (["V01", UNKNOWN_LABEL], ["V01", UNKNOWN_LABEL], [], "known_classes must not be empty"),
        (
            ["V01", UNKNOWN_LABEL],
            ["V01", UNKNOWN_LABEL],
            ["V01"],
            "exactly the active-known",
        ),
        (["V01", UNKNOWN_LABEL], ["V01", UNKNOWN_LABEL], ["V01", "V01"], "duplicates"),
        (
            ["V01", UNKNOWN_LABEL],
            ["V01", UNKNOWN_LABEL],
            ["V01", UNKNOWN_LABEL],
            "output label",
        ),
        (["U01", UNKNOWN_LABEL], ["V01", UNKNOWN_LABEL], KNOWN, "y_true contains"),
        (["V01", UNKNOWN_LABEL], ["U01", UNKNOWN_LABEL], KNOWN, "y_pred contains"),
        (["V01", "V02"], ["V01", "V02"], KNOWN, "at least one UNKNOWN"),
        (
            [UNKNOWN_LABEL, UNKNOWN_LABEL],
            [UNKNOWN_LABEL, UNKNOWN_LABEL],
            KNOWN,
            "at least one known",
        ),
        (
            ["V01", UNKNOWN_LABEL],
            ["V01", UNKNOWN_LABEL],
            ["V01", "V05"],
            "optional supported",
        ),
    ],
)
def test_evaluator_rejects_invalid_inputs(truth, prediction, known, message):
    with pytest.raises(ValueError, match=message):
        evaluate_open_set(truth, prediction, known)


def test_unknown_speaker_ids_collapse_and_optional_v05_is_rejected():
    assert evaluation_label_for_speaker("V01") == "V01"
    assert evaluation_label_for_speaker("U01") == UNKNOWN_LABEL
    assert evaluation_label_for_speaker("U99") == UNKNOWN_LABEL
    assert evaluation_label_for_speaker(UNKNOWN_LABEL) == UNKNOWN_LABEL
    with pytest.raises(ValueError, match="optional"):
        evaluation_label_for_speaker("V05")
    with pytest.raises(ValueError, match="not eligible"):
        evaluation_label_for_speaker("U00")


def test_open04_decision_integrates_with_recording_level_evaluator():
    scores = ([3.1, 1.0, 0.2, 0.0], [1.0, 0.7, 0.2, 0.0])
    predictions = [
        open_set_decision(row, KNOWN, threshold=2.0).prediction for row in scores
    ]
    result = evaluate_open_set(["V01", UNKNOWN_LABEL], predictions, KNOWN)
    assert predictions == ["V01", UNKNOWN_LABEL]
    assert result.overall_accuracy == 1.0
    assert result.far == result.frr == 0.0


def test_evaluator_is_representation_agnostic_and_keeps_given_class_order():
    truth = ["V01", UNKNOWN_LABEL]
    predictions = ["V01", UNKNOWN_LABEL]
    mfcc_metrics = evaluate_open_set(truth, predictions, KNOWN)
    ecapa_metrics = evaluate_open_set(truth, predictions, KNOWN)
    reverse_known = tuple(reversed(KNOWN))
    reordered = evaluate_open_set(truth, predictions, reverse_known)
    assert mfcc_metrics == ecapa_metrics
    assert reordered.class_order == (*reverse_known, UNKNOWN_LABEL)
    assert len(reordered.confusion_matrix) == 5
    assert all(len(row) == 5 for row in reordered.confusion_matrix)
