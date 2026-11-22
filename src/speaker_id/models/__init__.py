"""Fixed classical speaker identification baseline."""

from .mfcc_svm import decision_scores, make_model, mean_recording_scores

__all__ = ["decision_scores", "make_model", "mean_recording_scores"]
