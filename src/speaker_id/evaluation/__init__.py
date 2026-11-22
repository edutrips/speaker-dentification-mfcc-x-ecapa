"""Leakage-safe exploratory evaluation helpers."""

from .metrics import classification_metrics
from .splits import TextFold, text_folds

__all__ = ["TextFold", "classification_metrics", "text_folds"]
