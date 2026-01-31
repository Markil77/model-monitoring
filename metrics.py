"""Metrics calculation functions for model monitoring."""

from dataclasses import dataclass
from typing import Union

import numpy as np


@dataclass
class MetricsResult:
    """Container for all calculated metrics."""

    precision: float
    recall: float
    f1: float
    fpr: float  # False positive rate
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int


def calculate_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate precision: TP / (TP + FP).

    Returns 0.0 if there are no predicted positives.
    """
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))

    denominator = true_positives + false_positives
    if denominator == 0:
        return 0.0

    return float(true_positives / denominator)


def calculate_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate recall (sensitivity): TP / (TP + FN).

    Returns 0.0 if there are no actual positives.
    """
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))

    denominator = true_positives + false_negatives
    if denominator == 0:
        return 0.0

    return float(true_positives / denominator)


def calculate_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate F1 score: 2 * (precision * recall) / (precision + recall).

    Returns 0.0 if both precision and recall are 0.
    """
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)

    denominator = precision + recall
    if denominator == 0:
        return 0.0

    return float(2 * (precision * recall) / denominator)


def calculate_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate false positive rate: FP / (FP + TN).

    Returns 0.0 if there are no actual negatives.
    """
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    true_negatives = np.sum((y_true == 0) & (y_pred == 0))

    denominator = false_positives + true_negatives
    if denominator == 0:
        return 0.0

    return float(false_positives / denominator)


def calculate_all_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list]
) -> MetricsResult:
    """
    Calculate all metrics and return as a MetricsResult dataclass.

    Args:
        y_true: Ground truth binary labels (0 or 1)
        y_pred: Predicted binary labels (0 or 1)

    Returns:
        MetricsResult containing all metrics and confusion matrix values
    """
    # Convert to numpy arrays if needed
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    # Calculate confusion matrix components
    true_positives = int(np.sum((y_true == 1) & (y_pred == 1)))
    false_positives = int(np.sum((y_true == 0) & (y_pred == 1)))
    true_negatives = int(np.sum((y_true == 0) & (y_pred == 0)))
    false_negatives = int(np.sum((y_true == 1) & (y_pred == 0)))

    # Calculate metrics
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)
    f1 = calculate_f1(y_true, y_pred)
    fpr = calculate_fpr(y_true, y_pred)

    return MetricsResult(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        fpr=round(fpr, 4),
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives
    )
