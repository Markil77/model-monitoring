"""Simulate model predictions with degrading performance over time."""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from config import MonitoringConfig
from metrics import MetricsResult, calculate_all_metrics


@dataclass
class DailyMetrics:
    """Metrics for a single day."""

    day: int
    precision: float
    recall: float
    f1: float
    fpr: float
    total_transactions: int
    fraud_count: int
    predicted_fraud: int


def get_degradation_parameters(day: int) -> Tuple[float, float, float]:
    """
    Get degradation parameters based on the day.

    With ~2% fraud rate (~7 fraud cases per day out of ~333 transactions),
    we need very low FPR to achieve high precision. For example:
    - 85% precision with 7 TP requires < 1.2 FP, so FPR < 0.4%

    Returns:
        Tuple of (true_positive_rate, false_positive_rate_boost, noise_factor)
    """
    if day <= 10:
        # Days 1-10: Strong performance
        # Model accurately identifies fraud with very few false positives
        tp_rate = 0.95  # High true positive rate
        fp_boost = 0.003  # Very low false positive rate (~1 FP per day)
        noise = 0.01

    elif day <= 20:
        # Days 11-20: Gradual degradation
        # Increasing false positives (model becomes more aggressive)
        progress = (day - 10) / 10  # 0 to 1 over this period
        tp_rate = 0.95 - (0.15 * progress)  # Drops from 0.95 to 0.80
        fp_boost = 0.003 + (0.030 * progress)  # Rises from 0.3% to 3.3%
        noise = 0.02

    else:
        # Days 21-30: Significant drift
        # Missing more fraud cases, many more false positives
        progress = (day - 20) / 10  # 0 to 1 over this period
        tp_rate = 0.80 - (0.30 * progress)  # Drops from 0.80 to 0.50
        fp_boost = 0.033 + (0.067 * progress)  # Rises from 3.3% to 10%
        noise = 0.03

    return tp_rate, fp_boost, noise


def simulate_predictions(
    y_true: np.ndarray,
    day: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Simulate model predictions for a given day with appropriate degradation.

    Args:
        y_true: Ground truth labels
        day: Day number (1-30)
        rng: Random number generator

    Returns:
        Predicted labels array
    """
    tp_rate, fp_boost, noise = get_degradation_parameters(day)

    y_pred = np.zeros_like(y_true)

    # For actual fraud cases
    fraud_mask = y_true == 1
    fraud_probs = tp_rate + rng.uniform(-noise, noise, size=np.sum(fraud_mask))
    fraud_probs = np.clip(fraud_probs, 0, 1)
    y_pred[fraud_mask] = (rng.random(np.sum(fraud_mask)) < fraud_probs).astype(int)

    # For legitimate cases (false positives)
    legit_mask = y_true == 0
    fp_probs = fp_boost + rng.uniform(-noise/2, noise/2, size=np.sum(legit_mask))
    fp_probs = np.clip(fp_probs, 0, 1)
    y_pred[legit_mask] = (rng.random(np.sum(legit_mask)) < fp_probs).astype(int)

    return y_pred


def run_simulation(
    df: pd.DataFrame,
    config: MonitoringConfig
) -> Tuple[pd.DataFrame, List[DailyMetrics]]:
    """
    Run the 30-day simulation of model performance.

    Args:
        df: Transaction data with 'is_fraud' and 'day' columns
        config: Monitoring configuration

    Returns:
        Tuple of (DataFrame with predictions, list of daily metrics)
    """
    rng = np.random.default_rng(config.random_seed + 1000)

    all_predictions = []
    daily_metrics_list: List[DailyMetrics] = []

    for day in range(1, config.num_days + 1):
        day_data = df[df["day"] == day].copy()

        if len(day_data) == 0:
            continue

        y_true = day_data["is_fraud"].values
        y_pred = simulate_predictions(y_true, day, rng)

        day_data["predicted_fraud"] = y_pred
        all_predictions.append(day_data)

        # Calculate metrics for this day
        metrics = calculate_all_metrics(y_true, y_pred)

        daily_metrics = DailyMetrics(
            day=day,
            precision=metrics.precision,
            recall=metrics.recall,
            f1=metrics.f1,
            fpr=metrics.fpr,
            total_transactions=len(day_data),
            fraud_count=int(y_true.sum()),
            predicted_fraud=int(y_pred.sum())
        )
        daily_metrics_list.append(daily_metrics)

    result_df = pd.concat(all_predictions, ignore_index=True)

    return result_df, daily_metrics_list


def metrics_to_dataframe(daily_metrics: List[DailyMetrics]) -> pd.DataFrame:
    """Convert list of DailyMetrics to a pandas DataFrame."""
    data = [
        {
            "day": m.day,
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1,
            "fpr": m.fpr,
            "total_transactions": m.total_transactions,
            "fraud_count": m.fraud_count,
            "predicted_fraud": m.predicted_fraud
        }
        for m in daily_metrics
    ]
    return pd.DataFrame(data)
