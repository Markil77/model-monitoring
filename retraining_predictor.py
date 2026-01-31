"""Predict when model will need retraining based on metric degradation trends."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from config import MetricThresholds
from simulator import DailyMetrics


@dataclass
class MetricForecast:
    """Forecast result for a single metric."""

    metric_name: str
    current_value: float
    threshold: float
    slope: float  # Rate of change per day
    days_until_threshold: Optional[int]  # None if not degrading or already below
    r_squared: float  # Model fit quality
    trend: str  # "degrading", "improving", "stable"


@dataclass
class RetrainingPrediction:
    """Complete retraining prediction with all metric forecasts."""

    forecasts: List[MetricForecast]
    days_until_retraining: Optional[int]  # Earliest metric breach
    critical_metric: Optional[str]  # Which metric will breach first
    recommendation: str


def linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Perform simple linear regression.

    Args:
        x: Independent variable (days)
        y: Dependent variable (metric values)

    Returns:
        Tuple of (slope, intercept, r_squared)
    """
    n = len(x)
    if n < 2:
        return 0.0, y[0] if len(y) > 0 else 0.0, 0.0

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    if denominator == 0:
        return 0.0, y_mean, 0.0

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    # Calculate R-squared
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)

    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return slope, intercept, r_squared


def determine_trend(slope: float, threshold: float = 0.001) -> str:
    """Determine if metric is degrading, improving, or stable."""
    if abs(slope) < threshold:
        return "stable"
    elif slope < 0:
        return "degrading"
    else:
        return "improving"


def calculate_days_until_threshold(
    current_day: int,
    current_value: float,
    slope: float,
    threshold: float,
    metric_name: str,
    max_days: int = 365
) -> Optional[int]:
    """
    Calculate days until metric crosses threshold.

    For precision, recall, F1: threshold is a lower bound (bad if below)
    For FPR: threshold is an upper bound (bad if above)
    """
    if abs(slope) < 1e-6:
        return None  # No significant trend

    if metric_name == "fpr":
        # FPR: bad when it goes ABOVE threshold
        if current_value >= threshold:
            return 0  # Already breached
        if slope <= 0:
            return None  # Improving or stable, won't breach

        days = (threshold - current_value) / slope
    else:
        # Other metrics: bad when they go BELOW threshold
        if current_value <= threshold:
            return 0  # Already breached
        if slope >= 0:
            return None  # Improving or stable, won't breach

        days = (threshold - current_value) / slope

    if days < 0 or days > max_days:
        return None

    return int(np.ceil(days))


def forecast_metric(
    days: np.ndarray,
    values: np.ndarray,
    metric_name: str,
    threshold: float
) -> MetricForecast:
    """Generate forecast for a single metric."""
    slope, intercept, r_squared = linear_regression(days, values)

    current_value = values[-1]
    current_day = days[-1]

    # Determine trend direction
    # For FPR, positive slope is degrading; for others, negative slope is degrading
    if metric_name == "fpr":
        trend = "degrading" if slope > 0.001 else ("improving" if slope < -0.001 else "stable")
    else:
        trend = "degrading" if slope < -0.001 else ("improving" if slope > 0.001 else "stable")

    days_until = calculate_days_until_threshold(
        current_day, current_value, slope, threshold, metric_name
    )

    return MetricForecast(
        metric_name=metric_name,
        current_value=current_value,
        threshold=threshold,
        slope=slope,
        days_until_threshold=days_until,
        r_squared=r_squared,
        trend=trend
    )


def predict_retraining(
    daily_metrics: List[DailyMetrics],
    thresholds: MetricThresholds,
    lookback_days: Optional[int] = None
) -> RetrainingPrediction:
    """
    Predict when model will need retraining based on metric trends.

    Args:
        daily_metrics: List of daily metric observations
        thresholds: Configured metric thresholds
        lookback_days: Number of recent days to use for trend (None = all)

    Returns:
        RetrainingPrediction with forecasts and recommendation
    """
    if not daily_metrics:
        return RetrainingPrediction(
            forecasts=[],
            days_until_retraining=None,
            critical_metric=None,
            recommendation="Insufficient data for prediction. Need at least 1 day of metrics."
        )

    # Use lookback window if specified
    if lookback_days and len(daily_metrics) > lookback_days:
        metrics_subset = daily_metrics[-lookback_days:]
    else:
        metrics_subset = daily_metrics

    # Extract arrays
    days = np.array([m.day for m in metrics_subset])
    precision = np.array([m.precision for m in metrics_subset])
    recall = np.array([m.recall for m in metrics_subset])
    f1 = np.array([m.f1 for m in metrics_subset])
    fpr = np.array([m.fpr for m in metrics_subset])

    # Generate forecasts for each metric
    forecasts = [
        forecast_metric(days, precision, "precision", thresholds.precision),
        forecast_metric(days, recall, "recall", thresholds.recall),
        forecast_metric(days, f1, "f1", thresholds.f1),
        forecast_metric(days, fpr, "fpr", thresholds.fpr),
    ]

    # Find the metric that will breach first
    breach_times = [
        (f.metric_name, f.days_until_threshold)
        for f in forecasts
        if f.days_until_threshold is not None and f.days_until_threshold > 0
    ]

    if not breach_times:
        # Check if any already breached
        already_breached = [f for f in forecasts if f.days_until_threshold == 0]
        if already_breached:
            critical = already_breached[0].metric_name
            return RetrainingPrediction(
                forecasts=forecasts,
                days_until_retraining=0,
                critical_metric=critical,
                recommendation=f"URGENT: {critical} has already breached threshold. Retrain immediately."
            )

        # No degradation detected
        degrading = [f for f in forecasts if f.trend == "degrading"]
        if degrading:
            return RetrainingPrediction(
                forecasts=forecasts,
                days_until_retraining=None,
                critical_metric=None,
                recommendation="Metrics degrading but breach not imminent. Continue monitoring."
            )

        return RetrainingPrediction(
            forecasts=forecasts,
            days_until_retraining=None,
            critical_metric=None,
            recommendation="Model performance stable. No retraining needed at this time."
        )

    # Find earliest breach
    critical_metric, days_until = min(breach_times, key=lambda x: x[1])

    # Generate recommendation
    if days_until <= 7:
        urgency = "URGENT"
        action = "Schedule retraining within the next week."
    elif days_until <= 14:
        urgency = "WARNING"
        action = "Plan retraining within two weeks."
    elif days_until <= 30:
        urgency = "NOTICE"
        action = "Consider scheduling retraining within the month."
    else:
        urgency = "INFO"
        action = "Monitor trends. Retraining not immediately required."

    recommendation = (
        f"{urgency}: {critical_metric} projected to breach threshold in {days_until} days. "
        f"{action}"
    )

    return RetrainingPrediction(
        forecasts=forecasts,
        days_until_retraining=days_until,
        critical_metric=critical_metric,
        recommendation=recommendation
    )


def format_prediction_report(prediction: RetrainingPrediction) -> str:
    """Format prediction as a readable report."""
    lines = [
        "Retraining Prediction Report",
        "=" * 40,
        ""
    ]

    # Forecasts table
    lines.append(f"{'Metric':<12} {'Current':>8} {'Threshold':>10} {'Trend':>10} {'Days to Breach':>15}")
    lines.append("-" * 60)

    for f in prediction.forecasts:
        days_str = str(f.days_until_threshold) if f.days_until_threshold is not None else "N/A"
        lines.append(
            f"{f.metric_name:<12} {f.current_value:>8.3f} {f.threshold:>10.3f} "
            f"{f.trend:>10} {days_str:>15}"
        )

    lines.append("")
    lines.append("-" * 60)

    if prediction.days_until_retraining is not None:
        lines.append(f"Estimated days until retraining needed: {prediction.days_until_retraining}")
        lines.append(f"Critical metric: {prediction.critical_metric}")
    else:
        lines.append("No threshold breach predicted in forecast window.")

    lines.append("")
    lines.append(f"Recommendation: {prediction.recommendation}")

    return "\n".join(lines)
