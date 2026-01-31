"""Alert generation and management for model monitoring."""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List

from config import MonitoringConfig
from simulator import DailyMetrics


class Severity(str, Enum):
    """Alert severity levels."""

    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Represents a single metric alert."""

    day: int
    metric_name: str
    value: float
    threshold: float
    severity: Severity
    message: str
    timestamp: str = ""

    def __post_init__(self) -> None:
        """Set timestamp if not provided."""
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


def determine_severity(metric_name: str, value: float, threshold: float) -> Severity:
    """
    Determine alert severity based on how far the metric is from threshold.

    For precision, recall, F1: lower is worse
    For FPR: higher is worse
    """
    if metric_name == "fpr":
        # FPR should be below threshold
        deviation = value - threshold
        relative_deviation = deviation / threshold if threshold > 0 else deviation
    else:
        # Other metrics should be above threshold
        deviation = threshold - value
        relative_deviation = deviation / threshold if threshold > 0 else deviation

    # Critical if more than 15% deviation from threshold
    if relative_deviation > 0.15:
        return Severity.CRITICAL

    return Severity.WARNING


def create_alert_message(metric_name: str, value: float, threshold: float) -> str:
    """Create a human-readable alert message."""
    metric_display = {
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 Score",
        "fpr": "False Positive Rate"
    }

    display_name = metric_display.get(metric_name, metric_name)

    if metric_name == "fpr":
        return (
            f"{display_name} ({value:.2%}) exceeds threshold ({threshold:.2%}). "
            f"Model is flagging too many legitimate transactions as fraud."
        )
    else:
        return (
            f"{display_name} ({value:.2%}) is below threshold ({threshold:.2%}). "
            f"Model performance has degraded and requires investigation."
        )


def check_thresholds(
    daily_metrics: DailyMetrics,
    config: MonitoringConfig
) -> List[Alert]:
    """
    Check a day's metrics against configured thresholds.

    Args:
        daily_metrics: Metrics for a single day
        config: Monitoring configuration with thresholds

    Returns:
        List of Alert objects for any threshold violations
    """
    alerts: List[Alert] = []
    thresholds = config.thresholds

    # Check precision
    if daily_metrics.precision < thresholds.precision:
        alerts.append(Alert(
            day=daily_metrics.day,
            metric_name="precision",
            value=daily_metrics.precision,
            threshold=thresholds.precision,
            severity=determine_severity(
                "precision", daily_metrics.precision, thresholds.precision
            ),
            message=create_alert_message(
                "precision", daily_metrics.precision, thresholds.precision
            )
        ))

    # Check recall
    if daily_metrics.recall < thresholds.recall:
        alerts.append(Alert(
            day=daily_metrics.day,
            metric_name="recall",
            value=daily_metrics.recall,
            threshold=thresholds.recall,
            severity=determine_severity(
                "recall", daily_metrics.recall, thresholds.recall
            ),
            message=create_alert_message(
                "recall", daily_metrics.recall, thresholds.recall
            )
        ))

    # Check F1
    if daily_metrics.f1 < thresholds.f1:
        alerts.append(Alert(
            day=daily_metrics.day,
            metric_name="f1",
            value=daily_metrics.f1,
            threshold=thresholds.f1,
            severity=determine_severity(
                "f1", daily_metrics.f1, thresholds.f1
            ),
            message=create_alert_message(
                "f1", daily_metrics.f1, thresholds.f1
            )
        ))

    # Check FPR (should be below threshold, not above)
    if daily_metrics.fpr > thresholds.fpr:
        alerts.append(Alert(
            day=daily_metrics.day,
            metric_name="fpr",
            value=daily_metrics.fpr,
            threshold=thresholds.fpr,
            severity=determine_severity(
                "fpr", daily_metrics.fpr, thresholds.fpr
            ),
            message=create_alert_message(
                "fpr", daily_metrics.fpr, thresholds.fpr
            )
        ))

    return alerts


def check_all_days(
    daily_metrics_list: List[DailyMetrics],
    config: MonitoringConfig
) -> List[Alert]:
    """Check all days for threshold violations."""
    all_alerts: List[Alert] = []

    for daily_metrics in daily_metrics_list:
        day_alerts = check_thresholds(daily_metrics, config)
        all_alerts.extend(day_alerts)

    return all_alerts


def save_alerts(alerts: List[Alert], filepath: Path) -> None:
    """
    Save alerts to a JSON file.

    Args:
        alerts: List of Alert objects to save
        filepath: Path to the output JSON file

    Raises:
        IOError: If the file cannot be written
    """
    try:
        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert alerts to dictionaries
        alerts_data = []
        for alert in alerts:
            alert_dict = asdict(alert)
            # Convert Severity enum to string
            alert_dict["severity"] = alert.severity.value
            alerts_data.append(alert_dict)

        # Write JSON with pretty formatting
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "generated_at": datetime.now().isoformat(),
                    "total_alerts": len(alerts),
                    "alerts": alerts_data
                },
                f,
                indent=2
            )

    except OSError as e:
        raise IOError(f"Failed to save alerts to {filepath}: {e}") from e


def get_alerts_summary(alerts: List[Alert]) -> dict:
    """Get a summary of alerts by severity and metric."""
    summary = {
        "total": len(alerts),
        "by_severity": {"warning": 0, "critical": 0},
        "by_metric": {"precision": 0, "recall": 0, "f1": 0, "fpr": 0},
        "affected_days": set()
    }

    for alert in alerts:
        summary["by_severity"][alert.severity.value] += 1
        summary["by_metric"][alert.metric_name] += 1
        summary["affected_days"].add(alert.day)

    summary["affected_days"] = sorted(summary["affected_days"])

    return summary
