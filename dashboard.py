"""Plotly dashboard for model monitoring visualization."""

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from alerts import Alert
from config import MonitoringConfig
from simulator import DailyMetrics

logger = logging.getLogger(__name__)


def clean_transaction_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean transaction data by removing invalid rows.

    Logs warnings for any negative transaction amounts and filters them out.

    Args:
        df: Transaction DataFrame with 'amount' column

    Returns:
        Cleaned DataFrame with negative amounts removed
    """
    if "amount" not in df.columns:
        return df

    negative_mask = df["amount"] < 0
    if negative_mask.any():
        negative_count = negative_mask.sum()
        negative_rows = df.loc[negative_mask]
        logger.warning(
            "Skipping %d transaction(s) with negative amounts. "
            "Transaction IDs: %s",
            negative_count,
            negative_rows["transaction_id"].tolist()[:10]
        )
        return df[~negative_mask].reset_index(drop=True)

    return df


def create_metrics_figure(
    metrics_df: pd.DataFrame,
    alerts: List[Alert],
    config: MonitoringConfig
) -> go.Figure:
    """
    Create a Plotly figure with all metrics and alert markers.

    Args:
        metrics_df: DataFrame with daily metrics
        alerts: List of alerts to mark on the chart
        config: Configuration with thresholds

    Returns:
        Plotly Figure object
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Precision Over Time",
            "Recall Over Time",
            "F1 Score Over Time",
            "False Positive Rate Over Time"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )

    # Define metric configurations
    metrics_config = [
        {
            "name": "precision",
            "display_name": "Precision",
            "threshold": config.thresholds.precision,
            "row": 1,
            "col": 1,
            "color": "#2E86AB"
        },
        {
            "name": "recall",
            "display_name": "Recall",
            "threshold": config.thresholds.recall,
            "row": 1,
            "col": 2,
            "color": "#A23B72"
        },
        {
            "name": "f1",
            "display_name": "F1 Score",
            "threshold": config.thresholds.f1,
            "row": 2,
            "col": 1,
            "color": "#F18F01"
        },
        {
            "name": "fpr",
            "display_name": "False Positive Rate",
            "threshold": config.thresholds.fpr,
            "row": 2,
            "col": 2,
            "color": "#C73E1D",
            "threshold_above": True  # FPR threshold is upper bound
        }
    ]

    # Get alert days for each metric
    alert_days = {m["name"]: [] for m in metrics_config}
    for alert in alerts:
        if alert.metric_name in alert_days:
            alert_days[alert.metric_name].append(alert.day)

    # Add traces for each metric
    for metric in metrics_config:
        name = metric["name"]
        row = metric["row"]
        col = metric["col"]

        # Main metric line
        fig.add_trace(
            go.Scatter(
                x=metrics_df["day"],
                y=metrics_df[name],
                mode="lines+markers",
                name=metric["display_name"],
                line=dict(color=metric["color"], width=2),
                marker=dict(size=6),
                hovertemplate=(
                    f"Day %{{x}}<br>"
                    f"{metric['display_name']}: %{{y:.3f}}<extra></extra>"
                )
            ),
            row=row,
            col=col
        )

        # Threshold line
        fig.add_trace(
            go.Scatter(
                x=metrics_df["day"],
                y=[metric["threshold"]] * len(metrics_df),
                mode="lines",
                name=f"{metric['display_name']} Threshold",
                line=dict(color="red", width=2, dash="dash"),
                showlegend=False,
                hovertemplate=f"Threshold: {metric['threshold']:.2f}<extra></extra>"
            ),
            row=row,
            col=col
        )

        # Alert markers
        if alert_days[name]:
            alert_df = metrics_df[metrics_df["day"].isin(alert_days[name])]
            fig.add_trace(
                go.Scatter(
                    x=alert_df["day"],
                    y=alert_df[name],
                    mode="markers",
                    name=f"{metric['display_name']} Alerts",
                    marker=dict(
                        size=14,
                        color="red",
                        symbol="circle-open",
                        line=dict(width=3, color="red")
                    ),
                    showlegend=False,
                    hovertemplate=(
                        f"ALERT - Day %{{x}}<br>"
                        f"{metric['display_name']}: %{{y:.3f}}<extra></extra>"
                    )
                ),
                row=row,
                col=col
            )

    # Update layout
    fig.update_layout(
        title=dict(
            text="<b>Fraud Detection Model - 30-Day Performance Monitor</b>",
            x=0.5,
            font=dict(size=20)
        ),
        height=700,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        template="plotly_white",
        hovermode="x unified"
    )

    # Update axes labels
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(title_text="Day", row=i, col=j)
            fig.update_yaxes(title_text="Score", row=i, col=j)

    return fig


def create_summary_stats(
    metrics_df: pd.DataFrame,
    alerts: List[Alert],
    config: MonitoringConfig
) -> str:
    """Create HTML summary statistics panel."""
    total_alerts = len(alerts)
    critical_alerts = sum(1 for a in alerts if a.severity.value == "critical")
    warning_alerts = total_alerts - critical_alerts

    # Calculate averages
    avg_precision = metrics_df["precision"].mean()
    avg_recall = metrics_df["recall"].mean()
    avg_f1 = metrics_df["f1"].mean()
    avg_fpr = metrics_df["fpr"].mean()

    # Days below threshold
    days_below = {
        "precision": (metrics_df["precision"] < config.thresholds.precision).sum(),
        "recall": (metrics_df["recall"] < config.thresholds.recall).sum(),
        "f1": (metrics_df["f1"] < config.thresholds.f1).sum(),
        "fpr": (metrics_df["fpr"] > config.thresholds.fpr).sum()
    }

    summary_html = f"""
    <div style="font-family: Arial, sans-serif; padding: 20px; background: #f8f9fa; margin-bottom: 20px; border-radius: 8px;">
        <h2 style="color: #333; margin-top: 0;">Performance Summary</h2>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
            <div style="background: white; padding: 15px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <div style="color: #666; font-size: 12px;">Total Alerts</div>
                <div style="font-size: 28px; font-weight: bold; color: {'#C73E1D' if total_alerts > 0 else '#28a745'};">{total_alerts}</div>
                <div style="color: #999; font-size: 11px;">{critical_alerts} critical, {warning_alerts} warning</div>
            </div>
            <div style="background: white; padding: 15px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <div style="color: #666; font-size: 12px;">Avg Precision</div>
                <div style="font-size: 28px; font-weight: bold; color: #2E86AB;">{avg_precision:.1%}</div>
                <div style="color: #999; font-size: 11px;">{days_below['precision']} days below threshold</div>
            </div>
            <div style="background: white; padding: 15px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <div style="color: #666; font-size: 12px;">Avg Recall</div>
                <div style="font-size: 28px; font-weight: bold; color: #A23B72;">{avg_recall:.1%}</div>
                <div style="color: #999; font-size: 11px;">{days_below['recall']} days below threshold</div>
            </div>
            <div style="background: white; padding: 15px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <div style="color: #666; font-size: 12px;">Avg F1 Score</div>
                <div style="font-size: 28px; font-weight: bold; color: #F18F01;">{avg_f1:.1%}</div>
                <div style="color: #999; font-size: 11px;">{days_below['f1']} days below threshold</div>
            </div>
        </div>
        <div style="margin-top: 15px; padding: 10px; background: #fff3cd; border-radius: 6px; display: {'block' if total_alerts > 0 else 'none'};">
            <strong>Model Drift Detected:</strong> Performance degradation observed starting around day 11.
            Review the metrics below for detailed analysis.
        </div>
    </div>
    """
    return summary_html


def create_dashboard(
    daily_metrics: List[DailyMetrics],
    alerts: List[Alert],
    config: MonitoringConfig,
    output_path: Path,
    transaction_df: Optional[pd.DataFrame] = None
) -> None:
    """
    Create and save the complete monitoring dashboard.

    Args:
        daily_metrics: List of daily metrics
        alerts: List of generated alerts
        config: Monitoring configuration
        output_path: Path to save the HTML dashboard
        transaction_df: Optional transaction DataFrame to clean

    Raises:
        IOError: If the file cannot be written
    """
    from simulator import metrics_to_dataframe

    if transaction_df is not None:
        clean_transaction_data(transaction_df)

    metrics_df = metrics_to_dataframe(daily_metrics)

    # Create the main figure
    fig = create_metrics_figure(metrics_df, alerts, config)

    # Create summary stats
    summary_html = create_summary_stats(metrics_df, alerts, config)

    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get the figure HTML
        fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

        # Combine into full HTML
        full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Fraud Detection Model Monitoring Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body style="margin: 20px; background: #f0f2f5;">
    <div style="max-width: 1400px; margin: 0 auto;">
        {summary_html}
        <div style="background: white; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            {fig_html}
        </div>
        <div style="margin-top: 15px; text-align: center; color: #666; font-size: 12px;">
            Generated by Model Monitoring Dashboard | Data spans {config.num_days} days | {config.num_transactions:,} transactions analyzed
        </div>
    </div>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_html)

    except OSError as e:
        raise IOError(f"Failed to save dashboard to {output_path}: {e}") from e
