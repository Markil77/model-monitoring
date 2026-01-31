"""Main entry point for the model monitoring dashboard."""

import sys
from pathlib import Path

# Add the current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from alerts import check_all_days, get_alerts_summary, save_alerts
from config import get_default_config
from dashboard import create_dashboard
from data_generator import generate_transactions, get_daily_summary
from email_alerts import send_alert_email, should_send_alert
from simulator import metrics_to_dataframe, run_simulation


def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)


def print_metrics_table(metrics_df) -> None:
    """Print a summary metrics table."""
    print("\nDaily Metrics Summary (first 10 and last 5 days):")
    print("-" * 70)
    print(f"{'Day':>4} | {'Precision':>10} | {'Recall':>8} | {'F1':>8} | {'FPR':>8}")
    print("-" * 70)

    # Show first 10 days
    for _, row in metrics_df.head(10).iterrows():
        print(
            f"{int(row['day']):>4} | "
            f"{row['precision']:>10.3f} | "
            f"{row['recall']:>8.3f} | "
            f"{row['f1']:>8.3f} | "
            f"{row['fpr']:>8.3f}"
        )

    if len(metrics_df) > 15:
        print(f"{'...':>4} | {'...':>10} | {'...':>8} | {'...':>8} | {'...':>8}")

    # Show last 5 days
    for _, row in metrics_df.tail(5).iterrows():
        print(
            f"{int(row['day']):>4} | "
            f"{row['precision']:>10.3f} | "
            f"{row['recall']:>8.3f} | "
            f"{row['f1']:>8.3f} | "
            f"{row['fpr']:>8.3f}"
        )


def main() -> int:
    """Run the complete monitoring dashboard pipeline."""
    print_header("Fraud Detection Model Monitoring Dashboard")

    # Load configuration
    config = get_default_config()
    print(f"\nConfiguration:")
    print(f"  - Transactions: {config.num_transactions:,}")
    print(f"  - Days: {config.num_days}")
    print(f"  - Fraud rate: {config.fraud_rate:.1%}")
    print(f"  - Random seed: {config.random_seed}")

    print(f"\nThresholds:")
    print(f"  - Precision: {config.thresholds.precision:.2f}")
    print(f"  - Recall: {config.thresholds.recall:.2f}")
    print(f"  - F1: {config.thresholds.f1:.2f}")
    print(f"  - FPR (max): {config.thresholds.fpr:.2f}")

    # Generate transaction data
    print_header("Generating Synthetic Transaction Data")
    transactions_df = generate_transactions(config)

    daily_summary = get_daily_summary(transactions_df)
    total_fraud = transactions_df["is_fraud"].sum()
    print(f"Generated {len(transactions_df):,} transactions")
    print(f"Total fraud cases: {total_fraud:,} ({total_fraud/len(transactions_df):.2%})")

    # Run simulation
    print_header("Running 30-Day Performance Simulation")
    result_df, daily_metrics = run_simulation(transactions_df, config)
    metrics_df = metrics_to_dataframe(daily_metrics)

    print_metrics_table(metrics_df)

    # Check for alerts
    print_header("Checking Thresholds and Generating Alerts")
    alerts = check_all_days(daily_metrics, config)
    summary = get_alerts_summary(alerts)

    print(f"\nTotal alerts: {summary['total']}")
    print(f"  - Critical: {summary['by_severity']['critical']}")
    print(f"  - Warning: {summary['by_severity']['warning']}")
    print(f"\nAlerts by metric:")
    for metric, count in summary["by_metric"].items():
        if count > 0:
            print(f"  - {metric}: {count} days")

    if summary["affected_days"]:
        print(f"\nAffected days: {summary['affected_days'][:10]}...")

    # Save alerts
    try:
        save_alerts(alerts, config.alerts_file)
        print(f"\nAlerts saved to: {config.alerts_file}")
    except IOError as e:
        print(f"\nError saving alerts: {e}")
        return 1

    # Send email alerts if configured
    if should_send_alert(alerts):
        print_header("Sending Email Alerts")
        email_result = send_alert_email(alerts, config.email)
        if email_result.success:
            if config.email.enabled:
                print(f"Email sent to {email_result.recipients_reached} recipient(s)")
            else:
                print(f"Email alerts: {email_result.message}")
        else:
            print(f"Email alert failed: {email_result.message}")

    # Create dashboard
    print_header("Creating Dashboard")
    try:
        create_dashboard(
            daily_metrics, alerts, config, config.dashboard_file, transactions_df
        )
        print(f"Dashboard saved to: {config.dashboard_file}")
        print(f"\nOpen in browser: file://{config.dashboard_file.absolute()}")
    except IOError as e:
        print(f"\nError creating dashboard: {e}")
        return 1

    # Final summary
    print_header("Summary")
    if summary["total"] > 0:
        print(
            f"Model drift detected. {summary['total']} alerts generated "
            f"across {len(summary['affected_days'])} days."
        )
        print("Review the dashboard for detailed analysis and consider:")
        print("  1. Investigating data drift in recent transactions")
        print("  2. Retraining the model with recent data")
        print("  3. Adjusting thresholds if business requirements changed")
    else:
        print("All metrics within acceptable thresholds. Model performing well.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
