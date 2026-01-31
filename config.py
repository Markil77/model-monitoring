"""Configuration for the model monitoring dashboard."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class EmailConfig:
    """Configuration for email alerts."""

    enabled: bool = False
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    use_tls: bool = True
    sender_email: str = ""
    sender_password: str = ""  # Use app password or env variable
    recipients: List[str] = field(default_factory=list)
    subject_prefix: str = "[Model Alert]"


@dataclass
class MetricThresholds:
    """Threshold values for model performance metrics."""

    precision: float = 0.85
    recall: float = 0.70
    f1: float = 0.75
    fpr: float = 0.10  # False positive rate (upper bound)


@dataclass
class MonitoringConfig:
    """Configuration for the monitoring dashboard."""

    # Data generation parameters
    num_transactions: int = 10000
    num_days: int = 30
    fraud_rate: float = 0.02  # 2% baseline fraud rate
    random_seed: int = 42

    # Metric thresholds
    thresholds: MetricThresholds = None

    # Email alerts configuration
    email: EmailConfig = None

    # Output paths
    output_dir: Path = None
    alerts_file: Path = None
    dashboard_file: Path = None

    def __post_init__(self) -> None:
        """Initialize default values after dataclass creation."""
        if self.thresholds is None:
            self.thresholds = MetricThresholds()

        if self.email is None:
            self.email = EmailConfig()

        if self.output_dir is None:
            self.output_dir = Path(__file__).parent / "output"

        if self.alerts_file is None:
            self.alerts_file = self.output_dir / "alerts.json"

        if self.dashboard_file is None:
            self.dashboard_file = self.output_dir / "dashboard.html"


def get_default_config() -> MonitoringConfig:
    """Return the default monitoring configuration."""
    return MonitoringConfig()
