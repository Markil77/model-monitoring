"""Generate synthetic fraud transaction data for monitoring."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd

from config import MonitoringConfig


@dataclass
class Transaction:
    """Represents a single transaction."""

    transaction_id: str
    timestamp: datetime
    amount: float
    merchant_category: str
    is_fraud: bool


# Merchant categories with associated fraud risk levels
MERCHANT_CATEGORIES = {
    "grocery": 0.005,      # Low risk
    "gas_station": 0.01,   # Low risk
    "restaurant": 0.01,    # Low risk
    "online_retail": 0.03, # Medium risk
    "electronics": 0.04,   # Medium-high risk
    "jewelry": 0.05,       # High risk
    "wire_transfer": 0.08, # Very high risk
    "cryptocurrency": 0.10 # Highest risk
}


def generate_transaction_amount(is_fraud: bool, rng: np.random.Generator) -> float:
    """Generate a realistic transaction amount based on fraud status."""
    if is_fraud:
        # Fraud transactions tend to be higher amounts
        # Bimodal: either small test transactions or large fraudulent ones
        if rng.random() < 0.3:
            # Small test transaction
            amount = rng.uniform(1, 10)
        else:
            # Large fraudulent transaction
            amount = rng.lognormal(mean=7, sigma=1.5)
            amount = min(amount, 50000)  # Cap at 50k
    else:
        # Legitimate transactions follow typical spending patterns
        amount = rng.lognormal(mean=4, sigma=1.2)
        amount = min(amount, 10000)  # Cap at 10k

    return round(amount, 2)


def generate_timestamp(
    day: int,
    is_fraud: bool,
    base_date: datetime,
    rng: np.random.Generator
) -> datetime:
    """Generate timestamp with fraud patterns (more fraud at night/early morning)."""
    if is_fraud and rng.random() < 0.6:
        # Fraud more likely during off-hours
        hour = int(rng.choice([0, 1, 2, 3, 4, 5, 22, 23]))
    else:
        # Normal business hours distribution
        hour = int(rng.normal(14, 4))
        hour = max(0, min(23, hour))

    minute = int(rng.integers(0, 60))
    second = int(rng.integers(0, 60))

    return base_date + timedelta(days=day, hours=hour, minutes=minute, seconds=second)


def generate_transactions(config: MonitoringConfig) -> pd.DataFrame:
    """Generate synthetic transaction data with realistic fraud patterns."""
    rng = np.random.default_rng(config.random_seed)

    transactions: List[dict] = []
    base_date = datetime(2024, 1, 1)

    categories = list(MERCHANT_CATEGORIES.keys())
    category_weights = list(MERCHANT_CATEGORIES.values())

    # Normalize weights for category selection (inverse of fraud rate for frequency)
    category_probs = [1 / (w + 0.01) for w in category_weights]
    category_probs = [p / sum(category_probs) for p in category_probs]

    transactions_per_day = config.num_transactions // config.num_days

    for day in range(config.num_days):
        for i in range(transactions_per_day):
            # Select merchant category
            category = rng.choice(categories, p=category_probs)
            category_fraud_rate = MERCHANT_CATEGORIES[category]

            # Determine if fraud based on category risk and base fraud rate
            adjusted_fraud_rate = config.fraud_rate * (category_fraud_rate / 0.02)
            is_fraud = rng.random() < adjusted_fraud_rate

            transaction = {
                "transaction_id": f"TXN{day:03d}{i:05d}",
                "timestamp": generate_timestamp(day, is_fraud, base_date, rng),
                "amount": generate_transaction_amount(is_fraud, rng),
                "merchant_category": category,
                "is_fraud": is_fraud,
                "day": day + 1  # 1-indexed for display
            }
            transactions.append(transaction)

    df = pd.DataFrame(transactions)
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def get_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Get summary statistics for each day."""
    summary = df.groupby("day").agg(
        total_transactions=("transaction_id", "count"),
        total_amount=("amount", "sum"),
        fraud_count=("is_fraud", "sum"),
        fraud_rate=("is_fraud", "mean")
    ).reset_index()

    summary["total_amount"] = summary["total_amount"].round(2)
    summary["fraud_rate"] = (summary["fraud_rate"] * 100).round(2)

    return summary
