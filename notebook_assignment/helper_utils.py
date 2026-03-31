"""Helper utilities for the Categorical Feature Analysis assignment."""

import numpy as np
import pandas as pd


def get_orders_df() -> pd.DataFrame:
    """
    Generate the e-commerce orders DataFrame used throughout this assignment.

    Returns a deterministic 300-row DataFrame with the following columns:
        order_value, days_to_ship, items_ordered, customer_age,
        region, returned, days_since_last_purchase.
    """
    rng = np.random.default_rng(42)
    n = 300
    raw_data = {
        "order_value": rng.exponential(scale=85, size=n).round(2),
        "days_to_ship": rng.uniform(1, 30, size=n).astype(float),
        "items_ordered": rng.integers(1, 12, size=n).astype(float),
        "customer_age": rng.normal(loc=38, scale=11, size=n).clip(18, 75).round(),
        "region": rng.choice(
            ["North", "South", "East", "West", "Central"],
            size=n,
            p=[0.30, 0.25, 0.20, 0.15, 0.10],
        ).tolist(),
        "returned": rng.choice([0, 1], size=n, p=[0.82, 0.18]).tolist(),
        "days_since_last_purchase": rng.lognormal(mean=3.5, sigma=0.5, size=n).round(1),
    }
    return pd.DataFrame(raw_data)


def show_context(title: str, description: str) -> None:
    """Print a formatted section header for context.

    Args:
        title: Short section title displayed in the header banner.
        description: Multi-line description printed below the banner.
    """
    separator = "=" * 60
    print(f"\n{separator}")
    print(f"  {title}")
    print(separator)
    print(description)
    print()
