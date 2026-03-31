"""Helper utilities for the Matplotlib Fundamentals assignment."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_sales_df() -> pd.DataFrame:
    """
    Generate the sales DataFrame used throughout this assignment.

    Returns a deterministic 50-row DataFrame with the following columns:
        order_id, customer, region, category, order_date, amount, units, status.
    """
    rng = np.random.default_rng(42)
    raw_data = {
        "order_id":   list(range(1001, 1051)),
        "customer":   rng.choice(["Alice", "Bob", "Clara", "David", "Eva"], 50).tolist(),
        "region":     rng.choice(["North", "South", "East", "West"], 50).tolist(),
        "category":   rng.choice(["Electronics", "Clothing", "Food", "Books"], 50).tolist(),
        "order_date": pd.date_range("2024-01-01", periods=50, freq="7D").tolist(),
        "amount":     (rng.normal(loc=250, scale=80, size=50).clip(50)).round(2).tolist(),
        "units":      rng.integers(1, 15, size=50).tolist(),
        "status":     rng.choice(
            ["Delivered", "Pending", "Returned"], 50, p=[0.7, 0.2, 0.1]
        ).tolist(),
    }
    return pd.DataFrame(raw_data)


def show_exercise_context(title: str, description: str) -> None:
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
