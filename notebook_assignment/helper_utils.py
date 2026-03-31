"""
Helper utilities for the Data Type Conversion assignment.

This module provides data generation, visualization functions, and display
utilities to support learning about data type conversion strategies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_mixed_type_dataset(n_samples: int = 200, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a DataFrame with messy string columns that need type conversion.

    Columns created:
    - age_str      : integers 18–80 as strings; ~5 % replaced with 'N/A' or 'unknown'
    - price_str    : prices formatted as '$X.XX' strings
    - date_str     : ISO dates '2020-01-01' – '2023-12-31'; ~3 % replaced with 'invalid'
    - category_str : 'low' / 'medium' / 'high' strings
    - quantity_str : integers 1–99 as strings; ~5 % replaced with empty string ''

    Args:
        n_samples:    Number of rows to generate.
        random_state: Random seed for reproducibility.

    Returns:
        pd.DataFrame with all columns having object dtype.
    """
    rng = np.random.default_rng(random_state)

    # ── age_str ──────────────────────────────────────────────────────────────
    ages = rng.integers(18, 81, size=n_samples)
    age_str = ages.astype(str).tolist()
    n_age_bad = max(1, int(n_samples * 0.05))
    age_bad_idx = rng.choice(n_samples, n_age_bad, replace=False)
    bad_age_values = rng.choice(["N/A", "unknown"], size=n_age_bad)
    for i, idx in enumerate(age_bad_idx):
        age_str[int(idx)] = bad_age_values[i]

    # ── price_str ─────────────────────────────────────────────────────────────
    prices = rng.uniform(1.0, 500.0, size=n_samples)
    price_str = [f"${p:.2f}" for p in prices]

    # ── date_str ──────────────────────────────────────────────────────────────
    start_date = pd.Timestamp("2020-01-01")
    end_date = pd.Timestamp("2023-12-31")
    range_days = (end_date - start_date).days
    offsets = rng.integers(0, range_days + 1, size=n_samples)
    date_str = [
        (start_date + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d")
        for d in offsets
    ]
    n_date_bad = max(1, int(n_samples * 0.03))
    date_bad_idx = rng.choice(n_samples, n_date_bad, replace=False)
    for idx in date_bad_idx:
        date_str[int(idx)] = "invalid"

    # ── category_str ─────────────────────────────────────────────────────────
    category_str = rng.choice(["low", "medium", "high"], size=n_samples).tolist()

    # ── quantity_str ──────────────────────────────────────────────────────────
    quantities = rng.integers(1, 100, size=n_samples)
    quantity_str = quantities.astype(str).tolist()
    n_qty_bad = max(1, int(n_samples * 0.05))
    qty_bad_idx = rng.choice(n_samples, n_qty_bad, replace=False)
    for idx in qty_bad_idx:
        quantity_str[int(idx)] = ""

    return pd.DataFrame(
        {
            "age_str": age_str,
            "price_str": price_str,
            "date_str": date_str,
            "category_str": category_str,
            "quantity_str": quantity_str,
        }
    )


def visualize_type_conversion_results(
    original_df: pd.DataFrame, converted_df: pd.DataFrame
) -> None:
    """
    Print a before/after dtype comparison table and show a grouped bar chart
    of type distributions before and after conversion.

    Args:
        original_df:  DataFrame before type conversion.
        converted_df: DataFrame after type conversion.
    """
    # ── textual comparison ────────────────────────────────────────────────────
    print("=" * 60)
    print("DATA TYPES: BEFORE vs AFTER CONVERSION")
    print("=" * 60)
    comparison = pd.DataFrame(
        {
            "Before": original_df.dtypes.astype(str),
            "After":  converted_df.dtypes.astype(str),
        }
    )
    comparison["Changed"] = comparison["Before"] != comparison["After"]
    print(comparison.to_string())
    print()

    n_changed = comparison["Changed"].sum()
    print(f"Columns changed: {n_changed} / {len(comparison)}")
    print()

    # ── bar chart of type distributions ───────────────────────────────────────
    type_counts_before = original_df.dtypes.astype(str).value_counts()
    type_counts_after  = converted_df.dtypes.astype(str).value_counts()

    all_types = sorted(
        set(type_counts_before.index) | set(type_counts_after.index)
    )
    before_counts = [int(type_counts_before.get(t, 0)) for t in all_types]
    after_counts  = [int(type_counts_after.get(t, 0))  for t in all_types]

    x = np.arange(len(all_types))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(all_types) * 1.8), 5))
    bars_before = ax.bar(
        x - width / 2, before_counts, width,
        label="Before", color="#e74c3c", alpha=0.85
    )
    bars_after = ax.bar(
        x + width / 2, after_counts, width,
        label="After", color="#2ecc71", alpha=0.85
    )

    for bars in (bars_before, bars_after):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(
                    str(int(h)),
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=9,
                )

    ax.set_xlabel("Data Type", fontsize=11)
    ax.set_ylabel("Number of Columns", fontsize=11)
    ax.set_title("Column Data Types Before and After Conversion", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(all_types, rotation=30, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_conversion_errors(error_summary: dict) -> None:
    """
    Display a bar chart showing the number of conversion errors per column.

    Args:
        error_summary: dict mapping column name -> number of conversion errors.
    """
    if not error_summary:
        print("No conversion errors to display.")
        return

    columns = list(error_summary.keys())
    counts  = list(error_summary.values())

    colors = ["#e74c3c" if c > 0 else "#2ecc71" for c in counts]

    fig, ax = plt.subplots(figsize=(max(6, len(columns) * 1.4), 5))
    bars = ax.bar(columns, counts, color=colors, alpha=0.87, edgecolor="white", linewidth=0.8)

    for bar, count in zip(bars, counts):
        ax.annotate(
            str(count),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4), textcoords="offset points",
            ha="center", fontsize=10, fontweight="bold",
        )

    ax.set_xlabel("Column", fontsize=11)
    ax.set_ylabel("Number of Conversion Errors", fontsize=11)
    ax.set_title("Conversion Errors by Column", fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=30, ha="right")
    ax.set_ylim(0, max(counts) * 1.25 + 1)

    plt.tight_layout()
    plt.show()
