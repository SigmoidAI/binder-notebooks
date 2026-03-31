"""
Helper utilities for the Inconsistency Resolution assignment.

This module provides data generation, visualization functions, and display
utilities to support learning about inconsistency resolution strategies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re


def generate_messy_dataset(n_samples: int = 200, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a DataFrame with inconsistent formats in multiple columns.

    Columns created:
    - country   : mix of 'USA', 'US', 'United States', 'u.s.a', 'United States of America'
    - price     : mix of '$1,200.50', '1200.50 USD', '1,200', 'EUR 900.00'
    - date      : mix of '2023-01-15', '01/15/2023', '15.01.2023', 'Jan 15, 2023'
    - weight_kg : numeric, some with 'kg'/'KG' suffix, e.g. '72.5kg', '72.5', '72.5 KG'
    - status    : mix of 'Active', 'active', 'ACTIVE', 'Inactive', 'inactive', 'INACTIVE'

    Args:
        n_samples:    Number of rows to generate.
        random_state: Random seed for reproducibility.

    Returns:
        pd.DataFrame with all columns having object dtype.
    """
    rng = np.random.default_rng(random_state)

    # ── country ───────────────────────────────────────────────────────────────
    country_variants = ["USA", "US", "United States", "u.s.a", "United States of America"]
    country = rng.choice(country_variants, size=n_samples).tolist()

    # ── price ─────────────────────────────────────────────────────────────────
    base_prices = rng.uniform(100.0, 2000.0, size=n_samples)
    price_format_indices = rng.integers(0, 4, size=n_samples)
    price = []
    for p, fi in zip(base_prices, price_format_indices):
        if fi == 0:
            price.append(f"${p:,.2f}")
        elif fi == 1:
            price.append(f"{p:.2f} USD")
        elif fi == 2:
            price.append(f"{p:,.0f}")
        else:
            price.append(f"EUR {p:.2f}")

    # ── date ──────────────────────────────────────────────────────────────────
    start_date = pd.Timestamp("2023-01-01")
    end_date = pd.Timestamp("2023-12-31")
    range_days = (end_date - start_date).days
    offsets = rng.integers(0, range_days + 1, size=n_samples)
    date_formats = ["%Y-%m-%d", "%m/%d/%Y", "%d.%m.%Y", "%b %d, %Y"]
    date_format_indices = rng.integers(0, len(date_formats), size=n_samples)
    date = []
    for offset, fi in zip(offsets, date_format_indices):
        d = start_date + pd.Timedelta(days=int(offset))
        date.append(d.strftime(date_formats[int(fi)]))

    # ── weight_kg ─────────────────────────────────────────────────────────────
    weights = rng.uniform(40.0, 120.0, size=n_samples).round(1)
    weight_format_indices = rng.integers(0, 3, size=n_samples)
    weight_kg = []
    for w, fi in zip(weights, weight_format_indices):
        if fi == 0:
            weight_kg.append(f"{w}kg")
        elif fi == 1:
            weight_kg.append(f"{w} KG")
        else:
            weight_kg.append(str(w))

    # ── status ────────────────────────────────────────────────────────────────
    status_variants = ["Active", "active", "ACTIVE", "Inactive", "inactive", "INACTIVE"]
    status = rng.choice(status_variants, size=n_samples).tolist()

    return pd.DataFrame(
        {
            "country":   country,
            "price":     price,
            "date":      date,
            "weight_kg": weight_kg,
            "status":    status,
        }
    )


def visualize_before_after(
    original_col: pd.Series,
    cleaned_col: pd.Series,
    col_name: str,
) -> None:
    """
    Display a side-by-side bar chart of value counts for original vs cleaned column.

    Args:
        original_col: Series of original (messy) values.
        cleaned_col:  Series of cleaned values.
        col_name:     Column name for titling the chart.
    """
    before_counts = original_col.value_counts()
    after_counts  = cleaned_col.astype(str).value_counts()

    n_before = len(before_counts)
    n_after  = len(after_counts)
    fig_height = max(4, min(max(n_before, n_after), 15) * 0.45 + 2)

    fig, axes = plt.subplots(1, 2, figsize=(14, fig_height))

    # ── Before ────────────────────────────────────────────────────────────────
    axes[0].barh(
        before_counts.index.astype(str),
        before_counts.values,
        color="#e74c3c",
        alpha=0.85,
        edgecolor="white",
    )
    axes[0].set_title(
        f"Before – '{col_name}'\n({n_before} unique values)",
        fontsize=11, fontweight="bold",
    )
    axes[0].set_xlabel("Count", fontsize=10)
    axes[0].invert_yaxis()

    # ── After ─────────────────────────────────────────────────────────────────
    axes[1].barh(
        after_counts.index.astype(str),
        after_counts.values,
        color="#2ecc71",
        alpha=0.85,
        edgecolor="white",
    )
    axes[1].set_title(
        f"After – '{col_name}'\n({n_after} unique values)",
        fontsize=11, fontweight="bold",
    )
    axes[1].set_xlabel("Count", fontsize=10)
    axes[1].invert_yaxis()

    plt.suptitle(
        f"Value Distribution: {col_name}",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.show()


def plot_inconsistency_summary(df: pd.DataFrame) -> None:
    """
    Show unique value counts per column as a bar chart to highlight inconsistency hotspots.

    Higher unique-value counts typically indicate more inconsistency in that column.

    Args:
        df: DataFrame whose columns to analyse.
    """
    col_names     = list(df.columns)
    unique_counts = [df[c].nunique() for c in col_names]

    colors = sns.color_palette("coolwarm", len(col_names))

    fig, ax = plt.subplots(figsize=(max(6, len(col_names) * 1.4), 5))
    bars = ax.bar(
        col_names, unique_counts,
        color=colors, alpha=0.87, edgecolor="white", linewidth=0.8,
    )

    for bar, count in zip(bars, unique_counts):
        ax.annotate(
            str(count),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4), textcoords="offset points",
            ha="center", fontsize=10, fontweight="bold",
        )

    ax.set_xlabel("Column", fontsize=11)
    ax.set_ylabel("Number of Unique Values", fontsize=11)
    ax.set_title(
        "Unique Value Counts per Column  (higher = more inconsistency)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks(range(len(col_names)))
    ax.set_xticklabels(col_names, rotation=20, ha="right")
    ax.set_ylim(0, max(unique_counts) * 1.25 + 1)

    plt.tight_layout()
    plt.show()
