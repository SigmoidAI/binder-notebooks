"""Helper utilities for the Target Variable Deep Dive assignment."""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


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


def plot_feature_by_target(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    figsize: tuple = (8, 4),
) -> plt.Axes:
    """
    Plot a side-by-side violin/box analysis of a numeric feature split by target class.

    Args:
        df: Input DataFrame.
        feature_col: Name of the numeric column to analyse.
        target_col: Name of the binary target column.
        figsize: Figure size tuple.

    Returns:
        matplotlib Axes object.
    """
    classes = sorted(df[target_col].unique())
    data_per_class = [df.loc[df[target_col] == c, feature_col].dropna().values for c in classes]

    fig, ax = plt.subplots(figsize=figsize)
    parts = ax.violinplot(data_per_class, positions=range(len(classes)), showmedians=True)

    colors = ["#4C72B0", "#C44E52"]
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([f"{target_col}={c}" for c in classes])
    ax.set_ylabel(feature_col)
    ax.set_title(f"{feature_col} distribution by {target_col}")
    plt.tight_layout()
    return ax


def print_imbalance_report(minority_proportion: float) -> None:
    """
    Print a human-readable imbalance severity report.

    Args:
        minority_proportion: Proportion of the minority class (0–1).
    """
    pct = minority_proportion * 100
    if minority_proportion >= 0.40:
        severity = "Balanced"
        advice = "Standard accuracy metrics are reliable."
    elif minority_proportion >= 0.20:
        severity = "Moderate Imbalance"
        advice = "Consider class-weighted models or stratified splits."
    elif minority_proportion >= 0.05:
        severity = "Severe Imbalance"
        advice = "Use precision/recall/F1. Consider SMOTE or class weights."
    else:
        severity = "Critical Imbalance"
        advice = "Accuracy is misleading. Use PR-AUC. Consider oversampling."

    print(f"  Minority proportion : {pct:.1f}%")
    print(f"  Severity            : {severity}")
    print(f"  Recommendation      : {advice}")
