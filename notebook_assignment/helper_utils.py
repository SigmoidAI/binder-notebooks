"""
Helper utilities for the Outlier Investigation assignment.

This module provides data generation, visualization functions, and display utilities
to support learning about outlier detection and removal strategies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_sales_dataset(
    n_samples: int = 300,
    outlier_rate: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic sales dataset with injected outliers.

    Args:
        n_samples: Number of rows to generate.
        outlier_rate: Fraction of rows to turn into outliers (0.0 to 1.0).
        random_state: Random seed for reproducibility.

    Returns:
        pd.DataFrame with columns: sales_amount (lognormal), units_sold (Poisson),
        discount_pct (uniform 0–50), profit_margin (normal).  A fraction of rows
        equal to ``outlier_rate`` contain extreme values (multiplied by 10).
    """
    rng = np.random.default_rng(random_state)

    sales_amount = rng.lognormal(mean=5.0, sigma=0.8, size=n_samples)
    units_sold = rng.poisson(lam=20, size=n_samples)
    discount_pct = rng.uniform(0.0, 50.0, size=n_samples)
    profit_margin = rng.normal(loc=20.0, scale=5.0, size=n_samples)

    df = pd.DataFrame(
        {
            "sales_amount": np.round(sales_amount, 2),
            "units_sold": units_sold.astype(int),
            "discount_pct": np.round(discount_pct, 2),
            "profit_margin": np.round(profit_margin, 2),
        }
    )

    # ── Inject outliers ──────────────────────────────────────────────────────
    n_outliers = max(1, int(n_samples * outlier_rate))
    outlier_idx = rng.choice(n_samples, n_outliers, replace=False)

    df.loc[outlier_idx, "sales_amount"] = np.round(
        df.loc[outlier_idx, "sales_amount"] * 10, 2
    )
    df.loc[outlier_idx, "units_sold"] = (
        df.loc[outlier_idx, "units_sold"] * 10
    ).astype(int)
    df.loc[outlier_idx, "profit_margin"] = np.round(
        df.loc[outlier_idx, "profit_margin"] * 10, 2
    )

    return df


def visualize_outliers_boxplot(df: pd.DataFrame, columns=None) -> None:
    """
    Create a boxplot for each numeric column to visualise potential outliers.

    Args:
        df: Input DataFrame.
        columns: List of column names to plot.  Defaults to all numeric columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    n_cols = len(columns)
    if n_cols == 0:
        print("No numeric columns to plot.")
        return

    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        data = df[col].dropna()
        bp = ax.boxplot(
            data,
            vert=True,
            patch_artist=True,
            boxprops=dict(facecolor="#3498db", alpha=0.7),
            medianprops=dict(color="red", linewidth=2),
            whiskerprops=dict(linewidth=1.5),
            flierprops=dict(marker="o", color="#e74c3c", markersize=5, alpha=0.6),
        )
        ax.set_title(col, fontsize=12, fontweight="bold")
        ax.set_ylabel("Value")
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    plt.suptitle("Outlier Detection via Boxplots", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_outlier_comparison(
    original_df: pd.DataFrame, cleaned_df: pd.DataFrame, column: str
) -> None:
    """
    Side-by-side histogram with KDE overlay for a column before and after
    outlier removal.

    Args:
        original_df: DataFrame before outlier removal.
        cleaned_df: DataFrame after outlier removal.
        column: Column name to compare.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    orig_vals = original_df[column].dropna()
    clean_vals = cleaned_df[column].dropna()

    # ── Before ───────────────────────────────────────────────────────────────
    axes[0].hist(
        orig_vals, bins=30, color="#e74c3c", alpha=0.7, edgecolor="black", density=True
    )
    orig_vals.plot(kind="kde", ax=axes[0], color="darkred", linewidth=2, label="KDE")
    axes[0].set_title(
        f"{column} — Before  (n={len(orig_vals)})", fontweight="bold"
    )
    axes[0].set_xlabel(column)
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=9)

    # ── After ────────────────────────────────────────────────────────────────
    axes[1].hist(
        clean_vals,
        bins=30,
        color="#2ecc71",
        alpha=0.7,
        edgecolor="black",
        density=True,
    )
    clean_vals.plot(kind="kde", ax=axes[1], color="darkgreen", linewidth=2, label="KDE")
    axes[1].set_title(
        f"{column} — After  (n={len(clean_vals)})", fontweight="bold"
    )
    axes[1].set_xlabel(column)
    axes[1].set_ylabel("Density")
    axes[1].legend(fontsize=9)

    plt.suptitle(
        f"Outlier Removal Comparison: {column}", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()
