"""
Helper utilities for the Duplicate Detection & Removal assignment.

This module provides data generation, visualization functions, and display utilities
to support learning about duplicate detection and deduplication strategies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_customer_dataset(
    n_samples: int = 500,
    duplicate_rate: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic customer dataset with exact and near-duplicate rows.

    Args:
        n_samples: Number of base (unique) samples to generate before injection.
        duplicate_rate: Fraction of rows to duplicate (0.0 to 1.0).
        random_state: Random seed for reproducibility.

    Returns:
        pd.DataFrame with columns: customer_id, name, email, age,
        purchase_amount, signup_date.  The DataFrame contains both
        exact duplicates and near-duplicates (name with minor casing / spacing
        variation) mixed in among the unique rows.
    """
    rng = np.random.default_rng(random_state)

    first_names = [
        "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry",
        "Iris", "Jack", "Karen", "Liam", "Mia", "Noah", "Olivia", "Peter",
        "Quinn", "Rachel", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xavier",
        "Yara", "Zoe",
    ]
    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
        "Davis", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson",
        "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee",
    ]

    n_unique = n_samples
    fn = rng.choice(first_names, n_unique)
    ln = rng.choice(last_names, n_unique)
    names = [f"{f} {l}" for f, l in zip(fn, ln)]

    customer_ids = [f"CUST{str(i).zfill(5)}" for i in range(1, n_unique + 1)]
    domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "mail.com"]
    emails = [
        f"{fn[i].lower()}.{ln[i].lower()}{rng.integers(1, 999)}@{rng.choice(domains)}"
        for i in range(n_unique)
    ]
    ages = rng.integers(18, 75, n_unique).tolist()
    purchase_amounts = np.round(rng.uniform(10.0, 2000.0, n_unique), 2).tolist()

    start = np.datetime64("2020-01-01")
    end = np.datetime64("2024-12-31")
    days_range = (end - start).astype(int)
    offsets = rng.integers(0, days_range, n_unique)
    signup_dates = [(start + np.timedelta64(int(o), "D")).astype(str) for o in offsets]

    df_base = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "name": names,
            "email": emails,
            "age": ages,
            "purchase_amount": purchase_amounts,
            "signup_date": signup_dates,
        }
    )

    # ── Inject exact duplicates ──────────────────────────────────────────────
    n_exact = int(n_unique * duplicate_rate * 0.7)
    exact_indices = rng.choice(n_unique, n_exact, replace=True)
    exact_dupes = df_base.iloc[exact_indices].copy()

    # ── Inject near-duplicates (name variation only) ─────────────────────────
    n_near = int(n_unique * duplicate_rate * 0.3)
    near_indices = rng.choice(n_unique, n_near, replace=True)
    near_dupes = df_base.iloc[near_indices].copy()

    def _vary_name(name: str, rng: np.random.Generator) -> str:
        """Apply a minor casing or whitespace variation to a name."""
        choice = rng.integers(0, 3)
        parts = name.split()
        if choice == 0:
            return name.upper()
        elif choice == 1:
            return name.lower()
        else:
            return "  ".join(parts)

    near_dupes["name"] = [
        _vary_name(n, rng) for n in near_dupes["name"]
    ]

    df_combined = pd.concat(
        [df_base, exact_dupes, near_dupes], ignore_index=True
    )
    df_shuffled = df_combined.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df_shuffled


def visualize_duplicate_summary(df: pd.DataFrame) -> None:
    """
    Display a bar chart of exact vs near-duplicate counts and print a summary.

    Args:
        df: Customer DataFrame (as returned by generate_customer_dataset).
    """
    exact_mask = df.duplicated(keep=False)
    exact_count = df.duplicated().sum()

    # Near-duplicate detection on names via simple normalisation
    normalised = df["name"].str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    name_counts = normalised.value_counts()
    near_dup_names = name_counts[name_counts > 1].index
    near_mask = normalised.isin(near_dup_names) & ~exact_mask
    near_count = near_mask.sum()

    counts = {"Exact duplicates": int(exact_count), "Near-duplicates (name)": int(near_count)}

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        counts.keys(),
        counts.values(),
        color=["#e74c3c", "#f39c12"],
        edgecolor="black",
        width=0.5,
    )
    for bar, val in zip(bars, counts.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(val),
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_title("Duplicate Summary", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(counts.values()) * 1.25 + 5)
    plt.tight_layout()
    plt.show()

    total = len(df)
    print("=" * 45)
    print(f"  Total rows          : {total}")
    print(f"  Exact duplicates    : {counts['Exact duplicates']}")
    print(f"  Near-duplicates     : {counts['Near-duplicates (name)']}")
    print(f"  Estimated clean rows: {total - counts['Exact duplicates']}")
    print("=" * 45)


def plot_duplicate_distribution(df: pd.DataFrame, original_df: pd.DataFrame) -> None:
    """
    Before-and-after comparison of key numeric column distributions.

    Args:
        df: Cleaned (deduplicated) DataFrame.
        original_df: Original DataFrame before deduplication.
    """
    numeric_cols = ["age", "purchase_amount"]
    fig, axes = plt.subplots(len(numeric_cols), 2, figsize=(12, 4 * len(numeric_cols)))

    for row_idx, col in enumerate(numeric_cols):
        if col not in original_df.columns or col not in df.columns:
            continue

        orig_vals = original_df[col].dropna()
        clean_vals = df[col].dropna()

        # Before
        axes[row_idx, 0].hist(orig_vals, bins=30, color="#3498db", alpha=0.8, edgecolor="black")
        axes[row_idx, 0].axvline(
            orig_vals.mean(), color="red", linestyle="--", linewidth=1.5,
            label=f"Mean: {orig_vals.mean():.1f}"
        )
        axes[row_idx, 0].set_title(f"{col} — Before (n={len(orig_vals)})", fontweight="bold")
        axes[row_idx, 0].legend(fontsize=9)

        # After
        axes[row_idx, 1].hist(clean_vals, bins=30, color="#2ecc71", alpha=0.8, edgecolor="black")
        axes[row_idx, 1].axvline(
            clean_vals.mean(), color="red", linestyle="--", linewidth=1.5,
            label=f"Mean: {clean_vals.mean():.1f}"
        )
        axes[row_idx, 1].set_title(f"{col} — After (n={len(clean_vals)})", fontweight="bold")
        axes[row_idx, 1].legend(fontsize=9)

    plt.suptitle("Distribution Comparison: Before vs After Deduplication", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()
