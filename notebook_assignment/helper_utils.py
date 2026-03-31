"""
Helper utilities for the Missingness Pattern Analysis assignment.

This module provides functions for generating datasets with different
missingness patterns (MCAR, MAR, MNAR) and visualization utilities
for analyzing missing data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional


def generate_mcar_data(
    n_samples: int = 500,
    missing_rate: float = 0.2,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate a dataset with Missing Completely At Random (MCAR) pattern.

    In MCAR, the probability of missingness is independent of both
    observed and unobserved data. The missing values are randomly
    distributed across all observations.

    Args:
        n_samples: Number of samples to generate
        missing_rate: Proportion of values to make missing (0 to 1)
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with columns: age, income, education_years, satisfaction
        where 'income' has MCAR missing values
    """
    np.random.seed(random_state)

    # Generate complete data
    age = np.random.normal(40, 12, n_samples).clip(18, 70).astype(int)
    income = np.random.normal(50000, 20000, n_samples).clip(20000, 150000)
    education_years = np.random.normal(14, 3, n_samples).clip(8, 22).astype(int)
    satisfaction = np.random.normal(7, 2, n_samples).clip(1, 10)

    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'education_years': education_years,
        'satisfaction': satisfaction
    })

    # Apply MCAR: Random missing values independent of any variable
    missing_mask = np.random.random(n_samples) < missing_rate
    df.loc[missing_mask, 'income'] = np.nan

    return df


def generate_mar_data(
    n_samples: int = 500,
    missing_rate: float = 0.3,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate a dataset with Missing At Random (MAR) pattern.

    In MAR, the probability of missingness depends on observed data
    but not on the missing value itself. Here, higher age makes
    income more likely to be missing (older people less likely to report income).

    Args:
        n_samples: Number of samples to generate
        missing_rate: Base proportion of values to make missing (0 to 1)
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with columns: age, income, education_years, satisfaction
        where 'income' has MAR missing values (depends on age)
    """
    np.random.seed(random_state)

    # Generate complete data
    age = np.random.normal(40, 12, n_samples).clip(18, 70).astype(int)
    income = np.random.normal(50000, 20000, n_samples).clip(20000, 150000)
    education_years = np.random.normal(14, 3, n_samples).clip(8, 22).astype(int)
    satisfaction = np.random.normal(7, 2, n_samples).clip(1, 10)

    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'education_years': education_years,
        'satisfaction': satisfaction
    })

    # Apply MAR: Probability of missing income increases with age
    # Normalize age to 0-1 range for probability calculation
    age_normalized = (age - age.min()) / (age.max() - age.min())

    # Higher age = higher probability of missing
    missing_prob = missing_rate * (0.5 + age_normalized)
    missing_prob = np.clip(missing_prob, 0, 0.8)

    missing_mask = np.random.random(n_samples) < missing_prob
    df.loc[missing_mask, 'income'] = np.nan

    return df


def generate_mnar_data(
    n_samples: int = 500,
    missing_rate: float = 0.3,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate a dataset with Missing Not At Random (MNAR) pattern.

    In MNAR, the probability of missingness depends on the unobserved
    (missing) value itself. Here, high earners are less likely to
    report their income (income itself affects its missingness).

    Args:
        n_samples: Number of samples to generate
        missing_rate: Base proportion of values to make missing (0 to 1)
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with columns: age, income, education_years, satisfaction
        where 'income' has MNAR missing values (depends on income itself)
    """
    np.random.seed(random_state)

    # Generate complete data
    age = np.random.normal(40, 12, n_samples).clip(18, 70).astype(int)
    income = np.random.normal(50000, 20000, n_samples).clip(20000, 150000)
    education_years = np.random.normal(14, 3, n_samples).clip(8, 22).astype(int)
    satisfaction = np.random.normal(7, 2, n_samples).clip(1, 10)

    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'education_years': education_years,
        'satisfaction': satisfaction
    })

    # Apply MNAR: Probability of missing income increases with income value
    # Higher income = higher probability of not reporting
    income_normalized = (income - income.min()) / (income.max() - income.min())

    missing_prob = missing_rate * (0.3 + 1.4 * income_normalized)
    missing_prob = np.clip(missing_prob, 0, 0.9)

    missing_mask = np.random.random(n_samples) < missing_prob
    df.loc[missing_mask, 'income'] = np.nan

    return df


def plot_missingness_heatmap(df: pd.DataFrame, title: str = "Missingness Heatmap") -> None:
    """
    Create a heatmap showing the pattern of missing values in the dataset.

    Args:
        df: DataFrame to visualize
        title: Title for the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create binary matrix of missingness
    missing_matrix = df.isnull().astype(int)

    # Plot heatmap
    sns.heatmap(
        missing_matrix.T,
        cmap=['lightblue', 'red'],
        cbar_kws={'label': 'Missing (1) / Present (0)'},
        yticklabels=df.columns,
        ax=ax
    )

    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)
    ax.set_title(f'{title}\n(Red = Missing, Blue = Present)', fontsize=14)

    plt.tight_layout()
    plt.show()


def plot_missingness_correlation(
    missingness_indicators: pd.DataFrame,
    title: str = "Missingness Correlation Matrix"
) -> None:
    """
    Create a correlation heatmap of missingness indicator variables.

    This visualization helps identify if missingness in one variable
    is related to missingness in other variables.

    Args:
        missingness_indicators: DataFrame of binary missingness indicators
        title: Title for the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Compute correlation matrix
    corr_matrix = missingness_indicators.corr()

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    # Plot heatmap
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.3f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax
    )

    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    plt.show()


def plot_missingness_vs_values(
    df: pd.DataFrame,
    missing_col: str,
    value_col: str,
    title: Optional[str] = None
) -> None:
    """
    Compare the distribution of a variable between groups where another
    variable is missing vs present. Useful for detecting MAR patterns.

    Args:
        df: DataFrame containing the data
        missing_col: Column name to check for missingness
        value_col: Column name to compare distributions
        title: Optional title for the plot
    """
    if title is None:
        title = f"Distribution of {value_col} by {missing_col} Missingness"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Create missingness indicator
    missing_indicator = df[missing_col].isnull()

    # Left plot: Box plots
    ax = axes[0]
    groups = [
        df.loc[~missing_indicator, value_col].dropna(),
        df.loc[missing_indicator, value_col].dropna()
    ]

    # Only plot if both groups have data
    valid_groups = []
    labels = []
    if len(groups[0]) > 0:
        valid_groups.append(groups[0])
        labels.append(f'{missing_col}\nPresent')
    if len(groups[1]) > 0:
        valid_groups.append(groups[1])
        labels.append(f'{missing_col}\nMissing')

    if len(valid_groups) > 0:
        bp = ax.boxplot(valid_groups, labels=labels, patch_artist=True)
        colors = ['lightgreen', 'salmon'][:len(valid_groups)]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

    ax.set_ylabel(value_col, fontsize=12)
    ax.set_title(f'{value_col} by {missing_col} Missingness Status', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Right plot: Histograms
    ax = axes[1]

    if len(groups[0]) > 0:
        ax.hist(groups[0], bins=20, alpha=0.6, label=f'{missing_col} Present',
                color='green', density=True)
    if len(groups[1]) > 0:
        ax.hist(groups[1], bins=20, alpha=0.6, label=f'{missing_col} Missing',
                color='red', density=True)

    ax.set_xlabel(value_col, fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{value_col} Distribution Comparison', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_missingness_summary(df: pd.DataFrame) -> None:
    """
    Create a summary visualization of missingness in the dataset.

    Args:
        df: DataFrame to analyze
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Bar chart of missing counts
    ax = axes[0]
    missing_counts = df.isnull().sum()
    total_counts = len(df)
    missing_pct = (missing_counts / total_counts) * 100

    colors = ['red' if pct > 0 else 'green' for pct in missing_pct]
    bars = ax.bar(missing_counts.index, missing_pct, color=colors, alpha=0.7, edgecolor='black')

    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Missing Percentage (%)', fontsize=12)
    ax.set_title('Missing Values by Feature', fontsize=14)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Add percentage labels on bars
    for bar, pct in zip(bars, missing_pct):
        if pct > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)

    ax.set_ylim(0, max(missing_pct) * 1.2 if max(missing_pct) > 0 else 10)
    plt.xticks(rotation=45, ha='right')

    # Right: Data completeness matrix (sample of rows)
    ax = axes[1]

    # Sample rows for visualization
    sample_size = min(50, len(df))
    sample_idx = np.linspace(0, len(df)-1, sample_size).astype(int)
    sample_df = df.iloc[sample_idx]

    missing_matrix = sample_df.isnull().astype(int).T

    sns.heatmap(
        missing_matrix,
        cmap=['lightgreen', 'red'],
        cbar=False,
        xticklabels=False,
        yticklabels=sample_df.columns,
        ax=ax
    )

    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_title('Missing Pattern (Sample)\n(Red = Missing)', fontsize=14)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\n" + "="*50)
    print("MISSINGNESS SUMMARY")
    print("="*50)
    print(f"Total samples: {len(df)}")
    print(f"Total features: {len(df.columns)}")
    print(f"\nMissing values per column:")
    for col in df.columns:
        count = df[col].isnull().sum()
        pct = (count / len(df)) * 100
        print(f"  {col}: {count} ({pct:.1f}%)")
    print(f"\nRows with any missing: {df.isnull().any(axis=1).sum()}")
    print(f"Complete rows: {df.dropna().shape[0]}")
    print("="*50)


def print_missingness_test_results(
    test_name: str,
    statistic: float,
    p_value: float,
    interpretation: str
) -> None:
    """
    Print formatted results of a missingness test.

    Args:
        test_name: Name of the statistical test
        statistic: Test statistic value
        p_value: P-value of the test
        interpretation: Text interpretation of results
    """
    print("\n" + "="*60)
    print(f"TEST: {test_name}")
    print("="*60)
    print(f"Test Statistic: {statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    print("-"*60)
    print(f"Interpretation: {interpretation}")
    print("="*60 + "\n")


def generate_multi_pattern_data(
    n_samples: int = 500,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate a dataset with multiple columns having different missingness patterns.

    This is useful for testing classification functions that need to identify
    different types of missingness in the same dataset.

    Args:
        n_samples: Number of samples to generate
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with:
        - income: MCAR pattern
        - salary_bonus: MAR pattern (depends on years_experience)
        - net_worth: MNAR pattern (depends on net_worth itself)
        - years_experience, age, department: Complete columns
    """
    np.random.seed(random_state)

    # Generate complete data
    age = np.random.normal(35, 10, n_samples).clip(22, 65).astype(int)
    years_experience = (age - 22 + np.random.normal(0, 3, n_samples)).clip(0, 40).astype(int)
    department = np.random.choice(['Sales', 'Engineering', 'Marketing', 'HR'], n_samples)

    income = np.random.normal(60000, 20000, n_samples).clip(30000, 150000)
    salary_bonus = np.random.normal(5000, 3000, n_samples).clip(0, 20000)
    net_worth = np.random.lognormal(11, 1, n_samples).clip(10000, 2000000)

    df = pd.DataFrame({
        'age': age,
        'years_experience': years_experience,
        'department': department,
        'income': income,
        'salary_bonus': salary_bonus,
        'net_worth': net_worth
    })

    # Apply MCAR to income (random 15% missing)
    mcar_mask = np.random.random(n_samples) < 0.15
    df.loc[mcar_mask, 'income'] = np.nan

    # Apply MAR to salary_bonus (depends on years_experience)
    exp_normalized = (years_experience - years_experience.min()) / (years_experience.max() - years_experience.min())
    mar_prob = 0.1 + 0.3 * exp_normalized
    mar_mask = np.random.random(n_samples) < mar_prob
    df.loc[mar_mask, 'salary_bonus'] = np.nan

    # Apply MNAR to net_worth (high values more likely missing)
    nw_normalized = (net_worth - net_worth.min()) / (net_worth.max() - net_worth.min())
    mnar_prob = 0.05 + 0.4 * nw_normalized
    mnar_mask = np.random.random(n_samples) < mnar_prob
    df.loc[mnar_mask, 'net_worth'] = np.nan

    return df
