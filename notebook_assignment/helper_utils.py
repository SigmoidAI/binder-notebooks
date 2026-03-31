"""
Helper utilities for the Imputation Strategy Comparison assignment.

This module provides visualization functions and data generation utilities
to help understand different imputation strategies and their effects on data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple


def generate_dataset_with_missing(
    n_samples: int = 200,
    missing_rate: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Generate a consistent dataset with missing values for comparison.

    Args:
        n_samples: Number of samples to generate
        missing_rate: Proportion of values to make missing (0.0 to 1.0)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (df_missing, df_complete, missing_mask)
    """
    np.random.seed(random_state)

    age = np.random.normal(45, 15, n_samples).clip(18, 80).astype(float)
    income = np.random.lognormal(10.5, 0.5, n_samples).astype(float)
    experience = (age - 18 + np.random.normal(0, 5, n_samples)).clip(0, 50).astype(float)
    score = np.random.uniform(0, 100, n_samples).astype(float)

    categories = ['A', 'B', 'C', 'D']
    category = np.random.choice(categories, n_samples, p=[0.3, 0.35, 0.25, 0.1])

    df_complete = pd.DataFrame({
        'age': age, 'income': income, 'experience': experience,
        'score': score, 'category': category
    })

    df_missing = df_complete.copy()
    numerical_cols = ['age', 'income', 'experience', 'score']
    missing_mask = np.random.random((n_samples, 4)) < missing_rate
    missing_mask[:10, :] = False

    for i, col in enumerate(numerical_cols):
        df_missing.loc[missing_mask[:, i], col] = np.nan

    cat_missing_mask = np.random.random(n_samples) < (missing_rate * 0.7)
    cat_missing_mask[:10] = False
    df_missing.loc[cat_missing_mask, 'category'] = np.nan

    return df_missing, df_complete, missing_mask


def plot_imputation_comparison(df_original, df_imputed, column, imputation_method="Imputed"):
    """Visualize distributions before and after imputation for a column."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    original_values = df_original[column].dropna()
    axes[0].hist(original_values, bins=25, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].axvline(original_values.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {original_values.mean():.2f}')
    axes[0].set_title(f'Original (n={len(original_values)})')
    axes[0].legend()

    imputed_values = df_imputed[column]
    axes[1].hist(imputed_values, bins=25, alpha=0.7, color='green', edgecolor='black')
    axes[1].axvline(imputed_values.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {imputed_values.mean():.2f}')
    axes[1].set_title(f'{imputation_method} (n={len(imputed_values)})')
    axes[1].legend()

    axes[2].hist(original_values, bins=25, alpha=0.5, color='steelblue', label='Original')
    axes[2].hist(imputed_values, bins=25, alpha=0.5, color='green', label=imputation_method)
    axes[2].legend()
    axes[2].set_title('Comparison')

    plt.suptitle(f'Distribution Comparison: {column}', fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_imputation_error(df_complete, imputed_results, missing_mask, columns=None):
    """Compare imputation error across methods using known ground truth values."""
    if columns is None:
        columns = ['age', 'income', 'experience', 'score']

    errors = []
    for method_name, df_imputed in imputed_results.items():
        for i, col in enumerate(columns):
            mask = missing_mask[:, i] if i < missing_mask.shape[1] else np.zeros(len(df_complete), dtype=bool)
            if mask.sum() == 0:
                continue
            true_values = df_complete.loc[mask, col].values
            imputed_values = df_imputed.loc[mask, col].values
            mae = np.mean(np.abs(true_values - imputed_values))
            rmse = np.sqrt(np.mean((true_values - imputed_values) ** 2))
            errors.append({'Method': method_name, 'Column': col, 'MAE': mae, 'RMSE': rmse})

    error_df = pd.DataFrame(errors)
    if len(error_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, metric in zip(axes, ['MAE', 'RMSE']):
            pivot_data = error_df.pivot(index='Column', columns='Method', values=metric)
            pivot_data.plot(kind='bar', ax=ax, alpha=0.8)
            ax.set_title(f'{metric} by Column and Method')
            ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()
    return error_df


def create_comparison_table(df_original, imputed_results, columns=None):
    """Create a summary table comparing all imputation results."""
    if columns is None:
        columns = df_original.select_dtypes(include=[np.number]).columns.tolist()

    summary_data = []
    for col in columns:
        original_values = df_original[col].dropna()
        summary_data.append({
            'Method': 'Original', 'Column': col,
            'Mean': original_values.mean(), 'Std': original_values.std(),
            'Median': original_values.median()
        })

    for method_name, df_imputed in imputed_results.items():
        for col in columns:
            values = df_imputed[col]
            summary_data.append({
                'Method': method_name, 'Column': col,
                'Mean': values.mean(), 'Std': values.std(),
                'Median': values.median()
            })
    return pd.DataFrame(summary_data)


def visualize_imputed_values(df_original, df_imputed, column, max_points=50):
    """Highlight imputed vs original values in a scatter plot."""
    fig, ax = plt.subplots(figsize=(12, 6))
    was_missing = df_original[column].isna()
    n_total = len(df_original)
    indices = np.arange(min(n_total, max_points))

    original_mask = ~was_missing.iloc[indices]
    ax.scatter(indices[original_mask], df_imputed[column].iloc[indices][original_mask],
               c='steelblue', s=60, alpha=0.7, label='Original Values')

    imputed_mask = was_missing.iloc[indices]
    ax.scatter(indices[imputed_mask], df_imputed[column].iloc[indices][imputed_mask],
               c='red', s=80, marker='s', label='Imputed Values')

    ax.legend()
    ax.set_title(f'Original vs Imputed Values: {column}', fontweight='bold')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel(column)
    plt.tight_layout()
    plt.show()


def visualize_missing_pattern(df):
    """Visualize the pattern of missing values in a DataFrame."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    missing_matrix = df.isna().astype(int)
    sns.heatmap(missing_matrix.head(100), cmap='RdYlBu_r', ax=axes[0], yticklabels=False)
    axes[0].set_title('Missing Value Pattern (Sample)')

    missing_counts = df.isna().sum()
    missing_pcts = (missing_counts / len(df)) * 100
    colors = ['steelblue' if pct < 10 else 'orange' if pct < 25 else 'red' for pct in missing_pcts]
    axes[1].bar(range(len(missing_counts)), missing_pcts, color=colors, alpha=0.7)
    axes[1].set_xticks(range(len(missing_counts)))
    axes[1].set_xticklabels(missing_counts.index, rotation=45)
    axes[1].set_ylabel('Missing (%)')
    axes[1].set_title('Missing Values by Column')

    plt.suptitle('Missing Data Analysis', fontweight='bold')
    plt.tight_layout()
    plt.show()


def print_imputation_summary(df_original, df_imputed, method_name):
    """Print a text summary of imputation results."""
    print(f"\n{'='*60}")
    print(f"Imputation Summary: {method_name}")
    print(f"{'='*60}")
    original_missing = df_original.isna().sum().sum()
    remaining_missing = df_imputed.isna().sum().sum()
    print(f"Original missing: {original_missing}, Remaining: {remaining_missing}")
    print(f"Values imputed: {original_missing - remaining_missing}")
    print(f"{'='*60}\n")


def compare_imputation_distributions(
    df_complete: pd.DataFrame,
    imputed_dict: Dict[str, pd.DataFrame],
    missing_df: pd.DataFrame,
    max_cols: int = 4,
) -> None:
    """
    Compare how each imputation strategy preserves the ground-truth distribution.

    For every numeric column (up to *max_cols*) a KDE row is drawn that shows:
    * the ground-truth complete distribution (dashed black)
    * the observed (non-missing) distribution
    * each imputed distribution

    Args:
        df_complete: Ground-truth DataFrame with no missing values (numeric only).
        imputed_dict: Mapping of strategy name -> fully-imputed DataFrame.
        missing_df: The DataFrame that contains NaN values (pre-imputation).
        max_cols: Maximum number of feature columns to plot.
    """
    columns = [c for c in df_complete.columns if c in missing_df.columns][:max_cols]
    n_cols = len(columns)
    if n_cols == 0:
        print("No overlapping numeric columns found.")
        return

    strategies = list(imputed_dict.keys())
    palette = sns.color_palette("tab10", n_colors=len(strategies))

    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), constrained_layout=True)
    if n_cols == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        # Ground truth
        df_complete[col].plot.kde(ax=ax, color='black', linestyle='--', linewidth=2,
                                   label='Ground truth')
        # Observed (non-missing)
        observed = missing_df[col].dropna()
        observed.plot.kde(ax=ax, color='grey', linestyle=':', linewidth=1.5,
                          label='Observed')
        # Each imputed strategy
        for color, (name, df_imp) in zip(palette, imputed_dict.items()):
            if col in df_imp.columns:
                df_imp[col].plot.kde(ax=ax, color=color, linewidth=2, label=name)

        ax.set_title(col, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Imputation Strategy: Distribution Comparison vs Ground Truth',
                 fontsize=13, fontweight='bold')
    plt.show()
