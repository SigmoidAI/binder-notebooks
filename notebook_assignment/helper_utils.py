"""
Helper utilities for the Missing Value Audit assignment.

This module provides visualization functions and sample data generation
to help understand missing value patterns in datasets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_sample_data_with_missing(n_rows=100, random_state=42):
    """
    Generate a sample DataFrame with various missing value patterns.

    This creates a realistic dataset with different types of missingness:
    - MCAR (Missing Completely At Random): age column
    - MAR (Missing At Random): income depends on employment
    - Structural missingness: years_experience for students

    Args:
        n_rows (int): Number of rows to generate
        random_state (int): Random seed for reproducibility

    Returns:
        pd.DataFrame: Sample dataset with missing values
    """
    np.random.seed(random_state)

    # Generate base data
    data = {
        'customer_id': range(1, n_rows + 1),
        'age': np.random.randint(18, 70, n_rows).astype(float),
        'income': np.random.normal(50000, 20000, n_rows),
        'years_experience': np.random.randint(0, 30, n_rows).astype(float),
        'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_rows),
        'employment_status': np.random.choice(['Employed', 'Unemployed', 'Student', 'Retired'], n_rows),
        'satisfaction_score': np.random.uniform(1, 10, n_rows),
        'num_purchases': np.random.poisson(5, n_rows),
    }

    df = pd.DataFrame(data)

    # Introduce MCAR missingness in age (random 10%)
    mcar_mask = np.random.random(n_rows) < 0.10
    df.loc[mcar_mask, 'age'] = np.nan

    # Introduce MAR missingness in income (higher for unemployed/students)
    for idx in df.index:
        if df.loc[idx, 'employment_status'] in ['Unemployed', 'Student']:
            if np.random.random() < 0.4:
                df.loc[idx, 'income'] = np.nan
        elif np.random.random() < 0.05:
            df.loc[idx, 'income'] = np.nan

    # Structural missingness: years_experience is NaN for students
    df.loc[df['employment_status'] == 'Student', 'years_experience'] = np.nan

    # Random missingness in satisfaction_score (15%)
    sat_mask = np.random.random(n_rows) < 0.15
    df.loc[sat_mask, 'satisfaction_score'] = np.nan

    # Some random missingness in education_level (5%)
    edu_mask = np.random.random(n_rows) < 0.05
    df.loc[edu_mask, 'education_level'] = np.nan

    return df


def plot_missing_heatmap(df, figsize=(12, 8), cmap='YlOrRd'):
    """
    Visualize missing value patterns as a heatmap.

    Each row represents an observation and each column represents a variable.
    Missing values are highlighted, making patterns easy to identify.

    Args:
        df (pd.DataFrame): Input DataFrame
        figsize (tuple): Figure size (width, height)
        cmap (str): Colormap for the heatmap

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create a boolean mask for missing values
    missing_mask = df.isnull().astype(int)

    # Create heatmap
    sns.heatmap(
        missing_mask,
        cmap=cmap,
        cbar_kws={'label': 'Missing (1) vs Present (0)'},
        yticklabels=False,
        ax=ax
    )

    ax.set_xlabel('Columns', fontsize=12)
    ax.set_ylabel('Rows (Observations)', fontsize=12)
    ax.set_title('Missing Value Heatmap\n(Yellow/Red = Missing, Dark = Present)', fontsize=14)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

    return fig


def plot_missing_bar(df, figsize=(10, 6), show_percentage=True):
    """
    Create a bar chart showing missing value counts/percentages per column.

    Args:
        df (pd.DataFrame): Input DataFrame
        figsize (tuple): Figure size (width, height)
        show_percentage (bool): If True, show percentages; otherwise show counts

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate missing values
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100

    # Sort by missing count (descending)
    sorted_idx = missing_count.sort_values(ascending=True).index

    if show_percentage:
        values = missing_pct[sorted_idx]
        xlabel = 'Missing Value Percentage (%)'
        title = 'Missing Value Percentage by Column'
    else:
        values = missing_count[sorted_idx]
        xlabel = 'Missing Value Count'
        title = 'Missing Value Count by Column'

    # Create color gradient based on missingness
    colors = plt.cm.RdYlGn_r(values / max(values.max(), 1))

    bars = ax.barh(range(len(values)), values, color=colors, edgecolor='black', alpha=0.8)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        if show_percentage:
            label = f'{val:.1f}%'
        else:
            label = f'{int(val)}'
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                label, va='center', fontsize=10)

    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(sorted_idx)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Columns', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(axis='x', alpha=0.3)

    # Set x-axis limit to accommodate labels
    max_val = values.max()
    ax.set_xlim(0, max_val * 1.15 if max_val > 0 else 1)

    plt.tight_layout()
    plt.show()

    return fig


def plot_missing_matrix(df, figsize=(14, 8)):
    """
    Create a matrix visualization showing where values are missing.

    This provides a detailed view of missing data patterns across the dataset,
    with observations on the y-axis and variables on the x-axis.

    Args:
        df (pd.DataFrame): Input DataFrame
        figsize (tuple): Figure size (width, height)

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create visualization data
    # 1 = present (will be light), 0 = missing (will be dark)
    vis_data = df.notnull().astype(int).values

    # Use imshow for efficient rendering
    cax = ax.imshow(vis_data, aspect='auto', cmap='gray', interpolation='none')

    ax.set_xlabel('Columns', fontsize=12)
    ax.set_ylabel('Row Index', fontsize=12)
    ax.set_title('Missing Data Matrix\n(White = Present, Black = Missing)', fontsize=14)

    # Set column labels
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha='right')

    # Add colorbar
    cbar = plt.colorbar(cax, ax=ax, shrink=0.8)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Missing', 'Present'])

    plt.tight_layout()
    plt.show()

    return fig


def plot_missing_correlation(df, figsize=(10, 8)):
    """
    Plot correlation between missing value indicators.

    This helps identify if missingness in one column is related to
    missingness in another column.

    Args:
        df (pd.DataFrame): Input DataFrame
        figsize (tuple): Figure size (width, height)

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create missing indicator DataFrame
    missing_indicators = df.isnull().astype(int)

    # Only include columns that have at least one missing value
    cols_with_missing = missing_indicators.columns[missing_indicators.sum() > 0]

    if len(cols_with_missing) < 2:
        ax.text(0.5, 0.5, 'Not enough columns with missing values\nto compute correlation',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_title('Missing Value Correlation Matrix', fontsize=14)
        plt.show()
        return fig

    missing_corr = missing_indicators[cols_with_missing].corr()

    # Create heatmap
    mask = np.triu(np.ones_like(missing_corr, dtype=bool), k=1)
    sns.heatmap(
        missing_corr,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax
    )

    ax.set_title('Correlation Between Missing Value Patterns', fontsize=14)

    plt.tight_layout()
    plt.show()

    return fig


def summarize_missing_patterns(df):
    """
    Print a summary of missing value patterns in the dataset.

    Args:
        df (pd.DataFrame): Input DataFrame
    """
    total_cells = df.size
    total_missing = df.isnull().sum().sum()
    total_pct = (total_missing / total_cells) * 100

    rows_with_missing = df.isnull().any(axis=1).sum()
    complete_rows = len(df) - rows_with_missing

    cols_with_missing = df.isnull().any(axis=0).sum()
    complete_cols = len(df.columns) - cols_with_missing

    print("=" * 50)
    print("MISSING VALUE SUMMARY")
    print("=" * 50)
    print(f"\nDataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Total Cells: {total_cells:,}")
    print(f"Total Missing: {total_missing:,} ({total_pct:.2f}%)")
    print(f"\nRows with Missing Values: {rows_with_missing} ({rows_with_missing/len(df)*100:.1f}%)")
    print(f"Complete Rows: {complete_rows} ({complete_rows/len(df)*100:.1f}%)")
    print(f"\nColumns with Missing Values: {cols_with_missing}")
    print(f"Complete Columns: {complete_cols}")
    print("=" * 50)

    print("\nMissing by Column:")
    print("-" * 40)
    missing_info = pd.DataFrame({
        'Missing Count': df.isnull().sum(),
        'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
    }).sort_values('Missing Count', ascending=False)
    print(missing_info.to_string())


def display_missing_dataframe(df, max_rows=10):
    """
    Display a styled DataFrame highlighting missing values.

    Args:
        df (pd.DataFrame): Input DataFrame
        max_rows (int): Maximum number of rows to display
    """
    def highlight_missing(val):
        if pd.isna(val):
            return 'background-color: #ffcccc'
        return ''

    display_df = df.head(max_rows)
    styled = display_df.style.applymap(highlight_missing)
    return styled
