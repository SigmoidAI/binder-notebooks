"""Helper utilities for Imputation Impact Assessment exercises."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats


def generate_complete_dataset(n_samples=500, n_features=5, noise_level=0.1, random_state=42):
    """
    Generates a synthetic dataset without any missing values.

    Args:
        n_samples (int): Number of samples to generate (default: 500)
        n_features (int): Number of features to generate (default: 5)
        noise_level (float): Amount of noise to add to the target (default: 0.1)
        random_state (int): Random seed for reproducibility (default: 42)

    Returns:
        tuple: (X, y) where X is a DataFrame with features and y is a Series with target
    """
    np.random.seed(random_state)
    X_data = {}
    X_data['feature_0'] = np.random.normal(50, 10, n_samples)
    X_data['feature_1'] = np.random.uniform(0, 100, n_samples)
    X_data['feature_2'] = np.random.exponential(20, n_samples) + 10
    mask = np.random.random(n_samples) > 0.5
    X_data['feature_3'] = np.where(mask, np.random.normal(30, 5, n_samples), np.random.normal(70, 5, n_samples))
    X_data['feature_4'] = np.random.normal(100, 25, n_samples)
    for i in range(5, n_features):
        X_data[f'feature_{i}'] = np.random.normal(50, 15, n_samples)
    X = pd.DataFrame(X_data)
    coefficients = [2.0, 0.5, -1.5, 0.8, 0.3] + [0.2] * (n_features - 5)
    y = np.zeros(n_samples)
    for i, col in enumerate(X.columns):
        y += coefficients[i] * X[col].values
    noise = np.random.normal(0, noise_level * np.std(y), n_samples)
    y = y + noise
    y = pd.Series(y, name='target')
    return X, y


def introduce_missing_values(X, missing_rate=0.2, missing_type='MCAR', random_state=42):
    """
    Introduces missing values into a dataset in a controlled manner.

    Args:
        X (pd.DataFrame): The complete dataset
        missing_rate (float): Proportion of values to make missing (default: 0.2)
        missing_type (str): Type of missingness mechanism ('MCAR', 'MAR', 'MNAR')
        random_state (int): Random seed for reproducibility

    Returns:
        pd.DataFrame: Dataset with missing values introduced
    """
    np.random.seed(random_state)
    X_missing = X.copy()
    n_samples, n_features = X.shape
    if missing_type == 'MCAR':
        mask = np.random.random((n_samples, n_features)) < missing_rate
        for i in range(n_samples):
            if mask[i].sum() == n_features:
                mask[i, np.random.randint(n_features)] = False
        X_missing = X_missing.mask(mask)
    elif missing_type == 'MAR':
        for col_idx in range(1, n_features):
            col = X.columns[col_idx]
            probs = (X.iloc[:, 0] - X.iloc[:, 0].min()) / (X.iloc[:, 0].max() - X.iloc[:, 0].min())
            probs = probs * missing_rate * 2
            probs = np.clip(probs, 0, 0.8)
            mask = np.random.random(n_samples) < probs
            X_missing.loc[mask, col] = np.nan
    elif missing_type == 'MNAR':
        for col in X.columns:
            z_scores = np.abs(stats.zscore(X[col]))
            probs = z_scores / z_scores.max() * missing_rate * 2
            probs = np.clip(probs, 0, 0.8)
            mask = np.random.random(n_samples) < probs
            X_missing.loc[mask, col] = np.nan
    return X_missing


def train_evaluate_model(X_train, X_test, y_train, y_test, model_type='linear'):
    """
    Trains and evaluates a regression model.

    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        y_train (pd.Series): Training target
        y_test (pd.Series): Test target
        model_type (str): Type of model to use ('linear' or 'random_forest')

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=50, random_state=42)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'predictions': y_pred}


def plot_performance_comparison(results_dict, metric='rmse', figsize=(10, 6)):
    """
    Creates a bar chart comparing model performance across different imputation strategies.

    Args:
        results_dict (dict): Dictionary where keys are strategy names and values contain metrics
        metric (str): Which metric to plot ('rmse', 'mae', or 'r2')
        figsize (tuple): Figure size

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    strategies = list(results_dict.keys())
    values = [results_dict[s][metric] for s in strategies]
    colors = ['#2ecc71' if s == 'baseline' else '#3498db' for s in strategies]
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(strategies, values, color=colors, edgecolor='black', linewidth=1.2)
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
    metric_labels = {'rmse': 'Root Mean Squared Error (RMSE)', 'mae': 'Mean Absolute Error (MAE)', 'r2': 'R-squared Score (R2)'}
    ax.set_xlabel('Imputation Strategy', fontsize=12)
    ax.set_ylabel(metric_labels.get(metric, metric), fontsize=12)
    ax.set_title(f'Model Performance Comparison: {metric.upper()}', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    if 'baseline' in results_dict:
        baseline_val = results_dict['baseline'][metric]
        ax.axhline(y=baseline_val, color='#e74c3c', linestyle='--', linewidth=2, label=f'Baseline: {baseline_val:.4f}')
        ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    return fig


def plot_distribution_shift(original_data, imputed_data, feature_name, figsize=(12, 5)):
    """
    Compares distributions before and after imputation using histograms and KDE plots.

    Args:
        original_data (pd.Series): Original complete data for a feature
        imputed_data (pd.Series): Data after imputation
        feature_name (str): Name of the feature being compared
        figsize (tuple): Figure size

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax1 = axes[0]
    ax1.hist(original_data, bins=30, alpha=0.6, label='Original', color='#3498db', density=True, edgecolor='black')
    ax1.hist(imputed_data, bins=30, alpha=0.6, label='After Imputation', color='#e74c3c', density=True, edgecolor='black')
    ax1.set_xlabel(feature_name, fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('Distribution Comparison (Histogram)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2 = axes[1]
    original_data.plot.kde(ax=ax2, label='Original', color='#3498db', linewidth=2)
    imputed_data.plot.kde(ax=ax2, label='After Imputation', color='#e74c3c', linewidth=2)
    ax2.set_xlabel(feature_name, fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Distribution Comparison (KDE)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    orig_mean, orig_std = original_data.mean(), original_data.std()
    imp_mean, imp_std = imputed_data.mean(), imputed_data.std()
    stats_text = f'Original: mean={orig_mean:.2f}, std={orig_std:.2f}\nImputed: mean={imp_mean:.2f}, std={imp_std:.2f}'
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.show()
    return fig


def calculate_distribution_metrics(original_data, imputed_data):
    """
    Calculates metrics to quantify distribution shift after imputation.

    Args:
        original_data (pd.Series or np.ndarray): Original complete data
        imputed_data (pd.Series or np.ndarray): Data after imputation

    Returns:
        dict: Dictionary containing distribution shift metrics
    """
    orig = np.array(original_data).flatten()
    imp = np.array(imputed_data).flatten()
    orig_mean, imp_mean = np.mean(orig), np.mean(imp)
    orig_var, imp_var = np.var(orig, ddof=1), np.var(imp, ddof=1)
    orig_std, imp_std = np.std(orig, ddof=1), np.std(imp, ddof=1)
    mean_shift = np.abs(imp_mean - orig_mean)
    mean_shift_pct = (mean_shift / np.abs(orig_mean)) * 100 if orig_mean != 0 else 0
    variance_change = np.abs(imp_var - orig_var)
    variance_change_pct = (variance_change / orig_var) * 100 if orig_var != 0 else 0
    std_change = np.abs(imp_std - orig_std)
    ks_stat, ks_pvalue = stats.ks_2samp(orig, imp)
    return {
        'mean_shift': mean_shift,
        'mean_shift_pct': mean_shift_pct,
        'variance_change': variance_change,
        'variance_change_pct': variance_change_pct,
        'std_change': std_change,
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pvalue
    }


def plot_missing_pattern(X_missing, figsize=(10, 6)):
    """
    Visualizes the pattern of missing values in a dataset.

    Args:
        X_missing (pd.DataFrame): Dataset with missing values
        figsize (tuple): Figure size

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax1 = axes[0]
    missing_matrix = X_missing.isnull().astype(int)
    sns.heatmap(missing_matrix.head(50), cmap='YlOrRd', cbar_kws={'label': 'Missing'}, ax=ax1, yticklabels=False)
    ax1.set_title('Missing Value Pattern (First 50 rows)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Features')
    ax1.set_ylabel('Samples')
    ax2 = axes[1]
    missing_pct = (X_missing.isnull().sum() / len(X_missing)) * 100
    bars = ax2.bar(missing_pct.index, missing_pct.values, color='#e74c3c', edgecolor='black')
    ax2.set_xlabel('Features', fontsize=11)
    ax2.set_ylabel('Missing Percentage (%)', fontsize=11)
    ax2.set_title('Missing Values by Feature', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(missing_pct.values) * 1.2 if max(missing_pct.values) > 0 else 1)
    plt.xticks(rotation=45, ha='right')
    for bar, pct in zip(bars, missing_pct.values):
        ax2.annotate(f'{pct:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    ax2.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig


def create_impact_summary_table(results_dict, baseline_key='baseline'):
    """
    Creates a summary DataFrame comparing all imputation strategies.

    Args:
        results_dict (dict): Dictionary of results from each strategy
        baseline_key (str): Key for the baseline results

    Returns:
        pd.DataFrame: Summary table with metrics and comparisons
    """
    rows = []
    baseline = results_dict.get(baseline_key, None)
    for strategy, metrics in results_dict.items():
        row = {'Strategy': strategy, 'RMSE': metrics['rmse'], 'MAE': metrics['mae'], 'R2': metrics['r2']}
        if baseline and strategy != baseline_key:
            row['RMSE_vs_baseline_%'] = ((metrics['rmse'] - baseline['rmse']) / baseline['rmse']) * 100
            row['MAE_vs_baseline_%'] = ((metrics['mae'] - baseline['mae']) / baseline['mae']) * 100
            row['R2_vs_baseline_%'] = ((metrics['r2'] - baseline['r2']) / baseline['r2']) * 100
        else:
            row['RMSE_vs_baseline_%'] = 0.0
            row['MAE_vs_baseline_%'] = 0.0
            row['R2_vs_baseline_%'] = 0.0
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df.set_index('Strategy')
    return df


def plot_impact_comparison(report: pd.DataFrame, figsize=(14, 5)) -> None:
    """
    Visualise the imputation impact report produced by ``create_impact_report``.

    Creates two side-by-side bar charts:
    * RMSE per strategy (lower is better)
    * R² per strategy   (higher is better)
    A horizontal dashed line marks the baseline value in each chart.

    Args:
        report: DataFrame with at minimum columns
                ``['strategy', 'rmse', 'r2', 'n_samples']``,
                as returned by ``create_impact_report``.
        figsize: Overall figure size.
    """
    strategies = report['strategy'].tolist()
    colors = ['#2ecc71' if s == 'baseline' else '#3498db' for s in strategies]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, metric, label, better in zip(
        axes,
        ['rmse', 'r2'],
        ['RMSE (lower is better)', 'R² (higher is better)'],
        ['lower', 'higher'],
    ):
        values = report[metric].tolist()
        bars = ax.bar(strategies, values, color=colors, edgecolor='black', linewidth=1.1)

        # Annotate bars
        for bar, val in zip(bars, values):
            ax.annotate(
                f'{val:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords='offset points',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
            )

        # Baseline reference line
        baseline_rows = report[report['strategy'] == 'baseline']
        if not baseline_rows.empty:
            baseline_val = baseline_rows.iloc[0][metric]
            ax.axhline(baseline_val, color='#e74c3c', linestyle='--', linewidth=1.8,
                       label=f'Baseline: {baseline_val:.4f}')
            ax.legend(fontsize=9)

        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Strategy', fontsize=11)
        ax.set_ylabel(metric.upper(), fontsize=11)
        ax.yaxis.grid(True, linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    # Add n_samples info per strategy as a subtitle
    sample_info = '  |  '.join(
        f"{row['strategy']}: n={row['n_samples']}" for _, row in report.iterrows()
    )
    fig.suptitle(
        f'Imputation Impact Assessment\n{sample_info}',
        fontsize=11, fontweight='bold',
    )
    plt.tight_layout()
    plt.show()


def print_impact_report(summary_df, distribution_metrics=None):
    """
    Prints a formatted impact assessment report.

    Args:
        summary_df (pd.DataFrame): Summary table from create_impact_summary_table
        distribution_metrics (dict, optional): Distribution metrics for each strategy
    """
    print("=" * 70)
    print("IMPUTATION IMPACT ASSESSMENT REPORT")
    print("=" * 70)
    print("\n--- Model Performance Metrics ---\n")
    formatted = summary_df.copy()
    for col in ['RMSE', 'MAE']:
        formatted[col] = formatted[col].apply(lambda x: f"{x:.4f}")
    formatted['R2'] = formatted['R2'].apply(lambda x: f"{x:.4f}")
    for col in ['RMSE_vs_baseline_%', 'MAE_vs_baseline_%', 'R2_vs_baseline_%']:
        formatted[col] = formatted[col].apply(lambda x: f"{x:+.2f}%")
    print(formatted.to_string())
    print("\n--- Key Findings ---\n")
    best_rmse = summary_df['RMSE'].idxmin()
    worst_rmse = summary_df['RMSE'].idxmax()
    print(f"Best performing strategy (lowest RMSE): {best_rmse}")
    print(f"Worst performing strategy (highest RMSE): {worst_rmse}")
    if distribution_metrics:
        print("\n--- Distribution Shift Analysis ---\n")
        for strategy, metrics in distribution_metrics.items():
            print(f"{strategy}:")
            print(f"  Mean shift: {metrics['mean_shift']:.4f} ({metrics['mean_shift_pct']:.2f}%)")
            print(f"  Variance change: {metrics['variance_change']:.4f} ({metrics['variance_change_pct']:.2f}%)")
            print(f"  KS statistic: {metrics['ks_statistic']:.4f} (p-value: {metrics['ks_pvalue']:.4f})")
    print("\n" + "=" * 70)
