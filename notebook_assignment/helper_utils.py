"""Helper utilities for Production-Ready Imputation Pipeline."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Any


def generate_train_test_data(
    n_train: int = 200,
    n_test: int = 50,
    missing_rate_train: float = 0.15,
    missing_rate_test: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate train/test split with different missing patterns."""
    np.random.seed(random_state)
    n_total = n_train + n_test
    
    age = np.random.normal(40, 12, n_total).clip(18, 80)
    income = np.random.lognormal(10.5, 0.5, n_total)
    experience = np.random.exponential(8, n_total).clip(0, 40)
    score = np.random.uniform(300, 850, n_total)
    education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'],
                                  n_total, p=[0.3, 0.4, 0.2, 0.1])
    region = np.random.choice(['North', 'South', 'East', 'West'], n_total)
    
    data = pd.DataFrame({
        'age': age, 'income': income, 'experience': experience,
        'credit_score': score, 'education': education, 'region': region
    })
    
    prob = 1 / (1 + np.exp(-(0.02 * (age - 40) + 0.00001 * (income - 40000) +
                             0.05 * experience - 0.005 * (score - 600))))
    target = (np.random.random(n_total) < prob).astype(int)
    
    X_train = data.iloc[:n_train].copy()
    X_test = data.iloc[n_train:].copy()
    y_train = pd.DataFrame({'target': target[:n_train]})
    y_test = pd.DataFrame({'target': target[n_train:]})
    
    _introduce_missing(X_train, missing_rate_train, random_state)
    _introduce_missing(X_test, missing_rate_test, random_state + 1)
    
    return X_train, X_test.reset_index(drop=True), y_train, y_test.reset_index(drop=True)


def _introduce_missing(df: pd.DataFrame, rate: float, seed: int) -> None:
    """Introduce missing values into a DataFrame in-place."""
    np.random.seed(seed)
    for col in df.columns:
        mask = np.random.random(len(df)) < rate
        if df[col].dtype == 'object':
            df.loc[mask, col] = None
        else:
            df.loc[mask, col] = np.nan


def simulate_new_data(
    n_samples: int = 30,
    include_edge_cases: bool = True,
    random_state: int = 123
) -> pd.DataFrame:
    """Generate new data with edge cases for testing imputation robustness."""
    np.random.seed(random_state)
    
    data = pd.DataFrame({
        'age': np.random.normal(42, 10, n_samples).clip(18, 80),
        'income': np.random.lognormal(10.5, 0.6, n_samples),
        'experience': np.random.exponential(7, n_samples).clip(0, 35),
        'credit_score': np.random.uniform(320, 800, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples)
    })
    
    if include_edge_cases:
        high_missing_mask = np.random.random(n_samples) < 0.7
        data.loc[high_missing_mask, 'experience'] = np.nan
        
        if n_samples > 5:
            data.loc[0, 'education'] = 'Doctorate'
            data.loc[1, 'region'] = 'Central'
        
        if n_samples > 10:
            data.loc[5, 'age'] = 95
            data.loc[6, 'income'] = 1000000
            data.loc[7, 'credit_score'] = 250
        
        if n_samples > 15:
            data.loc[10, ['age', 'income', 'experience']] = np.nan
            data.loc[11, ['education', 'region']] = None
        
        if n_samples > 20:
            data.loc[15, ['age', 'income', 'experience', 'credit_score']] = np.nan
    
    return data


def plot_pipeline_flow(
    pipeline_steps: List[Tuple[str, str]],
    figsize: Tuple[int, int] = (14, 6)
) -> None:
    """Visualize the pipeline steps in a flow diagram."""
    fig, ax = plt.subplots(figsize=figsize)
    
    n_steps = len(pipeline_steps)
    box_width = 0.15
    box_height = 0.3
    spacing = (1 - n_steps * box_width) / (n_steps + 1)
    
    colors = plt.cm.Blues(np.linspace(0.3, 0.8, n_steps))
    
    for i, ((name, desc), color) in enumerate(zip(pipeline_steps, colors)):
        x = spacing + i * (box_width + spacing)
        y = 0.35
        
        rect = plt.Rectangle((x, y), box_width, box_height,
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        ax.text(x + box_width/2, y + box_height/2, name,
               ha='center', va='center', fontsize=11, fontweight='bold')
        ax.text(x + box_width/2, y - 0.08, desc,
               ha='center', va='top', fontsize=9, style='italic')
        
        if i < n_steps - 1:
            arrow_start = x + box_width
            arrow_end = x + box_width + spacing
            ax.annotate('', xy=(arrow_end, y + box_height/2),
                       xytext=(arrow_start, y + box_height/2),
                       arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Pipeline Flow Diagram', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()


def validate_pipeline_output(
    original: pd.DataFrame,
    transformed: np.ndarray,
    expected_cols: int = None
) -> Dict[str, Any]:
    """Check that pipeline output is valid and report statistics."""
    results = {
        'is_valid': True,
        'issues': [],
        'statistics': {}
    }
    
    if isinstance(transformed, pd.DataFrame):
        transformed = transformed.values
    
    nan_count = np.isnan(transformed).sum() if np.issubdtype(transformed.dtype, np.number) else 0
    results['statistics']['nan_count'] = int(nan_count)
    if nan_count > 0:
        results['is_valid'] = False
        results['issues'].append(f"Found {nan_count} NaN values in output")
    
    if np.issubdtype(transformed.dtype, np.number):
        inf_count = np.isinf(transformed).sum()
        results['statistics']['inf_count'] = int(inf_count)
        if inf_count > 0:
            results['is_valid'] = False
            results['issues'].append(f"Found {inf_count} infinite values in output")
    
    results['statistics']['shape'] = transformed.shape
    results['statistics']['original_shape'] = original.shape
    
    if expected_cols is not None and transformed.shape[1] != expected_cols:
        results['is_valid'] = False
        results['issues'].append(f"Expected {expected_cols} columns, got {transformed.shape[1]}")
    
    if transformed.shape[0] != len(original):
        results['is_valid'] = False
        results['issues'].append(f"Row count mismatch: expected {len(original)}, got {transformed.shape[0]}")
    
    return results


def save_load_pipeline_demo(pipeline, filepath: str = 'demo_pipeline.joblib') -> None:
    """Demonstrate saving and loading a pipeline with joblib."""
    import joblib
    import tempfile
    import os
    
    temp_dir = tempfile.gettempdir()
    full_path = os.path.join(temp_dir, filepath)
    
    print("=" * 60)
    print("PIPELINE SAVE/LOAD DEMONSTRATION")
    print("=" * 60)
    
    print(f"\n1. Saving pipeline to: {full_path}")
    joblib.dump(pipeline, full_path)
    file_size = os.path.getsize(full_path)
    print(f"   File size: {file_size / 1024:.2f} KB")
    
    print(f"\n2. Loading pipeline from: {full_path}")
    loaded_pipeline = joblib.load(full_path)
    print(f"   Pipeline type: {type(loaded_pipeline).__name__}")
    
    print(f"\n3. Verifying pipeline structure:")
    if hasattr(loaded_pipeline, 'steps'):
        for name, step in loaded_pipeline.steps:
            print(f"   - {name}: {type(step).__name__}")
    else:
        print(f"   - Single transformer: {type(loaded_pipeline).__name__}")
    
    os.remove(full_path)
    print(f"\n4. Cleaned up demo file")
    print("=" * 60)


def visualize_missing_patterns(df: pd.DataFrame, figsize: Tuple[int, int] = (12, 6)) -> None:
    """Visualize missing value patterns in a DataFrame."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    ax = axes[0]
    sample_size = min(50, len(df))
    missing_matrix = df.head(sample_size).isnull().astype(int)
    sns.heatmap(missing_matrix, cbar=True, yticklabels=False, cmap='YlOrRd',
                ax=ax, vmin=0, vmax=1)
    ax.set_title(f'Missing Value Pattern (First {sample_size} rows)', fontsize=12)
    ax.set_xlabel('Features')
    
    ax = axes[1]
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=True)
    colors = ['green' if x < 10 else 'orange' if x < 30 else 'red' for x in missing_pct]
    missing_pct.plot(kind='barh', ax=ax, color=colors)
    ax.set_xlabel('Missing Percentage (%)')
    ax.set_title('Missing Values by Feature', fontsize=12)
    ax.axvline(x=10, color='green', linestyle='--', alpha=0.7, label='10% threshold')
    ax.axvline(x=30, color='red', linestyle='--', alpha=0.7, label='30% threshold')
    ax.legend()
    
    plt.tight_layout()
    plt.show()


def create_sample_pipeline_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create simple sample data for demonstrating pipeline concepts."""
    np.random.seed(42)
    n = 100
    
    X = pd.DataFrame({
        'numeric_1': np.random.normal(50, 10, n),
        'numeric_2': np.random.exponential(20, n),
        'category_1': np.random.choice(['A', 'B', 'C'], n),
        'category_2': np.random.choice(['X', 'Y'], n)
    })
    
    X.loc[np.random.choice(n, 15, replace=False), 'numeric_1'] = np.nan
    X.loc[np.random.choice(n, 10, replace=False), 'numeric_2'] = np.nan
    X.loc[np.random.choice(n, 8, replace=False), 'category_1'] = None
    
    y = pd.DataFrame({
        'target': (X['numeric_1'].fillna(50) + X['numeric_2'].fillna(20) > 75).astype(int)
    })
    
    return X, y


def print_imputation_summary(imputer, column_names: List[str]) -> None:
    """Print a summary of learned imputation values."""
    print("=" * 50)
    print("IMPUTATION SUMMARY")
    print("=" * 50)
    
    if not hasattr(imputer, 'statistics_'):
        print("Imputer has not been fitted yet!")
        return
    
    stats = imputer.statistics_
    strategy = getattr(imputer, 'strategy', 'unknown')
    
    print(f"\nStrategy: {strategy}")
    print(f"Number of features: {len(stats)}")
    print("\nLearned fill values:")
    print("-" * 50)
    
    for col, val in zip(column_names, stats):
        if isinstance(val, float):
            print(f"  {col:20s}: {val:.4f}")
        else:
            print(f"  {col:20s}: {val}")
    print("=" * 50)


def demonstrate_fit_transform_difference() -> None:
    """Demonstrate the difference between fit, transform, and fit_transform."""
    print("=" * 70)
    print("FIT vs TRANSFORM vs FIT_TRANSFORM")
    print("=" * 70)
    print("""
    fit(X_train):
    - LEARNS parameters from training data
    - For imputers: learns the fill values (mean, median, mode)
    - Does NOT modify the data
    - Returns: the fitted estimator itself

    transform(X):
    - APPLIES the learned parameters to data
    - Uses values learned during fit()
    - Can be applied to train, test, or new data
    - Returns: transformed data

    fit_transform(X_train):
    - Combines fit() and transform() in one step
    - Only use on TRAINING data!
    - More efficient than calling fit() then transform()
    - Returns: transformed training data

    CORRECT WORKFLOW:
    1. X_train_imp = imputer.fit_transform(X_train)  # Fit + transform train
    2. X_test_imp = imputer.transform(X_test)        # Transform test only

    WRONG (Data Leakage):
    X_test_imp = imputer.fit_transform(X_test)  # DON'T DO THIS!
    """)
    print("=" * 70)
