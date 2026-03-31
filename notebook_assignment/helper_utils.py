"""
Helper utilities for the Basic Statistics assignment.

This module provides visualization functions to help understand
statistical concepts like variance, covariance, and correlation.
"""

import numpy as np
import matplotlib.pyplot as plt


def visualize_variance(data):
    """
    Visualize variance by showing data points and their deviations from the mean.
    
    Args:
        data (np.ndarray): 1D array of data points
    """
    mean = np.mean(data)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Data points with mean line
    ax = axes[0]
    ax.scatter(range(len(data)), data, s=100, c='blue', alpha=0.7, label='Data points')
    ax.axhline(y=mean, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean:.2f}')
    
    # Draw deviation lines
    for i, val in enumerate(data):
        ax.plot([i, i], [mean, val], color='green', linestyle='-', alpha=0.5)
    
    ax.set_xlabel('Index', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Data Points and Deviations from Mean', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Right plot: Squared deviations
    ax = axes[1]
    squared_devs = (data - mean) ** 2
    colors = ['green' if d >= 0 else 'red' for d in (data - mean)]
    ax.bar(range(len(data)), squared_devs, color=colors, alpha=0.7)
    ax.axhline(y=np.mean(squared_devs), color='orange', linestyle='--', 
               linewidth=2, label=f'Variance = {np.var(data):.2f}')
    ax.set_xlabel('Index', fontsize=12)
    ax.set_ylabel('Squared Deviation', fontsize=12)
    ax.set_title('Squared Deviations from Mean', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def visualize_covariance():
    """
    Visualize positive, negative, and zero covariance with scatter plots.
    """
    np.random.seed(42)
    n = 50
    
    # Generate data with different covariance patterns
    x = np.linspace(0, 10, n)
    
    # Positive covariance
    y_pos = 2 * x + np.random.normal(0, 2, n)
    
    # Negative covariance
    y_neg = -1.5 * x + 15 + np.random.normal(0, 2, n)
    
    # Zero covariance (independent)
    y_zero = np.random.normal(5, 2, n)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    datasets = [
        (x, y_pos, 'Positive Covariance', 'blue'),
        (x, y_neg, 'Negative Covariance', 'red'),
        (x, y_zero, 'Near-Zero Covariance', 'green')
    ]
    
    for ax, (x_data, y_data, title, color) in zip(axes, datasets):
        ax.scatter(x_data, y_data, c=color, alpha=0.6, s=50)
        
        # Add trend line
        z = np.polyfit(x_data, y_data, 1)
        p = np.poly1d(z)
        ax.plot(x_data, p(x_data), color='black', linestyle='--', linewidth=2)
        
        cov = np.cov(x_data, y_data)[0, 1]
        ax.set_title(f'{title}\nCov = {cov:.2f}', fontsize=12)
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def visualize_correlation():
    """
    Visualize different correlation strengths with scatter plots.
    """
    np.random.seed(42)
    n = 100
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    correlations = [1.0, 0.8, 0.4, 0.0, -0.5, -1.0]
    titles = ['Perfect Positive (r=1)', 'Strong Positive (r≈0.8)', 
              'Moderate Positive (r≈0.4)', 'No Correlation (r≈0)',
              'Moderate Negative (r≈-0.5)', 'Perfect Negative (r=-1)']
    
    for ax, target_corr, title in zip(axes, correlations, titles):
        # Generate correlated data
        x = np.random.randn(n)
        
        if abs(target_corr) == 1:
            y = target_corr * x
        else:
            # Create correlated data using Cholesky decomposition
            noise = np.random.randn(n)
            y = target_corr * x + np.sqrt(1 - target_corr**2) * noise
        
        actual_corr = np.corrcoef(x, y)[0, 1]
        
        # Color based on sign of correlation
        if target_corr > 0:
            color = 'blue'
        elif target_corr < 0:
            color = 'red'
        else:
            color = 'green'
        
        ax.scatter(x, y, c=color, alpha=0.5, s=30)
        ax.set_title(f'{title}\nActual r = {actual_corr:.3f}', fontsize=11)
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add trend line for non-zero correlations
        if abs(target_corr) > 0.1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), color='black', linestyle='--', linewidth=2)
    
    plt.tight_layout()
    plt.show()


def plot_histogram(data, title="Distribution", bins=20):
    """
    Plot a histogram with mean and std lines.
    
    Args:
        data (np.ndarray): 1D array of data
        title (str): Plot title
        bins (int): Number of histogram bins
    """
    mean = np.mean(data)
    std = np.std(data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(data, bins=bins, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Add mean line
    ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean:.2f}')
    
    # Add std lines
    ax.axvline(mean - std, color='orange', linestyle=':', linewidth=2, label=f'Mean ± Std')
    ax.axvline(mean + std, color='orange', linestyle=':', linewidth=2)
    
    # Add 2*std lines
    ax.axvline(mean - 2*std, color='green', linestyle=':', linewidth=1.5, label=f'Mean ± 2*Std')
    ax.axvline(mean + 2*std, color='green', linestyle=':', linewidth=1.5)
    
    ax.set_xlabel('Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{title}\nMean = {mean:.2f}, Std = {std:.2f}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def compare_distributions(data1, data2, label1="Data 1", label2="Data 2"):
    """
    Compare two distributions side by side.
    
    Args:
        data1 (np.ndarray): First dataset
        data2 (np.ndarray): Second dataset
        label1 (str): Label for first dataset
        label2 (str): Label for second dataset
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, data, label in zip(axes, [data1, data2], [label1, label2]):
        mean = np.mean(data)
        std = np.std(data)
        
        ax.hist(data, bins=25, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean:.2f}')
        ax.axvline(mean - std, color='orange', linestyle=':', linewidth=2)
        ax.axvline(mean + std, color='orange', linestyle=':', linewidth=2, label=f'Std = {std:.2f}')
        
        ax.set_xlabel('Value', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{label}\nMean = {mean:.2f}, Std = {std:.2f}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def scatter_with_stats(x, y, xlabel='X', ylabel='Y'):
    """
    Create a scatter plot with statistics annotations.
    
    Args:
        x (np.ndarray): X values
        y (np.ndarray): Y values
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(x, y, alpha=0.6, s=50, c='steelblue', edgecolors='black')
    
    # Trend line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), color='red', linestyle='--', linewidth=2, label='Trend line')
    
    # Add mean lines
    ax.axvline(np.mean(x), color='green', linestyle=':', alpha=0.7, label=f'Mean X = {np.mean(x):.2f}')
    ax.axhline(np.mean(y), color='orange', linestyle=':', alpha=0.7, label=f'Mean Y = {np.mean(y):.2f}')
    
    # Statistics text box
    corr = np.corrcoef(x, y)[0, 1]
    cov = np.cov(x, y)[0, 1]
    stats_text = f'Correlation: {corr:.4f}\nCovariance: {cov:.4f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'Scatter Plot: {xlabel} vs {ylabel}', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def print_statistics(data, name="Data"):
    """
    Print comprehensive statistics for an array.
    
    Args:
        data (np.ndarray): Input array
        name (str): Name to display
    """
    print(f"=== Statistics for {name} ===")
    print(f"Count: {len(data)}")
    print(f"Mean: {np.mean(data):.4f}")
    print(f"Median: {np.median(data):.4f}")
    print(f"Std (population): {np.std(data, ddof=0):.4f}")
    print(f"Std (sample): {np.std(data, ddof=1):.4f}")
    print(f"Variance (population): {np.var(data, ddof=0):.4f}")
    print(f"Variance (sample): {np.var(data, ddof=1):.4f}")
    print(f"Min: {np.min(data):.4f}")
    print(f"Max: {np.max(data):.4f}")
    print(f"Range: {np.ptp(data):.4f}")
    print()
