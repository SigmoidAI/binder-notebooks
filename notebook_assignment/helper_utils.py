"""
Helper utilities for the NumPy Array Operations assignment.

This module provides visualization functions to help understand
NumPy concepts like broadcasting and array operations.
"""

import numpy as np
import matplotlib.pyplot as plt


def visualize_broadcasting():
    """
    Visualize how broadcasting works with a 2D array and a 1D array.
    Shows the conceptual expansion of the smaller array.
    """
    # Create example arrays
    matrix = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
    row = np.array([10, 20, 30])
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Plot the matrix
    ax = axes[0]
    im = ax.imshow(matrix, cmap='Blues')
    ax.set_title('Matrix (3x3)', fontsize=12)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center', fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Plot the row vector
    ax = axes[1]
    row_expanded = row.reshape(1, -1)
    ax.imshow(row_expanded, cmap='Oranges', aspect='auto')
    ax.set_title('Row Vector (1x3)', fontsize=12)
    for j in range(3):
        ax.text(j, 0, str(row[j]), ha='center', va='center', fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Show broadcasting concept
    ax = axes[2]
    broadcast_visual = np.tile(row, (3, 1))
    ax.imshow(broadcast_visual, cmap='Oranges')
    ax.set_title('Broadcast (conceptual)', fontsize=12)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(row[j]), ha='center', va='center', fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Show result
    ax = axes[3]
    result = matrix + row
    ax.imshow(result, cmap='Greens')
    ax.set_title('Matrix + Row = Result', fontsize=12)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(result[i, j]), ha='center', va='center', fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()


def visualize_array(arr, title="Array"):
    """
    Visualize a 1D or 2D NumPy array as a heatmap with values shown.
    
    Args:
        arr (np.ndarray): Array to visualize (1D or 2D)
        title (str): Title for the plot
    """
    # Convert 1D to 2D for visualization
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    
    fig, ax = plt.subplots(figsize=(max(arr.shape[1], 4), max(arr.shape[0], 2)))
    
    im = ax.imshow(arr, cmap='viridis', aspect='auto')
    ax.set_title(f'{title}\nShape: {arr.shape}', fontsize=12)
    
    # Add text annotations
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = arr[i, j]
            # Format based on value type
            if isinstance(value, (int, np.integer)):
                text = str(value)
            else:
                text = f'{value:.2f}'
            ax.text(j, i, text, ha='center', va='center', 
                   color='white' if arr[i, j] < arr.mean() else 'black', fontsize=10)
    
    ax.set_xticks(range(arr.shape[1]))
    ax.set_yticks(range(arr.shape[0]))
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()


def compare_arrays(arr1, arr2, title1="Array 1", title2="Array 2"):
    """
    Visualize two arrays side by side for comparison.
    
    Args:
        arr1 (np.ndarray): First array
        arr2 (np.ndarray): Second array
        title1 (str): Title for first array
        title2 (str): Title for second array
    """
    # Convert 1D to 2D for visualization
    if arr1.ndim == 1:
        arr1 = arr1.reshape(1, -1)
    if arr2.ndim == 1:
        arr2 = arr2.reshape(1, -1)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    for ax, arr, title in zip(axes, [arr1, arr2], [title1, title2]):
        im = ax.imshow(arr, cmap='viridis', aspect='auto')
        ax.set_title(f'{title}\nShape: {arr.shape}', fontsize=12)
        
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                value = arr[i, j]
                if isinstance(value, (int, np.integer)):
                    text = str(value)
                else:
                    text = f'{value:.2f}'
                ax.text(j, i, text, ha='center', va='center',
                       color='white' if arr[i, j] < arr.mean() else 'black', fontsize=10)
        
        ax.set_xticks(range(arr.shape[1]))
        ax.set_yticks(range(arr.shape[0]))
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()


def visualize_matrix_multiplication(A, B):
    """
    Visualize matrix multiplication A @ B with intermediate steps.
    
    Args:
        A (np.ndarray): First matrix
        B (np.ndarray): Second matrix
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Cannot multiply matrices of shapes {A.shape} and {B.shape}")
    
    C = A @ B
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot A
    ax = axes[0]
    im = ax.imshow(A, cmap='Blues')
    ax.set_title(f'A\n{A.shape}', fontsize=12)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax.text(j, i, f'{A[i, j]:.1f}', ha='center', va='center', fontsize=10)
    ax.set_xlabel('Columns')
    ax.set_ylabel('Rows')
    
    # Plot B
    ax = axes[1]
    im = ax.imshow(B, cmap='Oranges')
    ax.set_title(f'B\n{B.shape}', fontsize=12)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            ax.text(j, i, f'{B[i, j]:.1f}', ha='center', va='center', fontsize=10)
    ax.set_xlabel('Columns')
    ax.set_ylabel('Rows')
    
    # Plot C = A @ B
    ax = axes[2]
    im = ax.imshow(C, cmap='Greens')
    ax.set_title(f'C = A @ B\n{C.shape}', fontsize=12)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            ax.text(j, i, f'{C[i, j]:.1f}', ha='center', va='center', fontsize=10)
    ax.set_xlabel('Columns')
    ax.set_ylabel('Rows')
    
    plt.tight_layout()
    plt.show()
    
    return C


def print_array_info(arr, name="Array"):
    """
    Print detailed information about a NumPy array.
    
    Args:
        arr (np.ndarray): Array to inspect
        name (str): Name to display
    """
    print(f"=== {name} ===")
    print(f"Shape: {arr.shape}")
    print(f"Dimensions: {arr.ndim}")
    print(f"Data type: {arr.dtype}")
    print(f"Size (total elements): {arr.size}")
    print(f"Memory (bytes): {arr.nbytes}")
    if arr.size > 0:
        print(f"Min: {arr.min():.4f}, Max: {arr.max():.4f}")
        print(f"Mean: {arr.mean():.4f}, Std: {arr.std():.4f}")
    print()
