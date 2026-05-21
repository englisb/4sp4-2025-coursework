import sys
import csv
import numpy as np
import os

FOLDER = './data/model/'

def magnitude_prune(weight_matrix, sparsity_level):
    """
    Prune a weight matrix using magnitude-based pruning.
    
    Args:
        weight_matrix: numpy array of weights
        sparsity_level: float between 0 and 1, percentage of weights to remove
    
    Returns:
        Pruned weight matrix (same shape, with zeros for pruned weights)
    """
    # Calculate how many weights to keep
    keep_ratio = 1.0 - sparsity_level
    
    # Flatten the matrix to find the threshold
    flat_weights = np.abs(weight_matrix.flatten())
    
    # Calculate the threshold: keep the top (1 - sparsity_level) percent of weights
    # Percentile: if sparsity is 0.8, we keep top 20%, so threshold is at 20th percentile
    threshold = np.percentile(flat_weights, sparsity_level * 100)
    
    # Create a mask: keep weights with absolute value >= threshold
    mask = np.abs(weight_matrix) >= threshold
    
    # Apply mask: zero out weights below threshold
    pruned_matrix = weight_matrix * mask
    
    return pruned_matrix

def save_matrix_to_csv(matrix, filepath):
    """
    Save a numpy matrix to CSV file in dense format.
    
    Args:
        matrix: numpy array to save
        filepath: path to save the CSV file
    """
    np.savetxt(filepath, matrix, delimiter=',', fmt='%.6f')

# Load weights from CSV files
print('Loading weight matrices...')
if not os.path.exists(FOLDER):
    print(f'Error: {FOLDER} not found. Please ensure the model directory exists.')
    sys.exit(1)

W1_path = os.path.join(FOLDER, 'weights_hidden.csv')
W2_path = os.path.join(FOLDER, 'weights_output.csv')

if not os.path.exists(W1_path):
    print(f'Error: {W1_path} not found.')
    sys.exit(1)
if not os.path.exists(W2_path):
    print(f'Error: {W2_path} not found.')
    sys.exit(1)

# Load weight matrices
W1 = np.loadtxt(W1_path, delimiter=',')
W2 = np.loadtxt(W2_path, delimiter=',')

print(f'W1 shape: {W1.shape}')
print(f'W2 shape: {W2.shape}')

# Prune for sparsity levels from 50% to 95% in steps of 5%
sparsity_levels = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

print('\nPruning weight matrices...')
for sparsity in sparsity_levels:
    sparsity_percent = int(sparsity * 100)
    print(f'Processing {sparsity_percent}% sparsity...')
    
    # Prune W1 and W2
    W1_pruned = magnitude_prune(W1, sparsity)
    W2_pruned = magnitude_prune(W2, sparsity)
    
    # Calculate actual sparsity (percentage of zeros)
    W1_actual_sparsity = np.sum(W1_pruned == 0) / W1_pruned.size * 100
    W2_actual_sparsity = np.sum(W2_pruned == 0) / W2_pruned.size * 100
    
    print(f'  W1 actual sparsity: {W1_actual_sparsity:.2f}%')
    print(f'  W2 actual sparsity: {W2_actual_sparsity:.2f}%')
    
    # Save pruned matrices
    W1_filename = os.path.join(FOLDER, f'{sparsity_percent}_W1.csv')
    W2_filename = os.path.join(FOLDER, f'{sparsity_percent}_W2.csv')
    
    save_matrix_to_csv(W1_pruned, W1_filename)
    save_matrix_to_csv(W2_pruned, W2_filename)
    
    print(f'  Saved: {W1_filename}')
    print(f'  Saved: {W2_filename}')

print('\nPruning complete!')

