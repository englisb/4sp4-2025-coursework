import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. The Neural Network Definition
# ==========================================
class MnistNet(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super(MnistNet, self).__init__()
        # Layer 1 definition (W1 is hidden_dim x input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # Layer 2 definition (W2 is output_dim x hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def load_csv_weights(self, w_hidden_path, b_hidden_path, w_out_path, b_out_path):
        """
        Loads weights from CSV files into the PyTorch layers.
        Assumes CSVs have no headers.
        """
        try:
            # Load as numpy arrays
            w1 = pd.read_csv(w_hidden_path, header=None).values
            b1 = pd.read_csv(b_hidden_path, header=None).values.flatten()
            w2 = pd.read_csv(w_out_path, header=None).values
            b2 = pd.read_csv(b_out_path, header=None).values.flatten()

            # Assign to PyTorch layers
            # Note: PyTorch Linear stores weights as (out_features, in_features)
            # Ensure your CSV shape matches, or transpose here if necessary.
            self.fc1.weight.data = torch.tensor(w1, dtype=torch.float32)
            self.fc1.bias.data = torch.tensor(b1, dtype=torch.float32)
            self.fc2.weight.data = torch.tensor(w2, dtype=torch.float32)
            self.fc2.bias.data = torch.tensor(b2, dtype=torch.float32)
            print("Weights loaded successfully.")
        except FileNotFoundError as e:
            print(f"Error loading CSVs: {e}")
            print("Initializing with random weights for demonstration purposes...")

    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        
        # Layer 1: H = tanh(X * W1^T + b1)
        h = torch.tanh(self.fc1(x))
        
        # Layer 2: Z = sigmoid(H * W2^T + b2)
        z = torch.sigmoid(self.fc2(h))
        
        # Argmax is handled during evaluation, raw Z returned here
        return z, h  # Returning h to capture inputs for the next layer pruning

# ==========================================
# 2. CSV Dataset Loader
# ==========================================
class CSVDataset(Dataset):
    """Simple dataset class for loading MNIST from CSV"""
    def __init__(self, csv_path, normalize=True):
        # MNIST CSV has a header row (e.g., 'label', 'pixel0', ...)
        # Let pandas use the first row as header, then treat the rest as numeric data.
        data = pd.read_csv(csv_path)

        # First column is labels, rest are features
        # Explicitly cast to numeric dtypes to avoid mixed-type issues.
        self.labels = data.iloc[:, 0].astype(np.int64).values
        self.features = data.iloc[:, 1:].astype(np.float32).values
        
        # Normalize features to [0, 1] range (MNIST pixel values are 0-255)
        if normalize:
            self.features = self.features / 255.0
        
        # Convert to tensors
        self.features = torch.from_numpy(self.features)
        self.labels = torch.from_numpy(self.labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ==========================================
# 3. SparseGPT Pruner Implementation
# ==========================================
class SparseGPTPruner:
    def __init__(self, layer, sparsity=0.9):
        self.layer = layer
        self.sparsity = sparsity
        self.dev = layer.weight.device
        
        # Dimensions
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]
        
        # Hessian (Covariance Matrix of Inputs)
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, input_data):
        """
        Calibration step: Accumulate X * X^T to approximate the Hessian.
        """
        # Ensure input is 2D: (batch_size, features)
        if len(input_data.shape) == 1:
            input_data = input_data.unsqueeze(0)
        elif len(input_data.shape) > 2:
            input_data = input_data.view(-1, input_data.shape[-1])
            
        # Flatten to (batch_size, features) if needed
        tmp = input_data.view(-1, self.columns)
        self.H += torch.matmul(tmp.t(), tmp)
        self.nsamples += tmp.shape[0]

    def prune(self):
        """
        Executes the SparseGPT algorithm.
        1. Normalizes and inverts the Hessian.
        2. Prunes weights with lowest saliency.
        3. Updates remaining weights to compensate for error.
        """
        W = self.layer.weight.data.clone()
        
        # Normalize Hessian by number of samples to get covariance matrix
        if self.nsamples > 0:
            H = self.H / self.nsamples
        else:
            H = self.H
        
        # Add dampening for numerical stability on inversion
        damp = 0.01 * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        
        # Invert Hessian (Using Cholesky usually, but standard inv is safer for general cases)
        try:
            H_inv = torch.linalg.inv(H)
        except:
            print("Hessian non-invertible, using pseudo-inverse")
            H_inv = torch.linalg.pinv(H)
        
        # Quantize/Prune column by column (or blocks)
        # In SparseGPT for LLMs, they iterate columns. 
        # Here we calculate the pruning mask based on the diagonal of H_inv.
        
        # "Saliency" metric: w^2 / [H^-1]_ii
        # We process the whole matrix at once for this small scale (unlike massive GPTs)
        
        diag_H_inv = torch.diag(H_inv) # (in_features)
        
        # Prepare the mask
        # We want to prune the smallest elements.
        # W_metric = W^2 / diag_H_inv
        
        # Reshape for broadcasting
        W_metric = W.pow(2) / diag_H_inv.reshape(1, -1) 
        
        # Determine threshold for global sparsity
        # If sparsity=0.5, we want to keep top 50%, so threshold at (1-sparsity) quantile
        keep_ratio = 1.0 - self.sparsity
        threshold = torch.quantile(W_metric, keep_ratio)
        mask = W_metric >= threshold
        
        # SparseGPT Update Rule: 
        # W_new = W - (W_pruned / H_inv_ii) * H_inv_row
        
        # Since we are doing this "one-shot" for the whole layer rather than 
        # column-iterative (which is strictly needed for quantization but less so for pure pruning),
        # we apply the compensation logic.
        
        # Create the mask (1 = keep, 0 = prune)
        Q = torch.zeros_like(W)
        Q[mask] = 1.0
        
        # Apply mask to prune weights
        W_pruned = W * Q
        
        # SparseGPT weight update: compensate for pruned weights using Optimal Brain Surgeon (OBS) approximation
        # This is a simplified version - full SparseGPT does iterative column-wise updates
        # For each output neuron (row), update remaining weights to compensate for pruned ones
        W_new = W_pruned.clone()
        for i in range(self.rows):
            # Find which weights were pruned in this row
            pruned_mask_row = ~mask[i]
            if pruned_mask_row.any() and mask[i].any():  # Need both pruned and kept weights
                # Get pruned weight values
                w_pruned_vals = W[i, pruned_mask_row]
                # Get indices
                pruned_indices = torch.where(pruned_mask_row)[0]
                kept_indices = torch.where(mask[i])[0]
                
                # Compensation: delta_W = -H_inv[kept, pruned] @ inv(H_inv[pruned, pruned]) @ w_pruned
                try:
                    H_inv_pruned_block = H_inv[pruned_indices, :][:, pruned_indices]
                    H_inv_cross = H_inv[kept_indices, :][:, pruned_indices]
                    
                    if H_inv_pruned_block.numel() > 0:
                        # Add small regularization for numerical stability
                        reg = 1e-8 * torch.eye(H_inv_pruned_block.shape[0], device=self.dev)
                        H_inv_pruned_inv = torch.linalg.inv(H_inv_pruned_block + reg)
                        # Compute compensation
                        compensation = -torch.matmul(H_inv_cross, torch.matmul(H_inv_pruned_inv, w_pruned_vals))
                        W_new[i, kept_indices] += compensation
                except Exception as e:
                    # If compensation fails (e.g., singular matrix), just use masked weights
                    pass
        
        self.layer.weight.data = W_new
        
        # Explicit check for sparsity
        current_sparsity = 1.0 - (torch.count_nonzero(self.layer.weight.data) / self.layer.weight.data.numel())
        print(f"Layer Pruned. Target: {self.sparsity:.2f}, Achieved: {current_sparsity:.2f}")

# ==========================================
# 4. Execution Pipeline
# ==========================================
def run_pipeline():
    # Settings
    SPARSITY_RATIO = 0.10  # Remove 50% of weights
    CALIBRATION_SAMPLES = 128
    
    # 1. Load Data from CSV file
    csv_path = './data/mnist_train.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please ensure the data directory exists.")
        return
    
    print(f"Loading data from {csv_path}...")
    dataset = CSVDataset(csv_path, normalize=True)
    print(f"Loaded {len(dataset)} samples")
    
    # Create data loaders
    test_loader = DataLoader(dataset, batch_size=1000, shuffle=False)
    
    # Calibration loader (subset)
    calib_loader = DataLoader(dataset, batch_size=CALIBRATION_SAMPLES, shuffle=True)
    
    # 2. Initialize Model & Load Weights
    model = MnistNet()
    # Try loading from data/model folder first, then current directory
    model_dir = './data/model'
    if os.path.exists(model_dir):
        model.load_csv_weights(
            os.path.join(model_dir, 'weights_hidden.csv'),
            os.path.join(model_dir, 'biases_hidden.csv'),
            os.path.join(model_dir, 'weights_output.csv'),
            os.path.join(model_dir, 'biases_output.csv')
        )
    else:
        model.load_csv_weights(
            'weights_hidden.csv', 'biases_hidden.csv', 
            'weights_output.csv', 'biases_output.csv'
        )
    
    # 3. Calibration & Pruning Phase
    print("\n--- Starting SparseGPT Pruning ---")
    
    # Get calibration data
    calib_data, _ = next(iter(calib_loader))
    # Data from CSV is already flattened (784 features), no need to reshape
    
    # --- Prune Layer 1 ---
    pruner1 = SparseGPTPruner(model.fc1, sparsity=SPARSITY_RATIO)
    pruner1.add_batch(calib_data) # Compute Hessian for Layer 1 inputs (Raw X)
    pruner1.prune()
    
    # --- Prune Layer 2 ---
    # We must forward the calibration data through the (now pruned) Layer 1
    # to get the accurate inputs for Layer 2.
    with torch.no_grad():
        h_calib = torch.tanh(model.fc1(calib_data))
        
    pruner2 = SparseGPTPruner(model.fc2, sparsity=SPARSITY_RATIO)
    pruner2.add_batch(h_calib) # Compute Hessian for Layer 2 inputs (H)
    pruner2.prune()
    
    # 4. Evaluation
    print("\n--- Evaluating Performance ---")
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            z, _ = model(data)
            # Y = argmax(Z)
            predicted = torch.argmax(z, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy after SparseGPT Pruning: {accuracy:.2f}%')
    
    # 5. Save Pruned Weights to sparseGPT folder
    print("\n--- Saving Pruned Weights ---")
    sparsegpt_folder = './sparseGPT'
    os.makedirs(sparsegpt_folder, exist_ok=True)
    
    # Convert to numpy and save
    w1_np = model.fc1.weight.data.cpu().numpy()
    b1_np = model.fc1.bias.data.cpu().numpy()
    w2_np = model.fc2.weight.data.cpu().numpy()
    b2_np = model.fc2.bias.data.cpu().numpy()
    
    # Save weights and biases
    np.savetxt(os.path.join(sparsegpt_folder, 'weights_hidden.csv'), w1_np, delimiter=',', fmt='%.6f')
    np.savetxt(os.path.join(sparsegpt_folder, 'biases_hidden.csv'), b1_np, delimiter=',', fmt='%.6f')
    np.savetxt(os.path.join(sparsegpt_folder, 'weights_output.csv'), w2_np, delimiter=',', fmt='%.6f')
    np.savetxt(os.path.join(sparsegpt_folder, 'biases_output.csv'), b2_np, delimiter=',', fmt='%.6f')
    
    print(f"Saved pruned weights to {sparsegpt_folder}/")
    print(f"  - weights_hidden.csv (shape: {w1_np.shape})")
    print(f"  - biases_hidden.csv (shape: {b1_np.shape})")
    print(f"  - weights_output.csv (shape: {w2_np.shape})")
    print(f"  - biases_output.csv (shape: {b2_np.shape})")
    
    # Calculate and print sparsity statistics
    w1_sparsity = 1.0 - (np.count_nonzero(w1_np) / w1_np.size)
    w2_sparsity = 1.0 - (np.count_nonzero(w2_np) / w2_np.size)
    print(f"\nFinal Sparsity:")
    print(f"  Layer 1 (hidden): {w1_sparsity:.2%}")
    print(f"  Layer 2 (output): {w2_sparsity:.2%}")

if __name__ == "__main__":
    run_pipeline()