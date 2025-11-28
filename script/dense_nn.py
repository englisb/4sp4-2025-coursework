import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def tanh(x):
    """Hyperbolic tangent activation function"""
    return np.tanh(x)


def sigmoid(x):
    """Sigmoid activation function: 1/(1 + exp(-x))"""
    return 1 / (1 + np.exp(-x))


def forward_pass(X, W1, b1, W2, b2):
    """
    Forward pass through the dense neural network.
    
    Args:
        X: Input matrix (batch_size, n)
        W1: Weight matrix for first layer (h, n)
        b1: Bias vector for first layer (h,)
        W2: Weight matrix for second layer (m, h)
        b2: Bias vector for second layer (m,)
    
    Returns:
        Y: Predictions (batch_size,)
    """
    # Hidden layer: H = tanh(X * W1^T + b1)
    H = tanh(X @ W1.T + b1)
    
    # Output layer: Z = sigmoid(H * W2^T + b2)
    Z = sigmoid(H @ W2.T + b2)
    
    # Prediction: Y = argmax(Z)
    Y = np.argmax(Z, axis=1)
    
    return Y


# Read MNIST Dataset in from CSV file
print('Reading CSV Data...')
data_path = './data/mnist_train.csv'
if not os.path.exists(data_path):
    print(f'Error: {data_path} not found. Please ensure the data directory exists.')
    exit(1)

# Use pandas to read the CSV file
data = pd.read_csv(data_path)

# Preprocess Data using pandas operations
# Labels are in the first column, features are in the rest
labels_df = data.iloc[:, [0]]  # Keep as DataFrame first
features_df = data.iloc[:, 1:]  # Keep as DataFrame

# Normalize features to [0, 1] range (MNIST pixel values are 0-255) using pandas
features_df = features_df / 255.0

# Convert to numpy arrays for neural network operations
labels = labels_df.values.flatten().astype(np.int32)
features = features_df.values.astype(np.float32)

print(f'Loaded {data.shape[0]} samples with {features_df.shape[1]} features')
print(f'Data shape: {data.shape}, Features shape: {features_df.shape}, Labels shape: {labels_df.shape}')

# Load weights + Biases
model_dir = './data/model'
if not os.path.exists(model_dir):
    print(f'Error: {model_dir} not found. Please ensure the model directory exists.')
    exit(1)

print('Loading model weights and biases...')
# Use pandas to read all weight and bias files
W1_df = pd.read_csv(f'{model_dir}/weights_hidden.csv', header=None)
b1_df = pd.read_csv(f'{model_dir}/biases_hidden.csv', header=None)
W2_df = pd.read_csv(f'{model_dir}/weights_output.csv', header=None)
b2_df = pd.read_csv(f'{model_dir}/biases_output.csv', header=None)

# Convert to numpy arrays for neural network operations
W1 = W1_df.values.astype(np.float32)
b1 = b1_df.values.flatten().astype(np.float32)
W2 = W2_df.values.astype(np.float32)
b2 = b2_df.values.flatten().astype(np.float32)

print(f'W1 shape: {W1_df.shape}, b1 shape: {b1_df.shape}')
print(f'W2 shape: {W2_df.shape}, b2 shape: {b2_df.shape}')

# Measure accuracy on test set using forward propagation for all MNIST data
print('\nRunning forward pass on all samples...')
predictions = forward_pass(features, W1, b1, W2, b2)

# Calculate accuracy using pandas for comparison
predictions_series = pd.Series(predictions)
labels_series = pd.Series(labels)
correct_mask = predictions_series == labels_series
correct_predictions = correct_mask.sum()
total_samples = len(labels)
accuracy = (correct_predictions / total_samples) * 100.0

# Create a results DataFrame using pandas
results_df = pd.DataFrame({
    'True_Label': labels_series,
    'Predicted_Label': predictions_series,
    'Correct': correct_mask
})

print(f'\nResults:')
print(f'Correct predictions: {correct_predictions} / {total_samples}')
print(f'Total Accuracy: {accuracy:.2f}%')
print(f'\nConfusion Matrix Summary:')
print(results_df['Correct'].value_counts())

# Optionally visualize some of the results using matplotlib
print('\nGenerating visualization...')
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
fig.suptitle('Sample Predictions (Top: Correct, Bottom: Incorrect)', fontsize=14)

# Find some correct and incorrect predictions using pandas
correct_indices = results_df[results_df['Correct']].index[:5].values
incorrect_indices = results_df[~results_df['Correct']].index[:5].values

# Display 5 correct predictions (green)
for i in range(min(5, len(correct_indices))):
    idx = correct_indices[i]
    img = features[idx].reshape(28, 28)
    axes[0, i].imshow(img, cmap='gray')
    axes[0, i].set_title(f'True: {labels[idx]}, Pred: {predictions[idx]}', color='green', fontweight='bold')
    # Add green border
    for spine in axes[0, i].spines.values():
        spine.set_edgecolor('green')
        spine.set_linewidth(3)
    axes[0, i].axis('off')

# Display 5 incorrect predictions (red)
for i in range(min(5, len(incorrect_indices))):
    idx = incorrect_indices[i]
    img = features[idx].reshape(28, 28)
    axes[1, i].imshow(img, cmap='gray')
    axes[1, i].set_title(f'True: {labels[idx]}, Pred: {predictions[idx]}', color='red', fontweight='bold')
    # Add red border
    for spine in axes[1, i].spines.values():
        spine.set_edgecolor('red')
        spine.set_linewidth(3)
    axes[1, i].axis('off')

# Hide unused subplots
for i in range(len(correct_indices), 5):
    axes[0, i].axis('off')
for i in range(len(incorrect_indices), 5):
    axes[1, i].axis('off')

plt.tight_layout()
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/dense_nn_predictions.png', dpi=150, bbox_inches='tight')
print('Visualization saved to plots/dense_nn_predictions.png')

