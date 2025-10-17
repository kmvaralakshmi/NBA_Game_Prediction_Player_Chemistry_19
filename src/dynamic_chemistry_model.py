import numpy as np
from sklearn.metrics import accuracy_score
from pathlib import Path
import os

# --- Load processed data using the standard, consistent path ---
processed_data_path = "data/processed_data.npz"

try:
    data = np.load(processed_data_path, allow_pickle=True)
except FileNotFoundError:
    print(f"❌ Error: Preprocessed data '{processed_data_path}' not found. Please run 'python src/preprocess.py' first.")
    exit()

X, y = data["X"], data["y"]

# Dynamic chemistry weighting
weights = np.linspace(0.5, 1.5, X.shape[0])
X_weighted = X * weights[:, None]

# Graph-based quadratic term
n_features = X.shape[1]
S = np.random.randn(n_features, n_features) * 0.01
A = np.random.randn(n_features, n_features) * 0.01
w = np.random.randn(n_features) * 0.01

lr = 0.0005
epochs = 80

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

print(f"Starting Dynamic QCM training with {n_features} features...")
for epoch in range(epochs):
    # Forward pass uses weighted features
    logits = X_weighted @ w + np.einsum('bi,ij,bj->b', X_weighted, S + A, X_weighted)
    preds = sigmoid(logits)
    
    # Gradients use weighted features
    error = preds - y
    grad_w = X_weighted.T @ error / len(y)
    grad_Q = (X_weighted.T @ (error[:, None] * X_weighted) * 2) / len(y)

    # Update parameters
    w -= lr * grad_w
    S_A_combined = S + A
    S_A_combined -= lr * grad_Q
    
    S = (S_A_combined + S_A_combined.T) / 2
    A = (S_A_combined - S_A_combined.T) / 2
    
    if epoch % 10 == 0:
        loss = -np.mean(y*np.log(preds+1e-9) + (1-y)*np.log(1-preds+1e-9))
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

print("\n✅ Dynamic QCM training complete.")

# Create outputs directory if missing
Path("outputs").mkdir(exist_ok=True)

# Save final parameters
np.savez("outputs/dynamic_qcm_params.npz", w=w, S=S, A=A)

print("📁 Saved Dynamic QCM parameters to 'outputs/dynamic_qcm_params.npz'.")
