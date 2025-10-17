import numpy as np
from sklearn.metrics import accuracy_score
from pathlib import Path
import os

# --- Configuration ---
LR = 0.000005  # Reduced learning rate for stability due to large parameter count
EPOCHS = 200   # Increased epochs for better convergence
REPORT_INTERVAL = 20

# --- File Paths ---
PROCESSED_DATA_FILE = "data/processed_data.npz"
OUTPUTS_DIR = "outputs"
PRED_PROB_FILE = Path(OUTPUTS_DIR) / "y_pred_prob.npy"
Y_TEST_FILE = Path(OUTPUTS_DIR) / "y_test.npy"
MODEL_WEIGHTS_FILE = Path(OUTPUTS_DIR) / "model_weights.npz"


def sigmoid(z):
    """Sigmoid activation function."""
    # Clip z to prevent overflow in np.exp()
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def calculate_loss(y_true, y_pred_prob):
    """Binary Cross-Entropy Loss."""
    # Clip probabilities to prevent log(0)
    y_pred_prob = np.clip(y_pred_prob, 1e-9, 1 - 1e-9)
    loss = -np.mean(y_true * np.log(y_pred_prob) + (1 - y_true) * np.log(1 - y_pred_prob))
    return loss


# --- 1. Load Data ---
print("📊 Loading processed data...")
if not Path(PROCESSED_DATA_FILE).exists():
    print(f"Error: Data file not found at {PROCESSED_DATA_FILE}. Please run 'preprocess.py' first.")
    exit()

data = np.load(PROCESSED_DATA_FILE, allow_pickle=True)
X, y = data["X"], data["y"]

# Train-test split (80/20)
n_train = int(0.8 * len(X))
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

print(f"Data Loaded. Training samples: {len(y_train)}, Test samples: {len(y_test)}")


# --- 2. Initialize Parameters ---
n_features = X_train.shape[1]
# Initialize S, A, and w with small random values
S = np.random.randn(n_features, n_features) * 1e-3
A = np.random.randn(n_features, n_features) * 1e-3
w = np.random.randn(n_features) * 1e-3


# --- 3. Training Loop with Full Gradient Descent ---
print(f"🧠 Starting training for {EPOCHS} epochs (LR: {LR}, Model features: {n_features})")
m_train = len(y_train)

for epoch in range(1, EPOCHS + 1):
    # Forward Pass: Calculate logits for the Quadratic Model
    # Logit = (X @ w) + (X @ (S + A) @ X.T) -> simplified using np.einsum for batch processing
    quadratic_term = np.einsum('bi,ij,bj->b', X_train, S + A, X_train)
    logits = X_train @ w + quadratic_term
    preds_train = sigmoid(logits)
    
    error = preds_train - y_train
    
    # Calculate Gradients
    
    # 1. Gradient for linear term (w)
    grad_w = X_train.T @ error / m_train
    
    # 2. Gradient for quadratic term (G = S + A)
    # The term G_total is the gradient for the combined (S + A) matrix
    G_total = X_train.T @ np.diag(error) @ X_train / m_train
    
    # 3. Separate G_total into S (Symmetric) and A (Anti-symmetric) parts
    grad_S = 0.5 * (G_total + G_total.T) # S is symmetric: (G + G.T) / 2
    grad_A = 0.5 * (G_total - G_total.T) # A is anti-symmetric: (G - G.T) / 2
    
    # Parameter Update
    w -= LR * grad_w
    S -= LR * grad_S
    A -= LR * grad_A
    
    # Reporting
    if epoch % REPORT_INTERVAL == 0 or epoch == EPOCHS:
        train_loss = calculate_loss(y_train, preds_train)
        train_acc = accuracy_score(y_train, np.round(preds_train))
        print(f"Epoch {epoch:4d}/{EPOCHS} | Loss: {train_loss:.6f} | Train Acc: {train_acc:.4f}")

print("\n✅ Training complete.")

# --- 4. Evaluate on Test Set and Save Outputs ---

# Test set forward pass
quadratic_term_test = np.einsum('bi,ij,bj->b', X_test, S + A, X_test)
logits_test = X_test @ w + quadratic_term_test
preds_test = sigmoid(logits_test)
y_pred_binary = np.round(preds_test)

# Final evaluation
test_acc = accuracy_score(y_test, y_pred_binary)
print(f"📈 Final Test Accuracy: {test_acc:.4f}")

# Save results for the dashboard
Path(OUTPUTS_DIR).mkdir(exist_ok=True)

# Save predicted probabilities (needed for the dashboard's smooth analysis)
np.save(PRED_PROB_FILE, preds_test)
np.save(Y_TEST_FILE, y_test)

# Save model weights (optional, but good practice)
np.savez(MODEL_WEIGHTS_FILE, w=w, S=S, A=A)

print(f"📁 Saved predicted probabilities to '{PRED_PROB_FILE}'")
print(f"📁 Saved true test labels to '{Y_TEST_FILE}'")
print(f"📁 Saved final model weights to '{MODEL_WEIGHTS_FILE}'")
