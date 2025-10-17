import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
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

# Split data (80% train, 20% test)
n_train = int(0.8 * len(X))
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"✅ Baseline Accuracy: {acc:.4f}")
print("Confusion Matrix:\n", cm)

# Create outputs directory if missing
Path("outputs").mkdir(exist_ok=True)

# Save predictions and true labels
np.save("outputs/y_pred.npy", y_pred)
np.save("outputs/y_test.npy", y_test)

print("📁 Saved predictions to 'outputs/y_pred.npy' and 'outputs/y_test.npy' for evaluation.")
