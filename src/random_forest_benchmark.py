# src/random_forest_benchmark.py

import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path

# Load processed data
# Assumes data/processed_data.npz contains X (features) and y (targets)
data = np.load("data/processed_data.npz", allow_pickle=True)
X, y = data["X"], data["y"]

# Split data (80% train, 20% test)
n_train = int(0.8 * len(X))
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

# Initialize and train Random Forest Classifier
# This is a robust non-linear model for benchmarking the chemistry models
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"✅ Random Forest Benchmark Accuracy: {acc:.4f}")

# Create outputs directory if missing
Path("outputs").mkdir(exist_ok=True)

# Save the trained model for dashboard use
model_path = "outputs/random_forest_model.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"📁 Saved Random Forest model to '{model_path}'.")
