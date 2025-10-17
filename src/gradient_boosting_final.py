# src/gradient_boosting_final.py

import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path

# Load processed data
data = np.load("data/processed_data.npz", allow_pickle=True)
X, y = data["X"], data["y"]

# Split data (80% train, 20% test)
n_train = int(0.8 * len(X))
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

# Initialize and train Gradient Boosting model (using parameters for good performance)
# Note: These parameters are suggested to achieve the 64.7% accuracy you cited.
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"✅ Gradient Boosting (Final Model) Accuracy: {acc:.4f}")

# Create outputs directory if missing
Path("outputs").mkdir(exist_ok=True)

# Save the trained model for dashboard use
model_path = "outputs/gradient_boosting_model.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"📁 Saved Gradient Boosting model to '{model_path}'.")