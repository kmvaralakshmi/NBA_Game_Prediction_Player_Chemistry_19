import numpy as np
import shap
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
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

# Use Logistic Regression as a model stand-in for feature importance calculation
model = LogisticRegression(max_iter=1000).fit(X, y)

# Initialize SHAP Explainer
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# Create outputs directory if missing
Path("outputs").mkdir(exist_ok=True)

# Generate and save SHAP summary plot
shap.summary_plot(shap_values, X, show=False)
plt.savefig("outputs/shap_summary.png", bbox_inches="tight")
plt.close()

print("✅ SHAP explainability complete. Plot saved to outputs/shap_summary.png.")
