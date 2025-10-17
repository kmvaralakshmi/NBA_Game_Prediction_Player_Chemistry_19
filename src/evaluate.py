import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

# --- Load predicted and actual labels (Assuming these were saved to outputs/) ---
y_true_path = "outputs/y_test.npy"
y_pred_path = "outputs/y_pred.npy"

try:
    y_true = np.load(y_true_path)
    y_pred = np.load(y_pred_path)
except FileNotFoundError:
    print(f"❌ Error: Prediction files ('{y_true_path}' and/or '{y_pred_path}') not found. Please run the model training scripts first.")
    exit()

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Display the matrix
# Ensure the backend is interactive for display or file save
try:
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Loss', 'Win'])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - NBA Game Prediction (Baseline)")
    plt.show()
except Exception as e:
    # Fallback to saving if display fails (e.g., in a non-interactive environment)
    print(f"Could not display plot interactively ({e}). Saving to file.")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Loss', 'Win'])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - NBA Game Prediction (Baseline)")
    plt.savefig("outputs/confusion_matrix.png")

print("✅ Confusion Matrix processed successfully. Check the plot display or 'outputs/confusion_matrix.png'.")
