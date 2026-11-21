import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load results
results = pd.read_csv("data/anomaly_results.csv")

# Actual and predicted labels
y_true = results["known_anomaly"]
y_pred = results["ml_anomaly"]

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix:")
print(cm)

# Display
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Anomalous"]
)

disp.plot()

plt.title("FairVote Anomaly Detection - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "data/graphs/confusion_matrix.png",
    dpi=300
)

plt.show()

print("Confusion matrix saved successfully!")