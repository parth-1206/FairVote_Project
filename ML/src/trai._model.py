import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("data/election_data.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ==========================================
# 2. SELECT FEATURES
# ==========================================

features = [
    "total_voters",
    "votes_cast",
    "turnout_percentage",
    "candidate_A_share",
    "candidate_B_share",
    "candidate_C_share",
    "candidate_D_share",
    "rejected_vote_rate"
]

X = df[features]

# Ground-truth label
y_true = df["known_anomaly"]


# ==========================================
# 3. SCALE FEATURES
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nFeatures prepared successfully!")
print("Input shape:", X_scaled.shape)


# ==========================================
# 4. CREATE ISOLATION FOREST
# ==========================================

model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)


# ==========================================
# 5. TRAIN MODEL
# ==========================================

model.fit(X_scaled)

print("\nIsolation Forest trained successfully!")


# ==========================================
# 6. PREDICT ANOMALIES
# ==========================================

predictions = model.predict(X_scaled)
# ==========================================
# 7. CONVERT PREDICTIONS TO 0/1
# ==========================================

# Isolation Forest:
#  1  = Normal
# -1  = Anomaly

# We convert it to:
#  0  = Normal
#  1  = Anomaly

y_pred = (predictions == -1).astype(int)

print("\nConverted predictions:")
print(y_pred[:20])


# ==========================================
# 8. MODEL EVALUATION
# ==========================================

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=["Normal", "Anomalous"]
))


# ==========================================
# 9. INDIVIDUAL METRICS
# ==========================================

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\nModel Performance:")
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# ==========================================
# 10. PREDICTION DISTRIBUTION
# ==========================================

print("\nPredicted anomaly distribution:")
print(pd.Series(y_pred).value_counts())

import joblib
import os

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save trained Isolation Forest model
joblib.dump(model, "models/isolation_forest.pkl")

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

print("Model saved successfully!")
print("Scaler saved successfully!")