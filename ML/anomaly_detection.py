import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix


# ==========================================
# 1. Load dataset
# ==========================================

data = pd.read_csv("data/election_data.csv")


print("Dataset loaded successfully!")
print("Shape:", data.shape)


# ==========================================
# 2. Select ML features
# ==========================================

features = [
    "turnout_percentage",
    "candidate_A_share",
    "candidate_B_share",
    "candidate_C_share",
    "candidate_D_share",
    "rejected_vote_rate"
]


X = data[features]


# ==========================================
# 3. Create Isolation Forest
# ==========================================

model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)


# ==========================================
# 4. Train model
# ==========================================

model.fit(X)


print("Isolation Forest trained successfully!")


# ==========================================
# 5. Predict anomalies
# ==========================================

predictions = model.predict(X)


# Isolation Forest:
#  1  = normal
# -1  = anomaly

data["prediction"] = predictions


# ==========================================
# 6. Convert prediction
# ==========================================

data["ml_anomaly"] = data["prediction"].apply(
    lambda x: 1 if x == -1 else 0
)


# ==========================================
# 7. Anomaly score
# ==========================================

data["anomaly_score"] = model.decision_function(X)


# ==========================================
# 8. Compare with known labels
# ==========================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        data["known_anomaly"],
        data["ml_anomaly"]
    )
)


print("\nClassification Report:")

print(
    classification_report(
        data["known_anomaly"],
        data["ml_anomaly"]
    )
)


# ==========================================
# 9. Display detected anomalies
# ==========================================

anomalies = data[
    data["ml_anomaly"] == 1
]


print("\nDetected Anomalies:")

print(
    anomalies[
        [
            "booth_id",
            "turnout_percentage",
            "candidate_A_share",
            "rejected_vote_rate",
            "anomaly_score"
        ]
    ].head(20)
)


# ==========================================
# 10. Save results
# ==========================================

data.to_csv(
    "data/anomaly_results.csv",
    index=False
)


print("\nAnomaly results saved successfully!")

print(
    "Total anomalies detected:",
    data["ml_anomaly"].sum()
)