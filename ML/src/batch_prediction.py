import pandas as pd
import joblib


# ---------------------------------------------------
# 1. Load model and scaler
# ---------------------------------------------------

model = joblib.load("models/isolation_forest.pkl")
scaler = joblib.load("models/scaler.pkl")

print("Model loaded successfully!")
print("Scaler loaded successfully!")


# ---------------------------------------------------
# 2. Load election dataset
# ---------------------------------------------------

df = pd.read_csv("data/election_data.csv")

print("Dataset loaded successfully!")
print("Total booths:", len(df))


# ---------------------------------------------------
# 3. Features used by the ML model
# ---------------------------------------------------

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


# ---------------------------------------------------
# 4. Scale features
# ---------------------------------------------------

X_scaled = scaler.transform(X)


# ---------------------------------------------------
# 5. Make predictions
# ---------------------------------------------------

raw_predictions = model.predict(X_scaled)

# Isolation Forest:
#  1  = Normal
# -1  = Anomaly

df["predicted_anomaly"] = (raw_predictions == -1).astype(int)


# ---------------------------------------------------
# 6. Calculate anomaly scores
# ---------------------------------------------------

df["anomaly_score"] = model.decision_function(X_scaled)


# ---------------------------------------------------
# 7. Generate explanations
# ---------------------------------------------------

def generate_explanation(row):

    reasons = []

    turnout = row["turnout_percentage"]
    rejected_rate = row["rejected_vote_rate"]

    candidate_shares = [
        row["candidate_A_share"],
        row["candidate_B_share"],
        row["candidate_C_share"],
        row["candidate_D_share"]
    ]

    highest_share = max(candidate_shares)

    if turnout >= 90:
        reasons.append(
            f"Very high turnout ({turnout:.2f}%)"
        )

    if rejected_rate >= 3:
        reasons.append(
            f"High rejected vote rate ({rejected_rate:.2f}%)"
        )

    if highest_share >= 0.80:
        reasons.append(
            f"Highly concentrated candidate share "
            f"({highest_share * 100:.2f}%)"
        )

    if not reasons:
        return "No major rule-based indicators"

    return "; ".join(reasons)


df["explanation"] = df.apply(generate_explanation, axis=1)


# ---------------------------------------------------
# 8. Save results
# ---------------------------------------------------

output_path = "data/anomaly_results.csv"

df.to_csv(output_path, index=False)


# ---------------------------------------------------
# 9. Display summary
# ---------------------------------------------------

print("\n--------------------------------")
print("Batch Anomaly Detection")
print("--------------------------------")

print("Total booths:", len(df))

print("\nPredicted anomaly distribution:")
print(df["predicted_anomaly"].value_counts())

print("\nAnomalous booths:",
      (df["predicted_anomaly"] == 1).sum())

print("Normal booths:",
      (df["predicted_anomaly"] == 0).sum())

print("\nSample anomalous booths:")

print(
    df[df["predicted_anomaly"] == 1][
        [
            "booth_id",
            "turnout_percentage",
            "rejected_vote_rate",
            "anomaly_score",
            "explanation"
        ]
    ].head(10)
)

print("\nResults saved successfully!")
print("File:", output_path)