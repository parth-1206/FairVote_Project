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
# 2. New booth data
# ---------------------------------------------------

new_booth = pd.DataFrame([{
    "total_voters": 2000,
    "votes_cast": 1950,
    "turnout_percentage": 97.5,
    "candidate_A_share": 0.90,
    "candidate_B_share": 0.04,
    "candidate_C_share": 0.03,
    "candidate_D_share": 0.03,
    "rejected_vote_rate": 5.0
}])


# ---------------------------------------------------
# 3. Features
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

X_new = new_booth[features]


# ---------------------------------------------------
# 4. Scale
# ---------------------------------------------------

X_scaled = scaler.transform(X_new)


# ---------------------------------------------------
# 5. Prediction
# ---------------------------------------------------

prediction = model.predict(X_scaled)[0]

if prediction == -1:
    result = "ANOMALOUS"
else:
    result = "NORMAL"


# ---------------------------------------------------
# 6. Anomaly score
# ---------------------------------------------------

anomaly_score = model.decision_function(X_scaled)[0]


# ---------------------------------------------------
# 7. Explanation layer
# ---------------------------------------------------

reasons = []

turnout = new_booth["turnout_percentage"].iloc[0]
rejected_rate = new_booth["rejected_vote_rate"].iloc[0]

candidate_shares = [
    new_booth["candidate_A_share"].iloc[0],
    new_booth["candidate_B_share"].iloc[0],
    new_booth["candidate_C_share"].iloc[0],
    new_booth["candidate_D_share"].iloc[0]
]

highest_candidate_share = max(candidate_shares)


# High turnout
if turnout >= 90:
    reasons.append(
        f"Very high voter turnout ({turnout:.2f}%)"
    )


# High rejected vote rate
if rejected_rate >= 3:
    reasons.append(
        f"High rejected vote rate ({rejected_rate:.2f}%)"
    )


# Dominant candidate
if highest_candidate_share >= 0.80:
    reasons.append(
        f"Highly concentrated candidate vote share "
        f"({highest_candidate_share * 100:.2f}%)"
    )


# ---------------------------------------------------
# 8. Display result
# ---------------------------------------------------

print("\n--------------------------------")
print("FairVote Anomaly Detection")
print("--------------------------------")

print(f"Total Voters: {new_booth['total_voters'].iloc[0]}")
print(f"Votes Cast: {new_booth['votes_cast'].iloc[0]}")
print(f"Turnout: {turnout:.2f}%")
print(f"Rejected Vote Rate: {rejected_rate:.2f}%")

print("\nPrediction:", result)
print("Anomaly Score:", round(anomaly_score, 4))


if result == "ANOMALOUS":

    print("\nPossible Reasons:")

    if reasons:
        for reason in reasons:
            print(" -", reason)
    else:
        print(" - Unusual combination of multiple features detected by ML model.")

else:

    print("\nNo major anomaly indicators detected.")