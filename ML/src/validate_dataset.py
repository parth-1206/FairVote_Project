import pandas as pd
import numpy as np

# 1. Load the dataset
df = pd.read_csv("data/election_data.csv")

print("Dataset loaded successfully!")
print("-" * 40)

# 2. Check dataset shape
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 3. Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# 4. Check duplicate booth IDs
print("\nDuplicate booth IDs:", df["booth_id"].duplicated().sum())

# 5. Check turnout range
print("\nTurnout statistics:")
print(df["turnout_percentage"].describe())

# 6. Check candidate shares
share_columns = [
    "candidate_A_share",
    "candidate_B_share",
    "candidate_C_share",
    "candidate_D_share"
]

df["total_candidate_share"] = df[share_columns].sum(axis=1)

print("\nCandidate share statistics:")
print(df["total_candidate_share"].describe())

# 7. Check whether shares sum to 1
invalid_shares = np.isclose(
    df["total_candidate_share"],
    1.0,
    atol=1e-6
) == False

print(
    "Rows where candidate shares do not sum to 1:",
    invalid_shares.sum()
)

# 8. Check votes cast against total voters
invalid_votes = df["votes_cast"] > df["total_voters"]

print(
    "Rows where votes cast exceed total voters:",
    invalid_votes.sum()
)

# 9. Check anomaly distribution
print("\nAnomaly distribution:")
print(df["known_anomaly"].value_counts())

print("\nValidation completed!")