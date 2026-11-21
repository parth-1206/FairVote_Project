import numpy as np
import pandas as pd

# -----------------------------
# 1. Configuration
# -----------------------------

np.random.seed(42)

TOTAL_OBSERVATIONS = 1000
NORMAL_OBSERVATIONS = 950
ANOMALY_OBSERVATIONS = 50


# -----------------------------
# 2. Generate normal data
# -----------------------------

total_voters = np.random.randint(
    500,
    2000,
    NORMAL_OBSERVATIONS
)

turnout_percentage = np.random.normal(
    loc=65,
    scale=7,
    size=NORMAL_OBSERVATIONS
)

turnout_percentage = np.clip(
    turnout_percentage,
    40,
    85
)

votes_cast = (
    total_voters * turnout_percentage / 100
).astype(int)


# -----------------------------
# 3. Generate candidate shares
# -----------------------------

candidate_shares = np.random.dirichlet(
    alpha=[4, 3, 2, 1],
    size=NORMAL_OBSERVATIONS
)

candidate_A_share = candidate_shares[:, 0]
candidate_B_share = candidate_shares[:, 1]
candidate_C_share = candidate_shares[:, 2]
candidate_D_share = candidate_shares[:, 3]


# -----------------------------
# 4. Generate rejected vote rate
# -----------------------------

rejected_vote_rate = np.random.uniform(
    0.5,
    3.0,
    NORMAL_OBSERVATIONS
)


# -----------------------------
# 5. Generate regions
# -----------------------------

regions_normal = np.random.choice(
    ["North", "South", "East", "West"],
    size=NORMAL_OBSERVATIONS,
    p=[0.25, 0.25, 0.25, 0.25]
)


# -----------------------------
# 6. Create normal DataFrame
# -----------------------------

normal_data = pd.DataFrame({
    "booth_id": [
        f"B{i:04d}"
        for i in range(1, NORMAL_OBSERVATIONS + 1)
    ],

    "total_voters": total_voters,

    "votes_cast": votes_cast,

    "turnout_percentage": turnout_percentage,

    "candidate_A_share": candidate_A_share,

    "candidate_B_share": candidate_B_share,

    "candidate_C_share": candidate_C_share,

    "candidate_D_share": candidate_D_share,

    "rejected_vote_rate": rejected_vote_rate,

    "region": regions_normal,

    "known_anomaly": 0
})


# -----------------------------
# 7. Generate anomalous data
# -----------------------------

anomaly_total_voters = np.random.randint(
    500,
    2000,
    ANOMALY_OBSERVATIONS
)

anomaly_turnout = np.random.uniform(
    95,
    100,
    ANOMALY_OBSERVATIONS
)

anomaly_votes_cast = (
    anomaly_total_voters * anomaly_turnout / 100
).astype(int)


# -----------------------------
# 8. Anomalous candidate shares
# -----------------------------

anomaly_candidate_shares = np.zeros(
    (ANOMALY_OBSERVATIONS, 4)
)

anomaly_candidate_shares[:, 0] = np.random.uniform(
    0.90,
    0.99,
    ANOMALY_OBSERVATIONS
)

remaining_share = (
    1 - anomaly_candidate_shares[:, 0]
)

anomaly_candidate_shares[:, 1:] = (
    remaining_share[:, None] / 3
)


# -----------------------------
# 9. Anomalous rejected rate
# -----------------------------

anomaly_rejected_rate = np.random.uniform(
    15,
    25,
    ANOMALY_OBSERVATIONS
)


# -----------------------------
# 10. Generate anomaly regions
# -----------------------------

regions_anomaly = np.random.choice(
    ["North", "South", "East", "West"],
    size=ANOMALY_OBSERVATIONS,
    p=[0.20, 0.20, 0.20, 0.40]
)


# -----------------------------
# 11. Create anomaly DataFrame
# -----------------------------

anomaly_data = pd.DataFrame({
    "booth_id": [
        f"B{i:04d}"
        for i in range(
            NORMAL_OBSERVATIONS + 1,
            TOTAL_OBSERVATIONS + 1
        )
    ],

    "total_voters": anomaly_total_voters,

    "votes_cast": anomaly_votes_cast,

    "turnout_percentage": anomaly_turnout,

    "candidate_A_share": anomaly_candidate_shares[:, 0],

    "candidate_B_share": anomaly_candidate_shares[:, 1],

    "candidate_C_share": anomaly_candidate_shares[:, 2],

    "candidate_D_share": anomaly_candidate_shares[:, 3],

    "rejected_vote_rate": anomaly_rejected_rate,

    "region": regions_anomaly,

    "known_anomaly": 1
})


# -----------------------------
# 12. Combine datasets
# -----------------------------

dataset = pd.concat(
    [normal_data, anomaly_data],
    ignore_index=True
)


# -----------------------------
# 13. Shuffle observations
# -----------------------------

dataset = dataset.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# -----------------------------
# 14. Save CSV
# -----------------------------

dataset.to_csv(
    "data/election_data.csv",
    index=False
)


# -----------------------------
# 15. Display results
# -----------------------------

print("Dataset generated successfully!")

print(f"Total observations: {len(dataset)}")

print(
    f"Normal observations: "
    f"{sum(dataset['known_anomaly'] == 0)}"
)

print(
    f"Anomalous observations: "
    f"{sum(dataset['known_anomaly'] == 1)}"
)

print("\nRegion distribution:")
print(dataset["region"].value_counts())

print("\nAnomalies by region:")
print(
    pd.crosstab(
        dataset["region"],
        dataset["known_anomaly"]
    )
)

print("\nFirst 5 rows:")
print(dataset.head())