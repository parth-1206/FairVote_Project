import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
df = pd.read_csv("data/election_data.csv")

print("Dataset loaded successfully!")
print(df.head())

# 2. Create a histogram of turnout
plt.figure(figsize=(10, 6))

plt.hist(
    df["turnout_percentage"],
    bins=30,
    edgecolor="black"
)

plt.title("Distribution of Voter Turnout")
plt.xlabel("Turnout Percentage")
plt.ylabel("Number of Booths")

plt.grid(axis="y", alpha=0.3)

plt.show()

# Step 6.3: Compare normal and anomalous turnout

normal = df[df["known_anomaly"] == 0]
anomalous = df[df["known_anomaly"] == 1]

plt.figure(figsize=(10, 6))

plt.hist(
    normal["turnout_percentage"],
    bins=30,
    alpha=0.7,
    label="Normal Booths"
)

plt.hist(
    anomalous["turnout_percentage"],
    bins=30,
    alpha=0.7,
    label="Anomalous Booths"
)

plt.title("Normal vs Anomalous Booth Turnout")
plt.xlabel("Turnout Percentage")
plt.ylabel("Number of Booths")

plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.show()

# Step 6.5: Correlation analysis

# Select numerical columns
numeric_df = df.select_dtypes(include="number")

# Calculate correlation matrix
correlation_matrix = numeric_df.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

# Plot heatmap
plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Heatmap")

plt.show()

# ==========================================
# FEATURE SELECTION FOR ML
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

print("\nFeatures selected for ML:")
print(X.columns.tolist())

print("\nML Input Shape:")
print(X.shape)

print("\nFirst 5 rows of ML input:")
print(X.head())

# ==========================================
# FEATURE SCALING
# ==========================================



scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nScaled ML Input Shape:")
print(X_scaled.shape)

print("\nFirst 5 rows after scaling:")
print(X_scaled[:5])