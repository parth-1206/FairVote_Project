import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------
# 1. Load anomaly results
# -----------------------------------

df = pd.read_csv("data/anomaly_results.csv")

print("Results loaded successfully!")
print("Shape:", df.shape)

# -----------------------------------
# 2. Anomaly distribution
# -----------------------------------

anomaly_counts = df["ml_anomaly"].value_counts()

print("\nAnomaly Distribution:")
print(anomaly_counts)

plt.figure(figsize=(8, 5))

anomaly_counts.plot(
    kind="bar"
)

plt.title("Normal vs Anomalous Booths")
plt.xlabel("Prediction")
plt.ylabel("Number of Booths")
plt.xticks(
    [0, 1],
    ["Normal", "Anomalous"],
    rotation=0
)

plt.tight_layout()
plt.show()


# -----------------------------------
# 3. Turnout distribution
# -----------------------------------

plt.figure(figsize=(10, 6))

normal = df[df["ml_anomaly"] == 0]
anomalous = df[df["ml_anomaly"] == 1]

plt.hist(
    normal["turnout_percentage"],
    bins=30,
    alpha=0.7,
    label="Normal Booths"
)

plt.hist(
    anomalous["turnout_percentage"],
    bins=15,
    alpha=0.8,
    label="Anomalous Booths"
)

plt.title("Normal vs Anomalous Booth Turnout")
plt.xlabel("Turnout Percentage")
plt.ylabel("Number of Booths")
plt.legend()

plt.tight_layout()
plt.show()


# -----------------------------------
# 4. Rejected vote rate
# -----------------------------------

plt.figure(figsize=(10, 6))

plt.hist(
    normal["rejected_vote_rate"],
    bins=30,
    alpha=0.7,
    label="Normal Booths"
)

plt.hist(
    anomalous["rejected_vote_rate"],
    bins=15,
    alpha=0.8,
    label="Anomalous Booths"
)

plt.title("Rejected Vote Rate: Normal vs Anomalous")
plt.xlabel("Rejected Vote Rate (%)")
plt.ylabel("Number of Booths")
plt.legend()

plt.tight_layout()
plt.show()


# -----------------------------------
# 5. Turnout vs rejected vote rate
# -----------------------------------

plt.figure(figsize=(10, 6))

plt.scatter(
    normal["turnout_percentage"],
    normal["rejected_vote_rate"],
    alpha=0.5,
    label="Normal"
)

plt.scatter(
    anomalous["turnout_percentage"],
    anomalous["rejected_vote_rate"],
    alpha=0.8,
    label="Anomalous"
)

plt.title("Turnout vs Rejected Vote Rate")
plt.xlabel("Turnout Percentage")
plt.ylabel("Rejected Vote Rate (%)")
plt.legend()

plt.tight_layout()
plt.show()


print("\nVisualization completed successfully!")