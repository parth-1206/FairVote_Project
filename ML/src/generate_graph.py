import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# -----------------------------------------
# 1. Load results
# -----------------------------------------

results = pd.read_csv("data/anomaly_results.csv")

print("Results loaded successfully!")
print("Shape:", results.shape)


# -----------------------------------------
# 2. Create output directory
# -----------------------------------------

import os

os.makedirs("data/graphs", exist_ok=True)


# -----------------------------------------
# 3. Normal vs Anomalous Distribution
# -----------------------------------------

plt.figure(figsize=(7, 5))

results["ml_anomaly"].value_counts().sort_index().plot(
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
plt.savefig(
    "data/graphs/normal_vs_anomalous.png",
    dpi=300
)
plt.close()


# -----------------------------------------
# 4. Anomaly Score Distribution
# -----------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=results,
    x="anomaly_score",
    hue="ml_anomaly",
    bins=30,
    kde=True
)

plt.title("Anomaly Score Distribution")
plt.xlabel("Anomaly Score")
plt.ylabel("Number of Booths")

plt.tight_layout()
plt.savefig(
    "data/graphs/anomaly_score_distribution.png",
    dpi=300
)
plt.close()


# -----------------------------------------
# 5. Anomaly Rate by Region
# -----------------------------------------

region_analysis = (
    results.groupby("region")["ml_anomaly"]
    .mean()
    .mul(100)
    .reset_index()
)

plt.figure(figsize=(8, 5))

sns.barplot(
    data=region_analysis,
    x="region",
    y="ml_anomaly"
)

plt.title("Anomaly Rate by Region")
plt.xlabel("Region")
plt.ylabel("Anomaly Rate (%)")

plt.tight_layout()
plt.savefig(
    "data/graphs/anomaly_rate_by_region.png",
    dpi=300
)
plt.close()


# -----------------------------------------
# 6. Fairlearn Selection Rate
# -----------------------------------------

fairlearn_results = pd.read_csv(
    "data/fairlearn_results.csv"
)

plt.figure(figsize=(8, 5))

sns.barplot(
    data=fairlearn_results,
    x="region",
    y="selection_rate"
)

plt.title("Fairlearn Selection Rate by Region")
plt.xlabel("Region")
plt.ylabel("Selection Rate")

plt.tight_layout()
plt.savefig(
    "data/graphs/fairlearn_selection_rate.png",
    dpi=300
)
plt.close()


# -----------------------------------------
# 7. Fairlearn TPR and FPR
# -----------------------------------------

fairness_plot = fairlearn_results[
    ["region", "true_positive_rate", "false_positive_rate"]
].copy()

fairness_plot = fairness_plot.melt(
    id_vars="region",
    var_name="metric",
    value_name="value"
)

plt.figure(figsize=(9, 5))

sns.barplot(
    data=fairness_plot,
    x="region",
    y="value",
    hue="metric"
)

plt.title("Fairlearn TPR and FPR by Region")
plt.xlabel("Region")
plt.ylabel("Rate")

plt.tight_layout()
plt.savefig(
    "data/graphs/fairlearn_tpr_fpr.png",
    dpi=300
)
plt.close()


# -----------------------------------------
# 8. Summary
# -----------------------------------------

print()
print("====================================")
print("GRAPH GENERATION COMPLETED!")
print("====================================")

print()
print("Graphs saved in:")
print("data/graphs/")

print()
print("Generated files:")

for file in os.listdir("data/graphs"):
    print("-", file)