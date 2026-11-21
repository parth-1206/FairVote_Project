import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------
# 1. Load anomaly results
# ---------------------------------------

results = pd.read_csv("data/anomaly_results.csv")

print("Results loaded successfully!")
print("Shape:", results.shape)


# ---------------------------------------
# 2. Basic information
# ---------------------------------------

print("\nRegion distribution:")
print(results["region"].value_counts())


# ---------------------------------------
# 3. Anomaly distribution by region
# ---------------------------------------

region_analysis = pd.crosstab(
    results["region"],
    results["predicted_anomaly"]
)

print("\nAnomaly distribution by region:")
print(region_analysis)


# ---------------------------------------
# 4. Calculate anomaly rate
# ---------------------------------------

region_analysis["total_booths"] = (
    region_analysis.sum(axis=1)
)

region_analysis["anomaly_rate"] = (
    region_analysis[1]
    / region_analysis["total_booths"]
    * 100
)

print("\nAnomaly rate by region:")
print(
    region_analysis[
        ["total_booths", 0, 1, "anomaly_rate"]
    ]
)


# ---------------------------------------
# 5. Fairness disparity
# ---------------------------------------

max_rate = region_analysis["anomaly_rate"].max()
min_rate = region_analysis["anomaly_rate"].min()

disparity = max_rate - min_rate

print("\nFairness Analysis:")
print(f"Highest anomaly rate: {max_rate:.2f}%")
print(f"Lowest anomaly rate: {min_rate:.2f}%")
print(f"Disparity: {disparity:.2f} percentage points")


# ---------------------------------------
# 6. Identify highest-risk region
# ---------------------------------------

highest_region = region_analysis[
    "anomaly_rate"
].idxmax()

lowest_region = region_analysis[
    "anomaly_rate"
].idxmin()

print(
    f"\nHighest anomaly-rate region: {highest_region}"
)

print(
    f"Lowest anomaly-rate region: {lowest_region}"
)


# ---------------------------------------
# 7. Visualization
# ---------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    region_analysis.index,
    region_analysis["anomaly_rate"]
)

plt.xlabel("Region")
plt.ylabel("Anomaly Rate (%)")
plt.title("Anomaly Detection Rate by Region")

plt.tight_layout()

plt.savefig(
    "data/anomaly_rate_by_region.png"
)

plt.show()


# ---------------------------------------
# 8. Save fairness results
# ---------------------------------------

region_analysis.to_csv(
    "data/fairness_results.csv"
)

print("\nFairness analysis completed successfully!")
print("Results saved to: data/fairness_results.csv")