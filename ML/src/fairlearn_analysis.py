import pandas as pd

from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    false_positive_rate,
    true_positive_rate
)


# ---------------------------------------
# 1. Load results
# ---------------------------------------

results = pd.read_csv(
    "data/anomaly_results.csv"
)

print("Results loaded successfully!")
print("Shape:", results.shape)


# ---------------------------------------
# 2. Define variables
# ---------------------------------------

y_true = results["known_anomaly"]

y_pred = results["predicted_anomaly"]

sensitive_feature = results["region"]


# ---------------------------------------
# 3. Selection Rate
# ---------------------------------------

metric_frame = MetricFrame(
    metrics={
        "selection_rate": selection_rate,
        "false_positive_rate": false_positive_rate,
        "true_positive_rate": true_positive_rate
    },
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features=sensitive_feature
)


# ---------------------------------------
# 4. Display metrics by region
# ---------------------------------------

print("\nFairlearn Metrics by Region:")
print(metric_frame.by_group)


# ---------------------------------------
# 5. Overall metrics
# ---------------------------------------

print("\nOverall Metrics:")
print(metric_frame.overall)


# ---------------------------------------
# 6. Fairness disparities
# ---------------------------------------

print("\nFairness Disparities:")

differences = metric_frame.difference(
    method="between_groups"
)

ratios = metric_frame.ratio(
    method="between_groups"
)

for metric in [
    "selection_rate",
    "false_positive_rate",
    "true_positive_rate"
]:

    difference = differences[metric]
    ratio = ratios[metric]

    print(f"\n{metric}")

    print(
        f"Difference: {difference:.4f}"
    )

    print(
        f"Ratio: {ratio:.4f}"
    )


# ---------------------------------------
# 7. Save Fairlearn results
# ---------------------------------------

fairness_results = metric_frame.by_group

fairness_results.to_csv(
    "data/fairlearn_results.csv"
)

print(
    "\nFairlearn analysis completed successfully!"
)

print(
    "Results saved to: data/fairlearn_results.csv"
)