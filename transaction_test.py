import pandas as pd
import numpy as np
from pathlib import Path

# Load the held-out test dataset
X_test = pd.read_parquet(
    "data/processed/test_features.parquet"
)

y_test = pd.read_parquet(
    "data/processed/test_target.parquet"
)

# Confirm feature and target alignment
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

assert len(X_test) == len(y_test), "Feature/target row mismatch."

# Select 50 fraudulent and 50 legitimate transactions
rng = np.random.default_rng(42)

fraud_positions = np.flatnonzero(
    y_test["isFraud"].to_numpy() == 1
)

legitimate_positions = np.flatnonzero(
    y_test["isFraud"].to_numpy() == 0
)

selected_positions = np.concatenate([
    rng.choice(fraud_positions, 50, replace=False),
    rng.choice(legitimate_positions, 50, replace=False)
])

# Shuffle the selected transactions
rng.shuffle(selected_positions)

# Export all 432 model features
sample_features = X_test.iloc[selected_positions].copy()

sample_features.to_csv(
    "data/processed/employer_batch_test.csv",
    index=False
)

# Export actual labels separately for evaluation
sample_labels = y_test.iloc[selected_positions].copy()

sample_labels.insert(
    0,
    "row_reference",
    range(1, len(sample_labels) + 1)
)

sample_labels.to_csv(
    "data/processed/employer_batch_test_labels.csv",
    index=False
)

print("Employer test dataset created successfully.")
print("Feature shape:", sample_features.shape)
print("Label distribution:")
print(sample_labels["isFraud"].value_counts())