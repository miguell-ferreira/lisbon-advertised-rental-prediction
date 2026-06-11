from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"
DATASET_PATH = BASE_DIR / "data" / "processed" / "final_dataset.csv"
X_TEST_PATH = MODELS_DIR / "X_test_aug.pkl"

Y_OUTPUT_PATH = MODELS_DIR / "y_test_aug.pkl"
METADATA_OUTPUT_PATH = MODELS_DIR / "metadata_test_aug.pkl"


# ============================================================
# 2. LOAD FILES
# ============================================================

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Final dataset not found: {DATASET_PATH}")

if not X_TEST_PATH.exists():
    raise FileNotFoundError(f"X_test_aug not found: {X_TEST_PATH}")

df = pd.read_csv(DATASET_PATH)
X_test_aug = pd.read_pickle(X_TEST_PATH)

print("Final dataset shape:", df.shape)
print("X_test_aug shape:", X_test_aug.shape)
print("Final dataset index sample:", df.index[:10])
print("X_test_aug index sample:", X_test_aug.index[:10])


# ============================================================
# 3. RECOVER TARGET
# ============================================================

TARGET_COL = "price_m2"

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in final_dataset.csv")

missing_indices = [idx for idx in X_test_aug.index if idx not in df.index]

if missing_indices:
    raise ValueError(
        "Some X_test_aug indices were not found in final_dataset.csv.\n"
        f"First missing indices: {missing_indices[:10]}"
    )

# The model was trained on log1p(price_m2)
y_test_aug = np.log1p(df.loc[X_test_aug.index, TARGET_COL])

y_test_aug = pd.Series(
    y_test_aug,
    index=X_test_aug.index,
    name="y_test_aug"
)

y_test_aug.to_pickle(Y_OUTPUT_PATH)

print(f"\nSaved y_test_aug to: {Y_OUTPUT_PATH}")
print(y_test_aug.head())


# ============================================================
# 4. RECOVER METADATA FOR RESIDUAL PLOTS
# ============================================================

METADATA_COLS = ["parish", "date"]

for col in METADATA_COLS:
    if col not in df.columns:
        raise ValueError(f"Metadata column '{col}' not found in final_dataset.csv")

metadata_test_aug = df.loc[X_test_aug.index, METADATA_COLS].copy()

# Rename columns to make residual script easier to read
metadata_test_aug = metadata_test_aug.rename(
    columns={
        "parish": "Freguesia",
        "date": "Data"
    }
)

metadata_test_aug.to_pickle(METADATA_OUTPUT_PATH)

print(f"\nSaved metadata_test_aug to: {METADATA_OUTPUT_PATH}")
print(metadata_test_aug.head())


# ============================================================
# 5. CHECK
# ============================================================

print("\nRecovered test period:")
print(metadata_test_aug["Data"].min(), "to", metadata_test_aug["Data"].max())

print("\nRecovered parishes:")
print(metadata_test_aug["Freguesia"].unique())