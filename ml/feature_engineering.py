import pandas as pd
import numpy as np
import os
from ml.eda_analysis import project_root


def frequency_encode(df, col):
    """Frequency encoding for high-cardinality categorical columns."""
    freq = df[col].value_counts()
    df[col + "_Freq"] = df[col].map(freq)
    return df


def binary_encode(df, col):
    """Convert binary categorical variables to 0/1."""
    df[col] = df[col].astype(str).str.strip().str.upper()
    unique_vals = df[col].unique()

    if len(unique_vals) == 2:
        df[col] = (df[col] == unique_vals[0]).astype(int)
    else:
        df[col], _ = pd.factorize(df[col])

    return df


def feature_engineering():
    print("\n━━━━━━━━━━━━━━━━━━━━━━")
    print("FEATURE ENGINEERING INITIATED")
    print("━━━━━━━━━━━━━━━━━━━━━━")

    # Load cleaned dataset
    cleaned_path = os.path.join(project_root, "data", "cleaned_transaction.csv")
    df = pd.read_csv(cleaned_path)
    print("Loaded Cleaned Dataset:", df.shape)

    # ================================
    # 1. Date Handling
    # ================================
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.dropna(subset=["Date"], inplace=True)

    df["Day"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year
    df["Weekday"] = df["Date"].dt.weekday
    df["Is_Weekend"] = (df["Weekday"] >= 5).astype(int)
    df["Quarter"] = df["Date"].dt.quarter

    df = df.sort_values("Date")
    df["Days_Since_First"] = (df["Date"] - df["Date"].min()).dt.days

    # ================================
    # 2. Categorical Feature Handling
    # ================================
    high_card_cols = ["City", "Card Type", "Exp Type"]
    binary_cols = ["Gender"]

    # Frequency Encoding for high-cardinality columns
    for col in high_card_cols:
        df = frequency_encode(df, col)

    # Binary/Factor encoding for Gender
    for col in binary_cols:
        df = binary_encode(df, col)

    # Drop original string columns
    df.drop(columns=high_card_cols + binary_cols, inplace=True)

    print("Categorical Encoding Completed.")
    print("New Shape:", df.shape)

    # ================================
    # 3. Save Final Dataset
    # ================================
    final_path_parquet = os.path.join(project_root, "data", "final_features.parquet")
    final_path_csv = os.path.join(project_root, "data", "final_features_small.csv")

    df.to_parquet(final_path_parquet, index=False)
    df.to_csv(final_path_csv, index=False)

    print("✔ Feature Engineering Completed")
    print("Saved Parquet:", final_path_parquet)
    print("Saved Lightweight CSV:", final_path_csv)

    return df


if __name__ == "__main__":
    feature_engineering()
