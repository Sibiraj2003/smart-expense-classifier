import pandas as pd
import numpy as np
import os
from ml.eda_analysis import project_root


def load_data():
    """
    Loads feature-engineered dataset and merges it with the target column.

    Returns:
        pd.DataFrame: Merged dataset ready for model training.
    """

    # File paths (OS-agnostic)
    features_path = os.path.join(project_root, "data", "final_features_small.csv")
    cleaned_path = os.path.join(project_root, "data", "cleaned_transaction.csv")

    # Load datasets
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Feature dataset not found at: {features_path}")

    if not os.path.exists(cleaned_path):
        raise FileNotFoundError(f"Cleaned dataset not found at: {cleaned_path}")

    df_clean = pd.read_csv(cleaned_path)
    df_features = pd.read_csv(features_path)

    # Ensure row counts match before merging
    if len(df_clean) != len(df_features):
        raise ValueError(
            f"Row count mismatch: cleaned={len(df_clean)}, features={len(df_features)}"
        )

    # Check target existence
    if "Exp Type" not in df_clean.columns:
        raise ValueError("Target column 'Exp Type' is missing from cleaned dataset.")

    # Merge features + target (safe because row order preserved)
    df_merged = pd.concat([df_features, df_clean["Exp Type"]], axis=1)
    drop_cols = ["Date"]
    df_merged = df_merged.drop(columns=[col for col in drop_cols if col in df_merged.columns])

    mapping = {
    "Food": "Essential",
    "Grocery": "Essential",
    "Bills": "Essential",
    "Fuel": "Transport",
    "Entertainment": "Luxury",
    "Travel": "Luxury"
    }
    df_merged["Exp Type"] = df_merged["Exp Type"].map(mapping)

    # Save merged dataset for transparency/debugging
    output_path = os.path.join(project_root, "data", "model_training_dataset.csv")
    df_merged.to_csv(output_path, index=False)

    print(f"✔ Merged training dataset saved at: {output_path}")
    print(f"✔ Final training dataset shape: {df_merged.shape}")

    return df_merged



def validate_data(df):
    """
    Validates dataset before model training.

    Args:
        df (pd.DataFrame): Dataset to validate.

    Raises:
        ValueError: If validation fails.
    """

    # 1. Check empty dataset
    if df.empty:
        raise ValueError("Dataset is empty after merging. Cannot proceed.")

    # 2. Check target existence
    if "Exp Type" not in df.columns:
        raise ValueError("Missing target column 'Exp Type'.")

    # 3. Null values in target
    if df["Exp Type"].isnull().any():
        raise ValueError("Target column contains null values. Clean data first.")

    # 4. Infinite values in numeric columns
    num_df = df.select_dtypes(include=[np.number])
    if np.isinf(num_df).any().any():
        raise ValueError("Dataset contains infinite values. Fix preprocessing.")

    # 5. All features numeric?
    feature_cols = df.drop(columns=["Exp Type"]).columns
    non_numeric = df[feature_cols].select_dtypes(exclude=[np.number]).columns

    if len(non_numeric) > 0:
        raise ValueError(f"Found non-numeric columns: {list(non_numeric)}")

    # 6. Check duplicates (not fatal, but warn)
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        print(f"Warning: {dup_count} duplicate rows found. Consider removing.")

    print("Data validation passed successfully!")



if __name__ == "__main__":
    df = load_data()
    validate_data(df)
