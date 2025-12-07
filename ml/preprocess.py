import numpy as np
import pandas as pd
import os
from ml.eda_analysis import project_root

data_path = os.path.join(project_root, "data", "Credit card transactions.csv")
df = pd.read_csv(data_path)

def preprocess_transactions(df):
    """
    Full preprocessing pipeline:
    - Load dataset
    - Handle missing values
    - Winsorize (clip) outliers
    - Log transform
    - Save cleaned dataset
    """

    print("\n━━━━━━━━━━━━━━━━━━━━━━")
    print("PREPROCESSING STARTED")
    print("━━━━━━━━━━━━━━━━━━━━━━")

    print("Dataset Loaded:", df.shape)

    # Remove rows where Amount is missing
    df.dropna(subset=["Amount"], inplace=True)

    # Winsorization (1% - 99% clipping)
    lower = df["Amount"].quantile(0.01)
    upper = df["Amount"].quantile(0.99)
    df["Amount"] = np.clip(df["Amount"], lower, upper)

    # Log transform to reduce skew
    df["Amount_log"] = np.log1p(df["Amount"])

    # Save cleaned dataset
    cleaned_path = os.path.join(project_root, "data", "cleaned_transaction.csv")
    df.to_csv(cleaned_path, index=False)

    print("Preprocessing Completed ✔")
    print("Cleaned File Saved At:", cleaned_path)

    return df

if __name__ == "__main__":
    preprocess_transactions(df)
