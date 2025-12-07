"""
ml/features.py

Module responsible for:
- Extracting numeric feature matrix (X)
- Encoding target labels (y)
- Returning consistent, reproducible feature column ordering

Used by: model_training.py
"""

import numpy as np
import pandas as pd
import os
from ml.eda_analysis import project_root
from sklearn.preprocessing import LabelEncoder

data_path = os.path.join(project_root, "data", "model_training_dataset.csv")


def prepare_features(df):
    """
    Extract feature matrix (X) from the merged training dataset.

    - Removes the target column 'Exp Type'
    - Ensures all remaining feature columns are numeric
    - Returns a clean DataFrame ready for model training

    Args:
        df (pd.DataFrame): The merged dataset containing features + target.

    Returns:
        pd.DataFrame: Numeric feature matrix (X)
    
    Raises:
        ValueError: If any non-numeric columns are detected.
    """
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    if X.select_dtypes(exclude=[np.number]).shape[1] > 0:
        print(X)
        raise ValueError("Non-numeric columns found in features.")
    return X

def prepare_labels(df):
    """
    Extract and encode the target labels.

    - Reads 'Exp Type' column as y
    - Encodes labels using sklearn's LabelEncoder
    - Returns encoded label array + fitted encoder for inference

    Args:
        df (pd.DataFrame): Dataset containing 'Exp Type' column.

    Returns:
        tuple:
            y_encoded (np.ndarray): Encoded label values
            encoder (LabelEncoder): Fitted label encoder for saving/usage
    """
    y = df["Exp Type"]
    encoder = LabelEncoder()
    y_encoder = encoder.fit_transform(y)
    return y_encoder, encoder

def get_feature_columns(df):
    """
    Retrieves all feature columns from the dataset.

    - Excludes the target column 'Exp Type'
    - Returns the list of columns used for model input (X)

    Args:
        df (pd.DataFrame): Dataset containing features + target.

    Returns:
        list[str]: List of feature column names.
    """
    feature_cols = [col for col in df.columns if col!='Exp Type']
    return feature_cols

if __name__ == '__main__':
    df = pd.read_csv(data_path)
    print(get_feature_columns(df))