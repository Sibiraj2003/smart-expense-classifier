"""
ml/inference.py

Handles:
- Loading trained CatBoost model
- Loading label encoder
- Preparing input features for inference
- Predicting a single expense type
"""

import os
import pandas as pd
import numpy as np
from joblib import load
from ml.eda_analysis import project_root

# ------------------------------
# Load model + encoder + columns
# ------------------------------

MODEL_PATH = os.path.join(project_root, "ml", "output", "models", "model_refined.pkl")
ENCODER_PATH = os.path.join(project_root, "ml", "output", "models", "label_encoder.pkl")
FEATURE_COLS_PATH = os.path.join(project_root, "ml", "output", "models", "feature_columns.pkl")

model = load(MODEL_PATH)
label_encoder = load(ENCODER_PATH)
feature_columns = load(FEATURE_COLS_PATH)  # The IMPORTANT part


# ------------------------------
# Prepare input for prediction
# ------------------------------
def prepare_inference_features(data: dict):
    """
    Convert a Python dict into a row-matching DataFrame,
    reorder columns to match training, and fill missing values.
    """

    df = pd.DataFrame([data])  # convert input dict → 1-row DataFrame

    # Missing feature safety
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0   # or better: mean value, if you saved it

    # Reorder columns to EXACT training order
    df = df[feature_columns]

    # Convert to numeric
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    return df


# ------------------------------
# Predict a single input
# ------------------------------
def predict_single(data: dict):
    """
    Takes a Python dict of feature values and returns
    the predicted expense category label.
    """

    X = prepare_inference_features(data)

    pred_encoded = model.predict(X)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    return pred_label


# ------------------------------
# Standalone test run
# ------------------------------
if __name__ == "__main__":
    
    # Example sample input (must match training features)
    sample = {
        "Day": 14,
        "Month": 9,
        "Year": 2014,
        "Quarter": 3,
        "Days_Since_First": 300,
        "City_Freq": 2300,
        "Card Type_Freq": 1500,
        "Exp Type_Freq": 4000,
        "Is_Weekend": 0,
        "Amount": 20000,
        "Amount_log": 10.8,
        "Month_sin": 0.5,
        "Month_cos": -0.8,
    }
    
    print("Prediction:", predict_single(sample))
