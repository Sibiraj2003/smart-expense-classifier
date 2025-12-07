import numpy as np
import pandas as pd

numeric_cols = [
    "Amount", "Day", "Month", "Year", "Quarter",
    "Days_Since_First", "City_Freq", "Card Type_Freq",
    "Exp Type_Freq", "Is_Weekend", "index", "Amount_log",
    "Weekday"
]

def preprocess_input(data: dict, feature_cols: list):
    """
    Phase-2 reusable preprocessing:
    - convert numbers
    - add cyclic encodings
    - fill missing columns
    - enforce column order
    """

    df = pd.DataFrame([data])

    # Convert numeric string -> float
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Cyclic features
    if "Month" in df.columns:
        df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
        df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    if "Day" in df.columns:
        df["Day_sin"] = np.sin(2 * np.pi * df["Day"] / 31)
        df["Day_cos"] = np.cos(2 * np.pi * df["Day"] / 31)

    if "Weekday" in df.columns:
        df["Weekday_sin"] = np.sin(2 * np.pi * df["Weekday"] / 7)
        df["Weekday_cos"] = np.cos(2 * np.pi * df["Weekday"] / 7)

    # Add missing columns as 0
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    # correct column order
    df = df[feature_cols]

    return df
