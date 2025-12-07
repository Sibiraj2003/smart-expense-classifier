"""
ml/model_training.py

Responsible for:
- Loading merged dataset
- Preparing X (features) and y (labels)
- Splitting data
- Training baseline classification model
- Running cross-validation
- Saving the baseline model
"""

import numpy as np
import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from joblib import dump
from ml.eda_analysis import project_root
from ml.features import prepare_features, prepare_labels

data_path = os.path.join(project_root, "data", "model_training_dataset.csv")
df = pd.read_csv(data_path)

# Prepare features (X) and labels (y)
X = prepare_features(df)
y, label_encoder = prepare_labels(df)

print("Feature matrix shape:", X.shape)
print("Labels shape:", len(y))

# Cyclic Encoding for Time Features
if "Month" in X.columns:
    X["Month_sin"] = np.sin(2 * np.pi * X["Month"] / 12)
    X["Month_cos"] = np.cos(2 * np.pi * X["Month"] / 12)

if "Day" in X.columns:
    X["Day_sin"] = np.sin(2 * np.pi * X["Day"] / 31)
    X["Day_cos"] = np.cos(2 * np.pi * X["Day"] / 31)

if "Weekday" in X.columns:
    X["Weekday_sin"] = np.sin(2 * np.pi * X["Weekday"] / 7)
    X["Weekday_cos"] = np.cos(2 * np.pi * X["Weekday"] / 7)

drop_cols = ["index", "Exp Type_Freq"]  # Noisy based on importance

for col in drop_cols:
    if col in X.columns:
        X = X.drop(columns=[col])



# Train-Test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y)# Stratifying ensures train/test both have the same category proportions.

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)

smote = SMOTE(random_state=42, sampling_strategy="auto")
X_train, y_train = smote.fit_resample(X_train, y_train)

# print(df["Exp Type"].value_counts())

model = CatBoostClassifier(
    iterations=500,
    depth=5,
    learning_rate=0.05,
    l2_leaf_reg=5,
    loss_function="MultiClass",
    auto_class_weights="Balanced",
    random_seed=42,
    verbose=False
)


model.fit(X_train, y_train)


# train_acc = model.score(X_train, y_train)
# test_acc = model.score(X_test, y_test)

# print("Training Accuracy:", train_acc)
# print("Test Accuracy:", test_acc)

# cv_scores = cross_val_score(model, X, y, cv=5, scoring="f1_macro", n_jobs=-1)
# print("CV F1 Macro Scores:", cv_scores)
# print("Mean CV F1 Score:", cv_scores.mean())

#########################################

importances = model.get_feature_importance()
feature_names = list(X.columns)

low_importance_features = [
    feature_names[i] for i, score in enumerate(importances) if score < 1.0
]

print("Low-importance features to remove:", low_importance_features)

X_reduced = X.drop(columns=low_importance_features)

print("New feature shape after removal:", X_reduced.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X_reduced,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

model_refined = CatBoostClassifier(
    iterations=400,
    depth=6,
    learning_rate=0.1,
    loss_function="MultiClass",
    verbose=False
)

model_refined.fit(X_train, y_train)

# 7. Evaluate again
# train_acc_refined = model_refined.score(X_train, y_train)
# test_acc_refined = model_refined.score(X_test, y_test)

# print("\n Retraining Completed")
# print("Refined Train Accuracy:", train_acc_refined)
# print("Refined Test Accuracy:", test_acc_refined)

# # 8. Cross validation for refined model
# cv_scores_refined = cross_val_score(
#     model_refined,
#     X_reduced,
#     y,
#     cv=5,
#     scoring="f1_macro",
#     n_jobs=-1
# )

# print("Refined CV F1 Macro:", cv_scores_refined)
# print("Refined Mean CV F1:", cv_scores_refined.mean())


#################################
# TESTING FOR DJANGO INTEGRATION
# sample = X_test.iloc[:5]
# preds = model_refined.predict(sample)
# labels = label_encoder.inverse_transform(preds.astype(int))

# print("Sample Predictions:", labels)
#################################

# ================================
# Save final model + encoder + features
# ================================

output_dir = os.path.join(project_root, "ml", "output", "models")
os.makedirs(output_dir, exist_ok=True)

model_path = os.path.join(output_dir, "expense_classifier_v1.joblib")
encoder_path = os.path.join(output_dir, "label_encoder_v1.joblib")
features_path = os.path.join(output_dir, "feature_columns_v1.txt")

# Save model
dump(model_refined, model_path)
print("✔ Model saved to:", model_path)

# Save label encoder
dump(label_encoder, encoder_path)
print("✔ Label encoder saved to:", encoder_path)

# Save feature column order
with open(features_path, "w") as f:
    for col in X.columns:
        f.write(col + "\n")

print("✔ Feature column list saved")


#########################################################################################
model_path = os.path.join(project_root, "ml", "output", "models", "model_refined.pkl")
encoder_path = os.path.join(project_root, "ml", "output", "models", "label_encoder.pkl")
FEATURE_COLS_PATH = os.path.join(project_root, "ml", "output", "models", "feature_columns.pkl")

dump(list(X.columns), FEATURE_COLS_PATH)
dump(model_refined, model_path)
dump(label_encoder, encoder_path)

print("Refined model saved at:", model_path)
print("Label encoder saved at:", encoder_path)


feature_path = os.path.join(project_root, "ml", "output", "models", "refined_features.json")

with open(feature_path, "w") as f:
    json.dump(list(X_train.columns), f)

print("Feature list saved.")
