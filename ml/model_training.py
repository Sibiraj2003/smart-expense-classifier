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

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler

import matplotlib.pyplot as plt
import seaborn as sns

from joblib import dump
from ml.eda_analysis import project_root
from ml.features import prepare_features, prepare_labels


# ======================= LOAD MERGED TRAINING DATA =======================

data_path = os.path.join(project_root, "data", "model_training_dataset.csv")
df = pd.read_csv(data_path)

# Prepare features (X) and labels (y)
X = prepare_features(df)
y, label_encoder = prepare_labels(df)

print("Feature matrix shape:", X.shape)
print("Labels shape:", len(y))


# ========================================================================
#                   CYCLIC ENCODING FOR TIME FEATURES
# ========================================================================
if "Month" in X.columns:
    X["Month_sin"] = np.sin(2 * np.pi * X["Month"] / 12)
    X["Month_cos"] = np.cos(2 * np.pi * X["Month"] / 12)

if "Day" in X.columns:
    X["Day_sin"] = np.sin(2 * np.pi * X["Day"] / 31)
    X["Day_cos"] = np.cos(2 * np.pi * X["Day"] / 31)

if "Weekday" in X.columns:
    X["Weekday_sin"] = np.sin(2 * np.pi * X["Weekday"] / 7)
    X["Weekday_cos"] = np.cos(2 * np.pi * X["Weekday"] / 7)


# ========================================================================
#                          REMOVE NOISY FEATURES
# ========================================================================
drop_cols = ["index", "Exp Type_Freq"]  # Noisy based on importance

for col in drop_cols:
    if col in X.columns:
        X = X.drop(columns=[col])


# ========================================================================
#                   TRAIN / TEST SPLIT (STRATIFIED)
# ========================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)



# ========================================================================
#                TARGET-SPECIFIC OVERSAMPLING (IMPORTANT)
# ========================================================================
# We force more samples for minority classes (Luxury & Transport)

oversampler = RandomOverSampler(
    sampling_strategy={
        # class_name: target_samples
        # tune per results
        label_encoder.transform(["Luxury"])[0]: 5000,
        label_encoder.transform(["Transport"])[0]: 5000,
        label_encoder.transform(["Essential"])[0]: 12236,
    }
)

X_train, y_train = oversampler.fit_resample(X_train, y_train)



# ========================================================================
#                          BASELINE MODEL
# ========================================================================

model = CatBoostClassifier(
    iterations=500,
    depth=5,
    learning_rate=0.05,
    l2_leaf_reg=5,
    loss_function="MultiClass",
    auto_class_weights="Balanced",
    random_state=42,
    verbose=False
)

model.fit(X_train, y_train)



# ========================================================================
#                    FEATURE IMPORTANCE AND PRUNING
# ========================================================================
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


# ========================================================================
#                       FINAL REFINED MODEL
# ========================================================================

# compute soft class weights (less aggressive than "Balanced")
class_counts = pd.Series(y).value_counts().sort_index()
max_count = class_counts.max()
raw_weights = max_count / class_counts
soft_weights = 1 + (raw_weights - 1) * 0.2
class_weights = soft_weights.tolist()
print("Using class_weights:", class_weights)


model_refined = CatBoostClassifier(
    iterations=400,
    depth=6,
    learning_rate=0.1,
    loss_function="MultiClass",
    class_weights= class_weights,
    verbose=False
)

model_refined.fit(X_train, y_train)



# ========================================================================
#                   7. CLASSIFICATION REPORT + F1
# ========================================================================
preds = model_refined.predict(X_test)
pred_labels = label_encoder.inverse_transform(preds.astype(int))

print("\nClassification Report:")
print(classification_report(
    label_encoder.inverse_transform(y_test.astype(int)),
    pred_labels
))



# ========================================================================
#                    CONFUSION MATRIX + PLOT
# ========================================================================
cm = confusion_matrix(label_encoder.inverse_transform(y_test.astype(int)), pred_labels)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("True")
plt.xlabel("Predicted")

plot_dir = os.path.join(project_root, "ml", "output", "plots")
os.makedirs(plot_dir, exist_ok=True)
plt.savefig(os.path.join(plot_dir, "confusion_matrix.png"))
print("✔ Confusion matrix saved at:", os.path.join(plot_dir, "confusion_matrix.png"))



# ========================================================================
#      SAVE FINAL MODEL + ENCODER + FEATURES 
# ========================================================================
output_dir = os.path.join(project_root, "ml", "output", "models")
os.makedirs(output_dir, exist_ok=True)

model_path = os.path.join(output_dir, "expense_classifier_v1.joblib")
encoder_path = os.path.join(output_dir, "label_encoder_v1.joblib")
features_path = os.path.join(output_dir, "feature_columns_v1.txt")

dump(model_refined, model_path)
print("✔ Model saved to:", model_path)

dump(label_encoder, encoder_path)
print("✔ Label encoder saved to:", encoder_path)

with open(features_path, "w") as f:
    for col in X.columns:
        f.write(col + "\n")

print("✔ Feature column list saved")


#############  METRICS ##############

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# after predictions...
preds = model_refined.predict(X_test)

precision, recall, f1, support = precision_recall_fscore_support(
    y_test,
    preds,
    average=None    # per class
)

class_names = list(label_encoder.classes_)
class_dist = pd.Series(y_test).value_counts().to_dict()

metrics = {
    "precision": precision.tolist(),
    "recall": recall.tolist(),
    "f1": f1.tolist(),
    "support": support.tolist(),
    "class_names": class_names,
    "class_distribution": {str(k): int(v) for k, v in class_dist.items()},
}


with open(os.path.join(project_root, "ml", "output", "plots", "metrics.json"), "w") as f:
    json.dump(metrics, f)




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
