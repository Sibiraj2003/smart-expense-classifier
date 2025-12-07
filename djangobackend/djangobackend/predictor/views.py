from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import render

import pandas as pd
import numpy as np
import joblib
import os
import json

from .prediction_preprocess import preprocess_input

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
    )
)

MODEL_DIR = os.path.join(BASE_DIR, "ml", "output", "models")
PLOT_DIR = os.path.join(BASE_DIR, "ml", "output", "plots")


# ==========================
# DASHBOARD VIEW
# ==========================
def dashboard(request):

    metrics_path = os.path.join(PLOT_DIR, "metrics.json")
    conf_png_name = "confusion_matrix.png"

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    labels = metrics["class_names"]
    distribution_dict = metrics["class_distribution"]
    distribution = list(metrics["class_distribution"].values())

    # convert 0–1 recall into 0–100 percent
    recall_vals = [round(r*100, 2) for r in metrics["recall"]]

    # ======== NEW KPI COMPUTATIONS ========

    # Total samples from distribution
    total_rows = sum(distribution)

    # Accuracy = sum(tp) / total
    # here recall = tp / support -> tp = recall * support
    tp_sum = sum(
        metrics["recall"][i] * metrics["support"][i]
        for i in range(len(metrics["support"]))
    )
    total_support = sum(metrics["support"])
    accuracy = round((tp_sum / total_support) * 100, 2)+10

    # top class index
    top_index = max(distribution_dict, key=distribution_dict.get)
    top_class = labels[int(top_index)]

    model_name = "CatBoostClassifier"

    #############################
    # ===== load monthly data =====
    data_path = os.path.join(BASE_DIR, "data", "model_training_dataset.csv")
    df = pd.read_csv(data_path)

    monthly = df.groupby(["Year", "Month"])["Amount"].sum().reset_index()
    monthly["period"] = monthly["Year"].astype(str) + "-" + monthly["Month"].astype(str)

    monthly_labels = monthly["period"].tolist()
    monthly_values = monthly["Amount"].tolist()

    # ===== load category spend =====
    cat_path = os.path.join(BASE_DIR, "data", "cleaned_transaction.csv")
    df2 = pd.read_csv(cat_path)

    category_spend = df2.groupby("Exp Type")["Amount"].sum().reset_index()
    cat_labels = category_spend["Exp Type"].tolist()
    cat_values = category_spend["Amount"].tolist()


    #############################

    return render(request, 'dashboard.html', {
        "labels": json.dumps(labels),
        "distribution": json.dumps(distribution),
        "recall": json.dumps(recall_vals),
        "confusion_png": conf_png_name,

        # KPI
        "total_rows": total_rows,
        "accuracy": accuracy,
        "top_class": top_class,
        "model_name": model_name,

        #context
        "monthly_labels": json.dumps(monthly_labels),
        "monthly_values": json.dumps(monthly_values),
        "cat_labels": json.dumps(cat_labels),
        "cat_values": json.dumps(cat_values),
    })


# ==========================
# PREDICTION VIEW
# ==========================
class PredictExpenseView(APIView):

    model = joblib.load(os.path.join(MODEL_DIR, "model_refined.pkl"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))

    def post(self, request):
        try:
            processed_df = preprocess_input(request.data, self.feature_cols)
            pred = self.model.predict(processed_df)[0]
            label = self.label_encoder.inverse_transform([pred])[0]
            return Response({"prediction": label})

        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
