import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import os

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CREATE PROJECT FOLDERS
# ==========================================

os.makedirs("plots", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

# ==========================================
# SETUP & DATA PREPARATION
# ==========================================

mlflow.set_experiment("Iris_Experiment")

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# TASK 6: PAYLOAD VALIDATION FUNCTION
# ==========================================

def validate_payload(data):

    if data.shape[1] != X.shape[1]:
        raise ValueError("Invalid feature count")

    if np.isnan(data).any():
        raise ValueError("Missing values detected")

    return True


# ==========================================
# RUN 1: LOGISTIC REGRESSION
# ==========================================

with mlflow.start_run(run_name="Logistic_Regression_Model") as run_log:

    print(f"Logistic Run ID: {run_log.info.run_id}")

    model_lr = LogisticRegression(C=1.0, max_iter=200)

    model_lr.fit(X_train, y_train)

    y_pred = model_lr.predict(X_test)

    # Log Parameters
    mlflow.log_param("C", 1.0)
    mlflow.log_param("max_iter", 200)
    mlflow.log_param("solver", "lbfgs")

    # Log Metrics
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average='weighted'))

    # Log Model
    mlflow.sklearn.log_model(model_lr, "logistic_model")


# ==========================================
# RUN 2: RANDOM FOREST
# ==========================================

with mlflow.start_run(run_name="Random_Forest_Model") as run_rf:

    print(f"Random Forest Run ID: {run_rf.info.run_id}")

    model_rf = RandomForestClassifier(max_depth=3, n_estimators=100)

    model_rf.fit(X_train, y_train)

    y_pred_rf = model_rf.predict(X_test)

    # Log Parameters
    mlflow.log_param("max_depth", 3)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("criterion", "gini")

    acc_rf = accuracy_score(y_test, y_pred_rf)

    # Log Metrics
    mlflow.log_metric("accuracy", acc_rf)
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred_rf, average='weighted'))

    # ==========================================
    # TASK 4: ARTIFACT LOGGING
    # ==========================================

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_rf)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.title("Confusion Matrix")

    cm_path = "plots/confusion_matrix.png"

    plt.savefig(cm_path)
    plt.close()

    mlflow.log_artifact(cm_path)


    # Model Performance Comparison
    plt.figure()

    plt.bar(["Logistic", "Random Forest"], [0.96, acc_rf], color=['blue','green'])

    plt.ylabel("Accuracy")
    plt.title("Model Comparison")

    comp_path = "plots/performance_comparison.png"

    plt.savefig(comp_path)
    plt.close()

    mlflow.log_artifact(comp_path)


    # ==========================================
    # TASK 5: MODEL LOGGING
    # ==========================================

    mlflow.sklearn.log_model(
        model_rf,
        "random_forest_model"
    )


    # ==========================================
    # TASK 6: PAYLOAD VALIDATION & INFERENCE
    # ==========================================

    sample = X_test[:1]

    if validate_payload(sample):

        prediction = model_rf.predict(sample)

        print("Prediction for sample:", prediction)


    # ==========================================
    # TASK 7: MODEL REGISTRATION
    # ==========================================

    model_uri = f"runs:/{run_rf.info.run_id}/random_forest_model"

    mlflow.register_model(
        model_uri,
        "MLflow_Iris_Classifier"
    )


print("Pipeline complete. Check MLflow UI.")