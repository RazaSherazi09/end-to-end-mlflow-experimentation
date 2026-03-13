# End-to-End MLflow Experiment Tracking (Iris Dataset)

## Overview

This project demonstrates **end-to-end machine learning experiment tracking using MLflow**.  
The experiment trains multiple models on the **Iris dataset** and tracks parameters, metrics, artifacts, and models using MLflow.

The workflow also includes **model comparison, artifact logging, payload validation, and model registration** in the MLflow Model Registry.

---

## Objectives

The main goals of this project are:

- Implement MLflow experiment tracking
- Log model parameters and evaluation metrics
- Generate and log artifacts such as plots
- Validate input payload before inference
- Register trained models in MLflow Model Registry
- Compare multiple experiment runs

---

## Dataset

The project uses the **Iris dataset** from `sklearn.datasets`.

Dataset characteristics:

- 150 samples
- 4 numerical features
- 3 target classes

Features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

---

## Models Used

Two machine learning models were trained and evaluated:

### Logistic Regression

Used as a baseline linear classification model.

Parameters logged:
- C
- max_iter
- solver

### Random Forest Classifier

Used as an ensemble learning model.

Parameters logged:
- max_depth
- n_estimators
- criterion

---

## MLflow Experiment Tracking

MLflow was used to track the complete experiment lifecycle.

The following MLflow functionalities were used:

### Experiment Creation

mlflow.set_experiment("Iris_Experiment")


### Run Tracking

Each execution creates a new run in MLflow.

### Parameter Logging

mlflow.log_param()


### Metric Logging

mlflow.log_metric()


Metrics tracked:

- Accuracy
- F1 Score
- Precision

### Artifact Logging

The following artifacts are logged:

- Confusion Matrix Plot
- Model Performance Comparison Plot

### Model Logging

The trained Random Forest model is logged using:

mlflow.sklearn.log_model()


### Model Registry

The trained model is registered in MLflow Model Registry as:


MLflow_Iris_Classifier


---

## Payload Validation for Inference

Before performing prediction, input data is validated to ensure:

- Correct feature dimensions
- No missing values
- Proper numeric format

After validation, the trained model performs prediction successfully.

---

## Project Structure


end-to-end-mlflow-experimentation
│
├── artifacts
│
├── plots
│ ├── confusion_matrix.png
│ └── performance_comparison.png
│
├── src
│ └── train.py
│
├── data
│
├── README.md
├── requirements.txt
└── .gitignore


---

## Running the Project

Activate virtual environment:


venv\Scripts\activate


Run the training pipeline:

python src/train.py


Start MLflow UI:

mlflow ui


Open in browser:
127.0.0.1:5000


---

## Experiment Results

After running the pipeline:

- MLflow logs parameters and metrics
- Artifacts such as plots are stored
- Model versions appear in Model Registry
- Multiple runs can be compared using MLflow UI

### Best Model

The **Random Forest Classifier** performed better than Logistic Regression based on:

- Accuracy
- F1 Score

---

## Technologies Used

- Python
- Scikit-learn
- MLflow
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## Conclusion

This project demonstrates how MLflow can be used to manage the machine learning lifecycle.  
Using MLflow simplifies experiment tracking, model comparison, and reproducibility in ML projects.

---
