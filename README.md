# End-to-End MLflow Experiment Tracking (Iris Dataset)

## Overview

This project demonstrates end-to-end machine learning experiment tracking with MLflow using the Iris dataset. It trains multiple classification models, logs parameters and evaluation metrics, saves artifacts, validates inference payloads, and registers the best-performing model in the MLflow Model Registry.

## Objectives

- Track machine learning experiments with MLflow
- Log model parameters and evaluation metrics
- Generate and store visual artifacts
- Validate inference payloads before prediction
- Register trained models in the MLflow Model Registry
- Compare multiple runs in the MLflow UI

## Dataset

The project uses the Iris dataset from `sklearn.datasets`.

### Dataset Summary

- 150 samples
- 4 numerical input features
- 3 target classes

### Features

- Sepal length
- Sepal width
- Petal length
- Petal width

## Models Used

Two machine learning models are trained and evaluated.

### Logistic Regression

Used as the baseline linear classifier.

Logged parameters:

- `C`
- `max_iter`
- `solver`

### Random Forest Classifier

Used as the ensemble model.

Logged parameters:

- `max_depth`
- `n_estimators`
- `criterion`

## MLflow Tracking Features

This project uses MLflow to manage the full experiment lifecycle.

### Experiment Creation

```python
mlflow.set_experiment("Iris_Experiment")
```

### Run Tracking

Each model is executed inside its own MLflow run.

### Parameter Logging

```python
mlflow.log_param("name", value)
```

### Metric Logging

```python
mlflow.log_metric("metric_name", metric_value)
```

Tracked metrics include:

- Accuracy
- F1 score

### Artifact Logging

The pipeline logs the following artifacts:

- Confusion matrix plot
- Model performance comparison plot

### Model Logging

```python
mlflow.sklearn.log_model(model, "model_name")
```

### Model Registry

The registered model name is:

```text
MLflow_Iris_Classifier
```

## Payload Validation for Inference

Before prediction, input data is validated to ensure:

- The feature count matches the training data
- There are no missing values
- The payload is suitable for numeric inference

After validation, the trained model performs prediction on the sample input.

## Project Structure

```text
MLflow-Exp/
|-- artifacts/
|-- data/
|-- mlruns/
|-- plots/
|   |-- confusion_matrix.png
|   `-- performance_comparison.png
|-- src/
|   `-- train.py
|-- LICENSE
|-- README.md
|-- main.py
`-- requirements.txt
```

## Running the Project

### 1. Activate the virtual environment

```powershell
venv\Scripts\Activate.ps1
```

### 2. Run the training pipeline

```powershell
python src\train.py
```

### 3. Start the MLflow UI

```powershell
mlflow ui
```

### 4. Open MLflow in the browser

```text
http://127.0.0.1:5000
```

## Experiment Results

After running the pipeline:

- MLflow stores parameters and metrics for each run
- Plot artifacts are saved and logged
- The trained Random Forest model is logged and registered
- Multiple runs can be reviewed and compared in the MLflow UI

### Best Model

Based on the current script, the Random Forest classifier is presented as the stronger model using:

- Accuracy
- F1 score

## Technologies Used

- Python
- Scikit-learn
- MLflow
- Pandas
- NumPy
- Matplotlib
- Seaborn

## Conclusion

This project shows how MLflow can be used to support reproducible machine learning workflows. It combines experiment tracking, artifact management, model comparison, validation, and model registration in a single pipeline.
