import json

import joblib
import pandas as pd
from mcp.server.fastmcp import FastMCP

from explainability import build_explanation, get_top_features
from main import determine_task_type
from model_selection import select_model
from preprocessing import preprocess_data

app = FastMCP()


@app.tool()
def determine_task_type_tool(target: str, column_data: str) -> str:
    """
    Determine whether a column is a classification or regression target.

    Args:
        target (str): Path to the CSV file.
        column_data (str): Name of the target column.

    Returns:
        str: "classification", "regression", or "unknown or unsupported column data type".
    """
    return determine_task_type(target, column_data)


@app.tool()
def preprocess_data_tool(csv_path: str, target_column: str) -> str:
    """
    Preprocess CSV data: smart missing-value imputation, encoding, scaling.
    Returns feature matrix X, target y, and a per-column missing-value report.

    Args:
        csv_path (str): Path to the CSV file.
        target_column (str): Name of the target column.

    Returns:
        str: JSON with keys "X", "y", and "missing_value_report".
    """
    X_scaled, y, mv_report = preprocess_data(csv_path, target_column)
    return json.dumps({
        "X": X_scaled.to_json(),
        "y": y.to_json(),
        "missing_value_report": mv_report,
    })


@app.tool()
def select_model_tool(task_type: str, X_json: str, y_json: str) -> str:
    """
    Run cross-validation across all candidate models, tune the winner with
    RandomizedSearchCV, and return the result with overfitting diagnostics.

    Args:
        task_type (str): "classification" or "regression".
        X_json (str): JSON representation of the feature matrix.
        y_json (str): JSON representation of the target series.

    Returns:
        str: JSON with best_model, cv_score, train_score, overfitting_risk, all_scores.
    """
    X = pd.read_json(X_json)
    y = pd.read_json(y_json, typ="series")
    result = select_model(task_type, X, y)
    return json.dumps({
        "best_model": result["model_name"],
        "cv_score": result["cv_score"],
        "train_score": result["train_score"],
        "cv_score_final": result["cv_score_final"],
        "overfitting_risk": result["overfitting_risk"],
        "all_scores": result["all_scores"],
    })


@app.tool()
def train_model_tool(task_type: str, X_json: str, y_json: str) -> str:
    """
    Train and persist the best model found by select_model.

    Args:
        task_type (str): "classification" or "regression".
        X_json (str): JSON feature matrix.
        y_json (str): JSON target series.

    Returns:
        str: JSON with best_model name, cv_score, train_score, overfitting_risk.
    """
    X = pd.read_json(X_json)
    y = pd.read_json(y_json, typ="series")
    result = select_model(task_type, X, y)
    joblib.dump(result["model"], "best_model.pkl")
    return json.dumps({
        "best_model": result["model_name"],
        "cv_score": result["cv_score"],
        "train_score": result["train_score"],
        "overfitting_risk": result["overfitting_risk"],
    })


@app.tool()
def predict_tool(new_data_json: str) -> str:
    """
    Run predictions using the persisted model (best_model.pkl).

    Args:
        new_data_json (str): JSON feature matrix for inference.

    Returns:
        str: JSON with a "predictions" list.
    """
    model = joblib.load("best_model.pkl")
    new_data = pd.read_json(new_data_json)
    predictions = model.predict(new_data)
    return json.dumps({"predictions": predictions.tolist()})


@app.tool()
def explain_model_tool(csv_path: str, target_column: str) -> str:
    """
    Run the full AutoML pipeline and return a natural-language explanation of
    why the best model was chosen, including SHAP-derived top features.

    Args:
        csv_path (str): Path to the CSV file.
        target_column (str): Name of the target column.

    Returns:
        str: Human-readable explanation of model selection and feature importance.
    """
    task_type = determine_task_type(csv_path, target_column)
    X_scaled, y, _ = preprocess_data(csv_path, target_column)
    result = select_model(task_type, X_scaled, y)

    top_features = get_top_features(result["model"], X_scaled, n=3)
    explanation = build_explanation(
        model_name=result["model_name"],
        all_scores=result["all_scores"],
        top_features=top_features,
        cv_score=result["cv_score_final"],
        task_type=task_type,
    )
    return explanation


@app.tool()
def auto_ml_pipeline_tool(csv_path: str, target_column: str) -> str:
    """
    Run the complete enhanced AutoML pipeline end-to-end:
    preprocessing → model selection → hyperparameter tuning →
    overfitting detection → SHAP explainability.

    Args:
        csv_path (str): Path to the CSV file.
        target_column (str): Name of the target column.

    Returns:
        str: JSON with task_type, best_model, cv_score, train_score,
             overfitting_risk, top_features, explanation, and missing_value_report.
    """
    task_type = determine_task_type(csv_path, target_column)
    X_scaled, y, mv_report = preprocess_data(csv_path, target_column)
    result = select_model(task_type, X_scaled, y)

    joblib.dump(result["model"], "best_model.pkl")

    top_features = get_top_features(result["model"], X_scaled, n=3)
    explanation = build_explanation(
        model_name=result["model_name"],
        all_scores=result["all_scores"],
        top_features=top_features,
        cv_score=result["cv_score_final"],
        task_type=task_type,
    )

    return json.dumps({
        "task_type": task_type,
        "best_model": result["model_name"],
        "cv_score": round(result["cv_score_final"], 4),
        "train_score": round(result["train_score"], 4),
        "overfitting_risk": result["overfitting_risk"],
        "top_features": top_features,
        "explanation": explanation,
        "missing_value_report": mv_report,
    })
