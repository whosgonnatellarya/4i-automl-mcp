import mcp
from main import determine_task_type
from mcp.server.fastmcp import FastMCP
from preprocessing import preprocess_data
import json
from model_selection import select_model
import pandas as pd
import joblib
 

app = FastMCP()
@app.tool()
def determine_task_type_tool(target: str, column_data: str) -> str:
    """
    Determine the task type (classification or regression) based on the target CSV file and specified column data.

    Args:
        target (str): The path to the target CSV file.
        column_data (str): The name of the column to analyze.

    Returns:
        str: The determined task type ("classification", "regression", or "unknown or unsupported column data type").
    """
    return determine_task_type(target, column_data)

@app.tool()
def preprocess_data_tool(csv_path: str, target_column: str):
    """
    Preprocess the data from a CSV file by handling missing values, encoding categorical variables, and scaling numerical features.

    Args:
        csv_path (str): The path to the CSV file.
        target_column (str): The name of the target column.
    """
    X_scaled, y = preprocess_data(csv_path, target_column)
    return json.dumps({"X": X_scaled.to_json(), "y": y.to_json()})

@app.tool()
def select_model_tool(task_type: str, X_json: str, y_json: str):
    """
    Select the best model based on the task type (classification or regression) and the provided data.

    Args:
        task_type (str): The type of task ("classification" or "regression").
        X_json (str): The JSON representation of the feature data.
        y_json (str): The JSON representation of the target data.
        """

    X = pd.read_json(X_json)
    y = pd.read_json(y_json)

    best_model, best_score = select_model(task_type, X, y)
    return json.dumps({"best_model": type(best_model).__name__, "best_score": best_score})


@app.tool()
def train_model_tool(task_type: str, X_json: str, y_json: str):
    best_model, best_score = select_model(task_type, X_json, y_json)
    X = pd.read_json(X_json)
    y = pd.read_json(y_json)
    best_model.fit(X, y)
    joblib.dump(best_model, "best_model.pkl")
    return json.dumps({ "best_model": type(best_model).__name__, "best_score": best_score})

