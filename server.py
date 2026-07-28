import mcp
from main import determine_task_type
from mcp.server.fastmcp import FastMCP
from preprocessing import preprocess_data
import json

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
