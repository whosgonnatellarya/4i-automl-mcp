import mcp
from main import determine_task_type
from mcp.server.fastmcp import FastMCP

app = FastMCP()
@app.tool()
async def determine_task_type_tool(target: str, column_data: str) -> str:
    """
    Determine the task type (classification or regression) based on the target CSV file and specified column data.

    Args:
        target (str): The path to the target CSV file.
        column_data (str): The name of the column to analyze.

    Returns:
        str: The determined task type ("classification", "regression", or "unknown or unsupported column data type").
    """
    return determine_task_type(target, column_data)
