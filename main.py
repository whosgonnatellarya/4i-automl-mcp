import pandas as pd

def determine_task_type(target, column_data):
    
    df = pd.read_csv(target)
    col_data = df[column_data]

    if pd.api.types.is_string_dtype(col_data) or col_data.dtype == object:
        return "classification"
    elif pd.api.types.is_float_dtype(col_data):
        return "regression"

    elif pd.api.types.is_integer_dtype(col_data):
        unique_ratio = col_data.nunique() / len(col_data)
        
        if unique_ratio < 0.05:
            return "classification"
        else:
            return "regression"

    else:
        return "unknown or unsupported column data type"