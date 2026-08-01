# automl-mcp

an autonomous machine learning pipeline exposed as an mcp server. feed in a csv and a target column, and the system automatically preprocesses the data, detects whether it's a classification or regression problem, selects the best model, and trains it - no human decisions required.

## what it does

- detects task type (classification vs regression) from your data
- handles missing values, encodes categorical variables, scales features
- tries multiple models and picks the best one using cross validation
- exposes everything as mcp tools so ai agents can use the pipeline directly

## tools

- `determine_task_type_tool` - detects classification or regression from a csv column
- `preprocess_data_tool` - cleans and prepares the data for training
- `select_model_tool` - picks the best model for the task
- `train_model_tool` - trains the best model and saves it
- `predict_tool` - loads the trained model and returns predictions
- `auto_ml_pipeline_tool` - runs the full pipeline end to end in one call

## setup

```bash
pip install pandas scikit-learn joblib fastmcp
```

## usage

start the mcp server:

```bash
python server.py
```

or run the pipeline directly:

```bash
python test_pipeline.py
```

## example

input: titanic.csv, target column: "Survived"

output:

```json
{
  "task_type": "classification",
  "best_model": "RandomForestClassifier",
  "best_score": 0.797
}
```

## stack

python, scikit-learn, fastmcp, pandas, joblib
