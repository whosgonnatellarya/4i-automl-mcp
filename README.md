# automl-mcp

an autonomous machine learning pipeline exposed as an mcp server. feed in a csv and a target column — the system handles everything else.

## what it does

- detects task type (classification vs regression) from the target column's distribution
- preprocesses messy data with smart imputation: knnimputer for numeric columns, unknown-category handling for informative absence, hard dropping above 60% missing
- generates a missing value report per column explaining which strategy was applied and why
- tunes 5 candidate models (xgboost, lightgbm, randomforest, decisiontree, logistic/linearregression) via randomizedsearchcv
- flags overfitting when train/cv score gap exceeds 0.15
- generates shap-based explainability reports with natural language explanations of why the winning model was selected
- exposes everything as mcp tools so ai agents can trigger model selection and get interpretable results with no human in the loop

## tools

- `determine_task_type_tool` — detects classification or regression from a csv column
- `preprocess_data_tool` — cleans and prepares the data for training with smart imputation
- `select_model_tool` — tunes and picks the best model via cross-validation
- `train_model_tool` — trains the best model and saves it
- `predict_tool` — loads the trained model and returns predictions
- `explain_model_tool` — generates shap feature importance and a natural language explanation
- `auto_ml_pipeline_tool` — runs the full pipeline end to end in one call

## setup

```bash
pip install pandas scikit-learn joblib fastmcp xgboost lightgbm shap faker
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

input: invoices.csv (synthetic supply chain dataset), target column: "payment_status"

output:
```json
{
  "task_type": "classification",
  "best_model": "LGBMClassifier",
  "cv_score": 0.96,
  "train_score": 1.0,
  "overfitting_risk": "low",
  "top_features": ["amount_paid", "days_overdue", "amount_due"],
  "explanation": "LGBMClassifier was selected over RandomForestClassifier (0.96), XGBClassifier (0.95), DecisionTreeClassifier (0.94), and LogisticRegression (0.90)..."
}
```

## stack

python · scikit-learn · xgboost · lightgbm · shap · fastmcp · pandas · joblib · faker
