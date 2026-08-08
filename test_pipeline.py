import joblib

from explainability import build_explanation, get_top_features
from main import determine_task_type
from model_selection import select_model
from preprocessing import preprocess_data

csv_path = "titanic.csv"
target_column = "Survived"

# Step 1 — task detection
task_type = determine_task_type(csv_path, target_column)
print(f"task type: {task_type}")

# Step 2 — preprocessing
X, y, mv_report = preprocess_data(csv_path, target_column)
print(f"preprocessed shape: {X.shape}")
print("missing value report:")
for col, info in mv_report.items():
    print(f"  {col}: {info['strategy']} — {info['reason']}")

# Step 3 — model selection + tuning + overfitting detection
result = select_model(task_type, X, y)
print(f"\nbest model : {result['model_name']}")
print(f"cv score   : {result['cv_score_final']:.4f}")
print(f"train score: {result['train_score']:.4f}")
print(f"overfit    : {result['overfitting_risk']}")
print(f"all scores : {result['all_scores']}")

joblib.dump(result["model"], "best_model.pkl")

# Step 4 — SHAP explainability
top_features = get_top_features(result["model"], X, n=3)
explanation = build_explanation(
    model_name=result["model_name"],
    all_scores=result["all_scores"],
    top_features=top_features,
    cv_score=result["cv_score_final"],
    task_type=task_type,
)
print(f"\nexplanation:\n{explanation}")

# Step 5 — predictions on first 5 rows
model = joblib.load("best_model.pkl")
predictions = model.predict(X[:5])
print(f"\npredictions: {predictions}")
