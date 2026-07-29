
from main import determine_task_type
from preprocessing import preprocess_data
from model_selection import select_model
import joblib
import pandas as pd

csv_path = "titanic.csv"
target_column = "Survived"

# step 1
task_type = determine_task_type(csv_path, target_column)
print(f"task type: {task_type}")

# step 2
X, y = preprocess_data(csv_path, target_column)
print(f"preprocessed shape: {X.shape}")

# step 3
best_model, best_score = select_model(task_type, X, y)
print(f"best model: {type(best_model).__name__}, score: {best_score}")
best_model.fit(X, y)
joblib.dump(best_model, "best_model.pkl")

# step 4
model = joblib.load("best_model.pkl")
predictions = model.predict(X[:5])
print(f"predictions: {predictions}")
