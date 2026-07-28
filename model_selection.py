from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np

def select_model(task_type, X, y):
    if task_type == "classification":
        models = {
            "LogisticRegression": LogisticRegression(),
            "DecisionTree": DecisionTreeClassifier(),
            "RandomForest": RandomForestClassifier()
        }
        scoring = "accuracy"
        best_score = -float("inf")
        
    elif task_type == "regression":
        models = {
            "LinearRegression": LinearRegression(),
            "DecisionTree": DecisionTreeRegressor(),
            "RandomForest": RandomForestRegressor()
        }
        scoring = "neg_mean_squared_error"
        best_score = -float("inf")

    best_model = None

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring=scoring)
        mean_score = np.mean(scores)

        if mean_score > best_score:
            best_score = mean_score
            best_model = model

    return best_model, best_score