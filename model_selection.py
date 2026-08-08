import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import r2_score
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

PARAM_GRIDS = {
    "RandomForestClassifier": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
    },
    "RandomForestRegressor": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10],
    },
    "LogisticRegression": {"C": [0.1, 1, 10], "max_iter": [100, 200]},
    "LinearRegression": {},
    "DecisionTreeClassifier": {"max_depth": [None, 3, 5, 10]},
    "DecisionTreeRegressor": {"max_depth": [None, 3, 5, 10]},
    "XGBClassifier": {
        "n_estimators": [50, 100],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5],
    },
    "XGBRegressor": {
        "n_estimators": [50, 100],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5],
    },
    "LGBMClassifier": {
        "n_estimators": [50, 100],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5],
    },
    "LGBMRegressor": {
        "n_estimators": [50, 100],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5],
    },
}


def _get_models(task_type):
    if task_type == "classification":
        return {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
            "RandomForestClassifier": RandomForestClassifier(random_state=42),
            "XGBClassifier": XGBClassifier(verbosity=0, random_state=42),
            "LGBMClassifier": LGBMClassifier(verbose=-1, random_state=42),
        }
    elif task_type == "regression":
        return {
            "LinearRegression": LinearRegression(),
            "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
            "RandomForestRegressor": RandomForestRegressor(random_state=42),
            "XGBRegressor": XGBRegressor(verbosity=0, random_state=42),
            "LGBMRegressor": LGBMRegressor(verbose=-1, random_state=42),
        }
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def select_model(task_type, X, y):
    models = _get_models(task_type)
    scoring = "accuracy" if task_type == "classification" else "neg_mean_squared_error"

    # Step 1 — cross-validate all candidates and pick the best
    best_cv_score = -float("inf")
    best_model_name = None
    all_scores = {}

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring=scoring)
        mean_score = float(np.mean(scores))
        all_scores[name] = mean_score
        if mean_score > best_cv_score:
            best_cv_score = mean_score
            best_model_name = name

    best_model = models[best_model_name]

    # Step 2 — tune the winner with RandomizedSearchCV
    param_grid = PARAM_GRIDS.get(best_model_name, {})
    if param_grid:
        search = RandomizedSearchCV(
            best_model,
            param_grid,
            n_iter=10,
            cv=5,
            scoring=scoring,
            random_state=42,
            n_jobs=-1,
        )
        search.fit(X, y)
        best_model = search.best_estimator_
    else:
        best_model.fit(X, y)

    # Step 3 — overfitting detection using consistent R²/accuracy scoring
    ov_scoring = "accuracy" if task_type == "classification" else "r2"
    cv_scores_ov = cross_val_score(best_model, X, y, cv=5, scoring=ov_scoring)
    cv_score_final = float(np.mean(cv_scores_ov))

    if task_type == "classification":
        train_score = float(best_model.score(X, y))
    else:
        train_score = float(r2_score(y, best_model.predict(X)))

    overfitting_risk = "high" if (train_score - cv_score_final) > 0.15 else "low"

    return {
        "model": best_model,
        "model_name": best_model_name,
        "cv_score": best_cv_score,
        "train_score": train_score,
        "cv_score_final": cv_score_final,
        "overfitting_risk": overfitting_risk,
        "all_scores": all_scores,
    }
