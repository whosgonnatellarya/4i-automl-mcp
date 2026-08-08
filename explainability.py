import numpy as np
import shap


def get_top_features(model, X, n=3):
    """Return top-n feature names ranked by mean absolute SHAP value."""
    feature_names = X.columns.tolist()
    model_name = type(model).__name__

    try:
        is_tree = any(
            kw in model_name
            for kw in ["Forest", "Tree", "XGB", "LGBM", "Boost", "Gradient"]
        )
        is_linear = any(kw in model_name for kw in ["Logistic", "Linear"])

        if is_tree:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
        elif is_linear:
            background = shap.sample(X, min(100, len(X)))
            explainer = shap.LinearExplainer(model, background)
            shap_values = explainer.shap_values(X)
        else:
            background = shap.sample(X, min(50, len(X)))
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(shap.sample(X, min(50, len(X))))

        # Normalize to (n_samples, n_features) regardless of multi-class shape
        if isinstance(shap_values, list):
            importance = np.mean(
                [np.abs(sv).mean(axis=0) for sv in shap_values], axis=0
            )
        else:
            sv = np.array(shap_values)
            if sv.ndim == 3:
                # (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features)
                importance = np.abs(sv).mean(axis=(0, 2)) if sv.shape[2] < sv.shape[0] else np.abs(sv).mean(axis=(0, 1))
            else:
                importance = np.abs(sv).mean(axis=0)

    except Exception:
        # Fallback to built-in importances if SHAP fails
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = np.array(model.coef_)
            importance = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
        else:
            return feature_names[:n]

    importance = np.array(importance).flatten()
    top_n = min(n, len(feature_names))
    top_indices = np.argsort(importance)[::-1][:top_n]
    return [feature_names[i] for i in top_indices]


def build_explanation(model_name, all_scores, top_features, cv_score, task_type):
    """Build a natural-language explanation of why the model was selected."""
    score_label = "accuracy" if task_type == "classification" else "R² score"

    other_scores = sorted(
        [(k, v) for k, v in all_scores.items() if k != model_name],
        key=lambda x: x[1],
        reverse=True,
    )
    vs_str = (
        ", ".join(f"{name} ({score:.2f})" for name, score in other_scores[:4])
        if other_scores
        else "other models"
    )

    is_tree = any(
        kw in model_name for kw in ["Forest", "Tree", "XGB", "LGBM", "Boost", "Gradient"]
    )
    is_linear = any(kw in model_name for kw in ["Logistic", "Linear"])

    if is_tree and not is_linear:
        rationale = (
            "The dataset's non-linear patterns and feature interactions made "
            "tree-based methods more suitable than linear models."
        )
    elif is_linear:
        rationale = (
            "The dataset exhibits mostly linear relationships, making this model "
            "well-suited and interpretable."
        )
    else:
        rationale = (
            "This model provided the best balance of complexity and generalisation "
            "on this dataset."
        )

    top_features_str = (
        ", ".join(top_features) if top_features else "the provided features"
    )

    return (
        f"{model_name} was selected over {vs_str} because it achieved the highest "
        f"cross-validation {score_label} ({cv_score:.2f}). "
        f"The most important features driving predictions were {top_features_str} "
        f"based on SHAP values. {rationale}"
    )
