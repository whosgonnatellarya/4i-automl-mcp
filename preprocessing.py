import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer


def preprocess_data(csv_path, target_column):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    missing_value_report = {}
    cols_to_drop = []
    knn_cols = []

    for col in X.columns:
        missing_rate = X[col].isna().mean()

        # Drop columns with >60% missing — too sparse to be useful
        if missing_rate > 0.60:
            cols_to_drop.append(col)
            missing_value_report[col] = {
                "strategy": "dropped",
                "reason": f"Missing rate {missing_rate:.1%} exceeds 60% threshold",
                "missing_rate": f"{missing_rate:.1%}",
            }
            continue

        if pd.api.types.is_numeric_dtype(X[col]):
            unique_vals = set(X[col].dropna().unique())
            is_boolean_like = unique_vals.issubset({0, 1, 0.0, 1.0}) and len(unique_vals) <= 2

            if is_boolean_like:
                mode_val = X[col].mode()[0] if not X[col].mode().empty else 0
                X[col] = X[col].fillna(mode_val)
                missing_value_report[col] = {
                    "strategy": "mode_fill",
                    "reason": "Boolean-like (0/1) column; filled with mode — flagged",
                    "missing_rate": f"{missing_rate:.1%}",
                    "flagged": True,
                }
            elif 0 < missing_rate < 0.20:
                knn_cols.append(col)
                missing_value_report[col] = {
                    "strategy": "KNNImputer",
                    "reason": f"Low missing rate ({missing_rate:.1%}); KNN preserves local data structure",
                    "missing_rate": f"{missing_rate:.1%}",
                }
            elif missing_rate >= 0.20:
                X[col] = X[col].fillna(X[col].median())
                missing_value_report[col] = {
                    "strategy": "median_fill",
                    "reason": f"Moderate/high missing rate ({missing_rate:.1%}); median is robust to outliers",
                    "missing_rate": f"{missing_rate:.1%}",
                }
            else:
                missing_value_report[col] = {
                    "strategy": "none_needed",
                    "reason": "No missing values",
                    "missing_rate": "0.0%",
                }

        else:
            # Categorical: missingness >5% → treat as its own "Unknown" category
            if missing_rate > 0.05:
                X[col] = X[col].fillna("Unknown")
                missing_value_report[col] = {
                    "strategy": "unknown_category",
                    "reason": f"Missing rate ({missing_rate:.1%}) suggests absence is itself informative",
                    "missing_rate": f"{missing_rate:.1%}",
                }
            elif missing_rate > 0:
                mode_val = X[col].mode()[0] if not X[col].mode().empty else "Unknown"
                X[col] = X[col].fillna(mode_val)
                missing_value_report[col] = {
                    "strategy": "mode_fill",
                    "reason": f"Low missing rate ({missing_rate:.1%}); filled with mode",
                    "missing_rate": f"{missing_rate:.1%}",
                }
            else:
                missing_value_report[col] = {
                    "strategy": "none_needed",
                    "reason": "No missing values",
                    "missing_rate": "0.0%",
                }

    X = X.drop(columns=cols_to_drop)

    # Apply KNN imputation to eligible numeric columns
    valid_knn_cols = [c for c in knn_cols if c in X.columns]
    if valid_knn_cols:
        imputer = KNNImputer(n_neighbors=5)
        X[valid_knn_cols] = imputer.fit_transform(X[valid_knn_cols])

    # Drop high-cardinality categorical columns only (numeric high-cardinality is fine)
    high_card_cols = [
        col for col in X.columns
        if not pd.api.types.is_numeric_dtype(X[col]) and X[col].nunique() > 50
    ]
    X = X.drop(columns=high_card_cols)

    X_encoded = pd.get_dummies(X, drop_first=True, dtype=int)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_encoded),
        columns=X_encoded.columns,
        index=X_encoded.index,
    )

    return X_scaled, y, missing_value_report
