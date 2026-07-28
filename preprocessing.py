import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(csv_path, target_column):
    df = pd.read_csv(csv_path)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna(X[col].mode()[0])

    X_encoded = pd.get_dummies(X, drop_first=True, dtype=int)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_encoded),
        columns=X_encoded.columns,
        index=X_encoded.index
    )

    return X_scaled, y