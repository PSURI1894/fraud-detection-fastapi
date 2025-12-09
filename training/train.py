import os
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -----------------------
# CONFIG
# -----------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "transactions.csv"

MODEL_DIR = BASE_DIR / "app" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "lgbm_model.pkl"


# -----------------------
# LOAD DATA
# -----------------------

def load_data(path: Path) -> pd.DataFrame:
    """
    Load the fraud dataset.

    Expected format:
      - One target column: 'is_fraud' (0/1)
      - Several feature columns (numeric + categorical)

    Modify the column names as per your actual dataset.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Place your transactions.csv file under data/."
        )
    df = pd.read_csv(path)
    return df


# -----------------------
# BUILD PIPELINES
# -----------------------

def build_preprocessor(df: pd.DataFrame, target_col: str = "is_fraud"):
    """
    Build a ColumnTransformer that handles numeric + categorical features.
    """

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, feature_cols, numeric_features, categorical_features


def train_model():
    # 1. Load data
    df = load_data(DATA_PATH)

    if "is_fraud" not in df.columns:
        raise ValueError("Dataset must contain a 'is_fraud' target column (0/1).")

    target_col = "is_fraud"
    preprocessor, feature_cols, num_cols, cat_cols = build_preprocessor(
        df, target_col=target_col
    )

    X = df[feature_cols]
    y = df[target_col].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Build LightGBM classifier
    lgbm_clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        random_state=42,
        n_jobs=-1,
    )

    # 3. Build full pipeline: preprocessor + model
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", lgbm_clf),
        ]
    )

    # 4. Train
    print("Training LightGBM model...")
    model_pipeline.fit(X_train, y_train)

    # 5. Evaluate
    y_val_proba = model_pipeline.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_val, y_val_proba)
    print(f"Validation ROC-AUC: {auc:.4f}")
    print("Classification report:")
    print(classification_report(y_val, y_val_pred))

    # 6. Persist model pipeline
    print(f"Saving model pipeline to {MODEL_PATH}")
    joblib.dump(model_pipeline, MODEL_PATH)

    print("Training complete.")


if __name__ == "__main__":
    train_model()
