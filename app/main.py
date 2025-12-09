from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# -----------------------
# PATHS & MODEL LOADING
# -----------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "app" / "models" / "lgbm_model.pkl"

if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model file not found at {MODEL_PATH}. "
        "Run `python training/train.py` locally and ensure the "
        "generated model is committed/deployed."
    )

# Load the full pipeline (preprocessor + LightGBM model)
model_pipeline = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection using LightGBM and FastAPI.",
    version="1.0.0",
)


# -----------------------
# REQUEST / RESPONSE MODELS
# -----------------------

class TransactionInput(BaseModel):
    """
    Define the features expected by the model.

    Adjust fields to match the columns used in training.
    """

    amount: float = Field(..., description="Transaction amount")
    oldbalanceOrg: float = Field(
        ...,
        description="Original account balance before transaction",
    )
    newbalanceOrig: float = Field(
        ...,
        description="New account balance after transaction",
    )
    transaction_type: str = Field(
        ...,
        description="Type of transaction, e.g., CASH_OUT, PAYMENT",
    )
    # Add more fields here if your dataset includes them


class FraudPredictionResponse(BaseModel):
    is_fraud: int
    fraud_probability: float


# -----------------------
# ROUTES
# -----------------------

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok", "message": "Fraud Detection API is running"}


@app.post("/predict", response_model=FraudPredictionResponse, tags=["Prediction"])
def predict_fraud(transaction: TransactionInput):
    """
    Make a fraud prediction for a single transaction.
    """

    # Convert Pydantic model to a one-row DataFrame
    input_df = pd.DataFrame([transaction.dict()])

    # Model pipeline handles preprocessing + prediction
    proba = model_pipeline.predict_proba(input_df)[0, 1]
    label = int(proba >= 0.5)

    return FraudPredictionResponse(
        is_fraud=label,
        fraud_probability=float(proba),
    )
