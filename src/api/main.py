# src/api/main.py
"""
API d'inférence — Superstore Profitability Prediction
Charge le modèle depuis MLflow Model Registry
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow.sklearn
import pandas as pd
import os

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME          = os.getenv("MODEL_NAME", "superstore_GradientBoosting")
MODEL_STAGE         = os.getenv("MODEL_STAGE", "Production")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ─────────────────────────────────────────
# CHARGEMENT DU MODÈLE
# ─────────────────────────────────────────

def load_model():
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    print(f"Chargement du modèle : {model_uri}")
    return mlflow.sklearn.load_model(model_uri)

model = load_model()

# ─────────────────────────────────────────
# SCHÉMA DE LA REQUÊTE
# ─────────────────────────────────────────

class TransactionInput(BaseModel):
    Sales:        float = Field(..., gt=0,  description="Chiffre de vente ($)")
    Quantity:     int   = Field(..., gt=0,  description="Quantité commandée")
    Discount:     float = Field(..., ge=0, le=1, description="Taux de remise [0-1]")
    Ship_Mode:    str   = Field(..., description="Mode d'expédition")
    Segment:      str   = Field(..., description="Segment client")
    Region:       str   = Field(..., description="Région géographique")
    Category:     str   = Field(..., description="Catégorie produit")
    Sub_Category: str   = Field(..., description="Sous-catégorie produit")
    order_month:      int   = Field(..., ge=1, le=12)
    order_quarter:    int   = Field(..., ge=1, le=4)
    order_dayofweek:  int   = Field(..., ge=0, le=6)
    shipping_delay:   int   = Field(..., ge=0)
    unit_price:       float = Field(..., gt=0)

class PredictionOutput(BaseModel):
    prediction:  int
    probability: float
    label:       str
    model_name:  str
    model_stage: str

# ─────────────────────────────────────────
# APPLICATION FASTAPI
# ─────────────────────────────────────────

app = FastAPI(
    title       = "Superstore Profitability API",
    description = "Prédit si une transaction sera rentable",
    version     = "1.0.0"
)

@app.get("/")
def root():
    return {"status": "ok", "model": MODEL_NAME, "stage": MODEL_STAGE}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionOutput)
def predict(data: TransactionInput):
    try:
        # Construit le DataFrame attendu par le pipeline sklearn
        input_df = pd.DataFrame([{
            "Sales":          data.Sales,
            "Quantity":       data.Quantity,
            "Discount":       data.Discount,
            "Ship Mode":      data.Ship_Mode,
            "Segment":        data.Segment,
            "Region":         data.Region,
            "Category":       data.Category,
            "Sub-Category":   data.Sub_Category,
            "order_month":    data.order_month,
            "order_quarter":  data.order_quarter,
            "order_dayofweek": data.order_dayofweek,
            "shipping_delay": data.shipping_delay,
            "unit_price":     data.unit_price,
        }])

        prediction  = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][prediction])
        label       = "Rentable" if prediction == 1 else "Non rentable"

        return PredictionOutput(
            prediction  = prediction,
            probability = round(probability, 4),
            label       = label,
            model_name  = MODEL_NAME,
            model_stage = MODEL_STAGE,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))