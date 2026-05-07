# src/api/main.py
"""
API d'inférence — Superstore Profitability Prediction
Charge le modèle depuis un fichier local (variable d'environnement MODEL_PATH)
"""

from fastapi import FastAPI, HTTPException, Depends
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import os
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

# Récupère le chemin du modèle depuis l'environnement
MODEL_PATH = os.getenv("MODEL_PATH", "/app/mlartifacts/1/models/m-2acd3303576e4ab2967290e07fa3d929/artifacts/model.pkl")
MODEL_NAME = os.getenv("MODEL_NAME", "superstore_GradientBoosting")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

logger.info(f"Configuration: MODEL_PATH={MODEL_PATH}, MODEL_NAME={MODEL_NAME}")

# Cache pour le modèle
_model = None

# ─────────────────────────────────────────
# CHARGEMENT DU MODÈLE (LAZY)
# ─────────────────────────────────────────

def get_model():
    global _model
    if _model is None:
        logger.info(f"Tentative de chargement du modèle depuis : {MODEL_PATH}")
        try:
            with open(MODEL_PATH, 'rb') as f:
                _model = pickle.load(f)
            logger.info("✅ Modèle chargé avec succès.")
        except FileNotFoundError:
            logger.error(f"❌ Fichier non trouvé: {MODEL_PATH}")
            raise RuntimeError(f"Impossible de charger le modèle : fichier {MODEL_PATH} non trouvé")
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du modèle : {str(e)}")
            raise RuntimeError(f"Impossible de charger le modèle : {str(e)}")
    return _model

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
    version     = "2.0.0"
)

Instrumentator().instrument(app).expose(app)

@app.get("/")
def root():
    return {
        "status": "ok", 
        "model": MODEL_NAME, 
        "stage": MODEL_STAGE,
        "model_path": MODEL_PATH,
    }

@app.get("/health")
def health():
    model_loaded = _model is not None
    status = "healthy" if model_loaded else "warming_up"
    return {"status": status, "model_loaded": model_loaded}

@app.post("/predict", response_model=PredictionOutput)
def predict(data: TransactionInput, model=Depends(get_model)):
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
        logger.error(f"❌ Erreur lors de la prédiction : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
