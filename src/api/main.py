"""
API d'inférence FastAPI — Superstore Profitability Prediction

Charge le modèle depuis un fichier local ou MLflow Registry.
Fournit des endpoints pour:
- Health check
- Prediction (scoring)
- Monitoring

Le modèle contient le preprocessor sklearn complet.
"""

import pickle
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config import config
from src.logging_utils import get_logger, log_execution

logger = get_logger(__name__)


# ─────────────────────────────────────────
# LIFESPAN (STARTUP/SHUTDOWN)
# ─────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application."""
    # Startup
    logger.info("🚀 Démarrage de l'API")
    try:
        get_model()
        logger.info("✅ Modèle chargé au démarrage")
    except Exception as e:
        logger.error(f"⚠️  Erreur lors du chargement du modèle: {e}")

    yield

    # Shutdown
    logger.info("🛑 Arrêt de l'API")
    _model_cache.clear()


# Initialisation FastAPI
app = FastAPI(
    title="Superstore Profitability Prediction API",
    description="API MLOps pour prédire la rentabilité des transactions Superstore",
    version="3.0.0",
    lifespan=lifespan,
)

# Activer l'instrumentation Prometheus
if config.monitoring.prometheus_enabled:
    Instrumentator().instrument(app).expose(app)


# ─────────────────────────────────────────
# CACHE MODEL
# ─────────────────────────────────────────


class ModelCache:
    """Cache du modèle avec TTL."""

    def __init__(self, ttl_seconds: int = 3600):
        self.model = None
        self.preprocessor = None
        self.loaded_at = None
        self.ttl_seconds = ttl_seconds

    def is_expired(self) -> bool:
        """Vérifie si le cache a expiré."""
        if self.loaded_at is None:
            return True
        return (
            datetime.now(timezone.utc) - self.loaded_at
        ).total_seconds() > self.ttl_seconds

    def clear(self):
        """Vide le cache."""
        self.model = None
        self.preprocessor = None
        self.loaded_at = None


_model_cache = ModelCache(ttl_seconds=config.api.cache_ttl)


# ─────────────────────────────────────────
# SCHÉMAS PYDANTIC
# ─────────────────────────────────────────


class TransactionInput(BaseModel):
    """Schéma de requête pour les prédictions."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Sales": 150.0,
                "Quantity": 2,
                "Discount": 0.1,
                "Ship_Mode": "Standard Class",
                "Segment": "Consumer",
                "Region": "East",
                "Category": "Office Supplies",
                "Sub_Category": "Paper",
                "order_month": 5,
                "order_quarter": 2,
                "order_dayofweek": 2,
                "shipping_delay": 4,
                "unit_price": 75.0,
            }
        }
    )

    Sales: float = Field(..., gt=0, description="Chiffre de vente ($)")
    Quantity: int = Field(..., gt=0, description="Quantité commandée")
    Discount: float = Field(..., ge=0, le=1, description="Taux de remise [0-1]")
    Ship_Mode: str = Field(..., min_length=1, description="Mode d'expédition")
    Segment: str = Field(..., min_length=1, description="Segment client")
    Region: str = Field(..., min_length=1, description="Région géographique")
    Category: str = Field(..., min_length=1, description="Catégorie produit")
    Sub_Category: str = Field(..., min_length=1, description="Sous-catégorie produit")
    order_month: int = Field(..., ge=1, le=12, description="Mois de la commande [1-12]")
    order_quarter: int = Field(..., ge=1, le=4, description="Trimestre [1-4]")
    order_dayofweek: int = Field(
        ..., ge=0, le=6, description="Jour de la semaine [0-6]"
    )
    shipping_delay: int = Field(..., ge=0, description="Délai de livraison (jours)")
    unit_price: float = Field(..., gt=0, description="Prix unitaire ($)")

    @field_validator("Discount")
    @classmethod
    def validate_discount(cls, v):
        """Valide le discount."""
        if not (0 <= v <= 1):
            raise ValueError("Discount doit être entre 0 et 1")
        return v


class PredictionOutput(BaseModel):
    """Schéma de réponse pour les prédictions."""

    prediction: int = Field(..., description="Classe prédite (0 ou 1)")
    probability: float = Field(
        ..., ge=0, le=1, description="Probabilité de la classe prédite"
    )
    label: str = Field(..., description="Label lisible ('Rentable' ou 'Non rentable')")
    confidence: float = Field(..., ge=0, le=1, description="Confiance [0-1]")
    model_name: str = Field(..., description="Nom du modèle")
    model_stage: str = Field(
        ..., description="Stage du modèle (Production, Staging, etc)"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp de la prédiction",
    )


class HealthStatus(BaseModel):
    """Schéma de réponse pour le health check."""

    status: str = Field(..., description="État du service")
    model_loaded: bool = Field(..., description="Modèle chargé en mémoire")
    preprocessor_loaded: bool = Field(..., description="Preprocessor chargé")
    cache_status: Optional[str] = Field(None, description="État du cache")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ErrorResponse(BaseModel):
    """Schéma d'erreur standardisé."""

    error: str
    detail: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ─────────────────────────────────────────
# CHARGEMENT DU MODÈLE
# ─────────────────────────────────────────


def load_model_and_preprocessor() -> Tuple[Any, Any]:
    """
    Charge le modèle et preprocessor depuis les fichiers.

    Returns:
        Tuple[model, preprocessor]

    Raises:
        RuntimeError: Si les fichiers ne peuvent pas être chargés
    """
    model_path = Path(config.api.model_path)
    preprocessor_path = Path(
        config.api.preprocessor_path
        or config.model.preprocessor_save_path / "preprocessor.pkl"
    )

    logger.info(f"Tentative de chargement du modèle depuis : {model_path}")

    try:
        if not model_path.exists():
            raise FileNotFoundError(f"Modèle non trouvé: {model_path}")

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info("✅ Modèle chargé avec succès")
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement du modèle : {str(e)}")
        raise RuntimeError(f"Impossible de charger le modèle : {str(e)}") from e

    # Charger preprocessor si disponible
    preprocessor = None
    if preprocessor_path.exists():
        try:
            with open(preprocessor_path, "rb") as f:
                preprocessor = pickle.load(f)
            logger.info("✅ Preprocessor chargé avec succès")
        except Exception as e:
            logger.warning(f"⚠️  Impossible de charger le preprocessor: {e}")

    return model, preprocessor


@log_execution
def get_model() -> Tuple[Any, Any]:
    """
    Récupère le modèle du cache ou le charge si nécessaire.

    Returns:
        Tuple[model, preprocessor]: Modèle et preprocessor
    """
    if _model_cache.model is None or _model_cache.is_expired():
        logger.info("🔄 Rechargement du modèle (cache expiré ou vide)")
        model, preprocessor = load_model_and_preprocessor()
        _model_cache.model = model
        _model_cache.preprocessor = preprocessor
        _model_cache.loaded_at = datetime.now(timezone.utc)

    return _model_cache.model, _model_cache.preprocessor


# ─────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────


@app.get("/", response_model=Dict[str, str])
def root() -> Dict[str, str]:
    """
    Endpoint racine.

    Returns:
        Infos sur l'API et le modèle
    """
    return {
        "status": "ok",
        "service": "Superstore Profitability Prediction API",
        "version": "3.0.0",
        "model": config.api.model_name,
        "stage": config.api.model_stage,
    }


@app.get("/health", response_model=HealthStatus)
def health() -> HealthStatus:
    """
    Health check endpoint.

    Returns:
        État du service et du modèle
    """
    try:
        model, preprocessor = get_model()
        cache_status = "valid" if not _model_cache.is_expired() else "expired"

        return HealthStatus(
            status="healthy",
            model_loaded=model is not None,
            preprocessor_loaded=preprocessor is not None,
            cache_status=cache_status,
        )
    except Exception as e:
        logger.error(f"Health check échoué: {e}")
        return HealthStatus(
            status="unhealthy",
            model_loaded=False,
            preprocessor_loaded=False,
            cache_status="error",
        )


@app.post("/predict", response_model=PredictionOutput)
def predict(data: TransactionInput) -> PredictionOutput:
    """
    Prédit si une transaction sera rentable.

    Args:
        data: Données de la transaction

    Returns:
        Prédiction avec probabilité et confiance

    Raises:
        HTTPException: Erreur lors de la prédiction
    """
    try:
        model, preprocessor = get_model()

        # Construire le DataFrame
        input_df = pd.DataFrame(
            [
                {
                    "Sales": data.Sales,
                    "Quantity": data.Quantity,
                    "Discount": data.Discount,
                    "Ship Mode": data.Ship_Mode,
                    "Segment": data.Segment,
                    "Region": data.Region,
                    "Category": data.Category,
                    "Sub-Category": data.Sub_Category,
                    "order_month": data.order_month,
                    "order_quarter": data.order_quarter,
                    "order_dayofweek": data.order_dayofweek,
                    "shipping_delay": data.shipping_delay,
                    "unit_price": data.unit_price,
                }
            ]
        )

        # Prédiction
        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
        probability = float(probabilities[prediction])
        confidence = float(max(probabilities))

        label = "Rentable 💰" if prediction == 1 else "Non rentable ❌"

        # Log prédiction si activé
        if config.api.log_predictions:
            logger.info(f"Prédiction: {label}, confidence: {confidence:.4f}")

        return PredictionOutput(
            prediction=prediction,
            probability=round(probability, 4),
            label=label,
            confidence=round(confidence, 4),
            model_name=config.api.model_name,
            model_stage=config.api.model_stage,
        )

    except Exception as e:
        logger.error(f"❌ Erreur lors de la prédiction : {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}"
        ) from e


@app.post("/batch_predict")
def batch_predict(transactions: list[TransactionInput]) -> Dict[str, Any]:
    """
    Prédiction par batch (plusieurs transactions).

    Args:
        transactions: Liste de transactions

    Returns:
        Résultats pour chaque transaction
    """
    if not transactions:
        raise HTTPException(status_code=400, detail="Liste vide")

    if len(transactions) > 1000:
        raise HTTPException(
            status_code=400, detail="Maximum 1000 transactions par batch"
        )

    results = []
    for i, transaction in enumerate(transactions):
        try:
            result = predict(transaction)
            results.append({"index": i, "result": result})
        except Exception as e:
            results.append({"index": i, "error": str(e)})

    return jsonable_encoder(
        {
            "total": len(transactions),
            "successful": sum(1 for r in results if "result" in r),
            "failed": sum(1 for r in results if "error" in r),
            "predictions": results,
        }
    )


@app.get("/model/info")
def model_info() -> Dict[str, Any]:
    """
    Informations sur le modèle actuellement chargé.

    Returns:
        Détails du modèle
    """
    try:
        model, preprocessor = get_model()

        return {
            "name": config.api.model_name,
            "stage": config.api.model_stage,
            "path": str(config.api.model_path),
            "preprocessor_path": str(config.api.preprocessor_path),
            "model_type": str(type(model).__name__),
            "cache_enabled": config.api.cache_model,
            "cache_ttl_seconds": config.api.cache_ttl,
            "monitoring_enabled": config.api.enable_monitoring,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handler personnalisé pour les HTTPException."""
    error_content = ErrorResponse(error=str(exc.status_code), detail=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=jsonable_encoder(error_content),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=config.api.host,
        port=config.api.port,
        reload=config.api.reload,
        workers=config.api.workers,
    )
