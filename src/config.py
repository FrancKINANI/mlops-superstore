"""
Configuration centralisée pour le projet MLOps Superstore.
Gère tous les paramètres de configuration via variables d'environnement.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration des données."""

    raw_data_path: Path = Path(os.getenv("RAW_DATA_PATH", "data/raw/Superstore.csv"))
    processed_data_path: Path = Path(
        os.getenv("PROCESSED_DATA_PATH", "data/processed/Superstore_processed.csv")
    )
    test_size: float = float(os.getenv("TEST_SIZE", 0.2))
    random_state: int = int(os.getenv("RANDOM_STATE", 42))

    def __post_init__(self):
        """Valide les chemins de données."""
        self.raw_data_path = Path(self.raw_data_path)
        self.processed_data_path = Path(self.processed_data_path)
        self.processed_data_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def raw_data_exists(self) -> bool:
        """Vérifie si les données brutes existent."""
        return self.raw_data_path.exists()


@dataclass
class PreprocessingConfig:
    """Configuration du pipeline de preprocessing."""

    # Features catégorielles et numériques
    cat_features: list = None
    num_features: list = None
    cols_to_drop: list = None

    def __post_init__(self):
        """Initialise les listes de features."""
        if self.cat_features is None:
            self.cat_features = [
                "Ship Mode",
                "Segment",
                "Region",
                "Category",
                "Sub-Category",
            ]
        if self.num_features is None:
            self.num_features = [
                "Sales",
                "Quantity",
                "Discount",
                "order_month",
                "order_quarter",
                "order_dayofweek",
                "shipping_delay",
                "unit_price",
            ]
        if self.cols_to_drop is None:
            self.cols_to_drop = [
                "Row ID",
                "Order ID",
                "Customer ID",
                "Product ID",
                "Customer Name",
                "Product Name",
                "Order Date",
                "Ship Date",
                "Profit",
                "Country",
                "City",
                "State",
                "Postal Code",
            ]


@dataclass
class MLFlowConfig:
    """Configuration MLflow."""

    tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name: str = os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "superstore-profitability"
    )
    registry_uri: Optional[str] = os.getenv("MLFLOW_REGISTRY_URI", None)
    backend_store_uri: Optional[str] = os.getenv("MLFLOW_BACKEND_STORE_URI", None)

    def __post_init__(self):
        """Valide la configuration MLflow."""
        if not self.tracking_uri:
            logger.warning(
                "MLFLOW_TRACKING_URI non défini, utilisant valeur par défaut"
            )


@dataclass
class ModelConfig:
    """Configuration des modèles."""

    model_registry_path: Path = Path(os.getenv("MODEL_REGISTRY_PATH", "mlartifacts"))
    preprocessor_save_path: Path = Path(
        os.getenv("PREPROCESSOR_SAVE_PATH", "mlartifacts/preprocessor")
    )
    model_format: str = os.getenv("MODEL_FORMAT", "pkl")  # pkl ou joblib
    save_preprocessor: bool = os.getenv("SAVE_PREPROCESSOR", "true").lower() == "true"

    def __post_init__(self):
        """Initialise les chemins."""
        self.model_registry_path = Path(self.model_registry_path)
        self.preprocessor_save_path = Path(self.preprocessor_save_path)
        self.model_registry_path.mkdir(parents=True, exist_ok=True)
        self.preprocessor_save_path.mkdir(parents=True, exist_ok=True)


@dataclass
class APIConfig:
    """Configuration de l'API FastAPI."""

    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", 8000))
    reload: bool = os.getenv("API_RELOAD", "false").lower() == "true"
    workers: int = int(os.getenv("API_WORKERS", 1))

    model_path: str = os.getenv(
        "MODEL_PATH",
        "mlartifacts/1/models/m-2acd3303576e4ab2967290e07fa3d929/artifacts/model.pkl",
    )
    preprocessor_path: Optional[str] = os.getenv("PREPROCESSOR_PATH", None)
    model_name: str = os.getenv("MODEL_NAME", "superstore_GradientBoosting")
    model_stage: str = os.getenv("MODEL_STAGE", "Production")

    # Monitoring
    enable_monitoring: bool = os.getenv("ENABLE_MONITORING", "true").lower() == "true"
    log_predictions: bool = os.getenv("LOG_PREDICTIONS", "false").lower() == "true"

    # Performance
    cache_model: bool = os.getenv("CACHE_MODEL", "true").lower() == "true"
    cache_ttl: int = int(os.getenv("CACHE_TTL", 3600))


@dataclass
class AirflowConfig:
    """Configuration Airflow."""

    project_root: Path = Path(
        os.getenv(
            "PROJECT_ROOT",
            Path(__file__).resolve().parent.parent,
        )
    )
    dags_folder: Path = None
    logs_folder: Path = None
    max_retries: int = int(os.getenv("AIRFLOW_MAX_RETRIES", 1))
    retry_delay_minutes: int = int(os.getenv("AIRFLOW_RETRY_DELAY_MINUTES", 2))

    def __post_init__(self):
        """Initialise les chemins Airflow."""
        self.dags_folder = self.project_root / "dags"
        self.logs_folder = self.project_root / "logs"
        self.dags_folder.mkdir(parents=True, exist_ok=True)
        self.logs_folder.mkdir(parents=True, exist_ok=True)


@dataclass
class MonitoringConfig:
    """Configuration du monitoring."""

    prometheus_enabled: bool = os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true"
    grafana_enabled: bool = os.getenv("GRAFANA_ENABLED", "true").lower() == "true"
    drift_detection_enabled: bool = (
        os.getenv("DRIFT_DETECTION_ENABLED", "true").lower() == "true"
    )

    prometheus_config_path: Path = Path(
        os.getenv("PROMETHEUS_CONFIG_PATH", "monitoring/prometheus.yml")
    )
    drift_report_path: Path = Path(
        os.getenv("DRIFT_REPORT_PATH", "monitoring/reports/drift_report.html")
    )

    def __post_init__(self):
        """Initialise les chemins de monitoring."""
        self.drift_report_path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class LoggingConfig:
    """Configuration du logging."""

    level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    use_json: bool = os.getenv("USE_JSON_LOGGING", "false").lower() == "true"


class Config:
    """Configuration globale du projet."""

    def __init__(self):
        """Initialise toutes les configurations."""
        self.data = DataConfig()
        self.preprocessing = PreprocessingConfig()
        self.mlflow = MLFlowConfig()
        self.model = ModelConfig()
        self.api = APIConfig()
        self.airflow = AirflowConfig()
        self.monitoring = MonitoringConfig()
        self.logging = LoggingConfig()

        # Initialiser le logging
        self._setup_logging()

    def _setup_logging(self):
        """Configure le logging global."""
        import logging.config

        logging.basicConfig(
            level=self.logging.level,
            format=self.logging.format,
            datefmt=self.logging.date_format,
        )

    def get_env(self, key: str, default: str = None) -> str:
        """Récupère une variable d'environnement."""
        return os.getenv(key, default)

    def __repr__(self) -> str:
        """Représentation string de la config."""
        return (
            f"Config(\n"
            f"  data_path={self.data.raw_data_path},\n"
            f"  mlflow_uri={self.mlflow.tracking_uri},\n"
            f"  api_port={self.api.port},\n"
            f"  log_level={self.logging.level}\n"
            f")"
        )


# Instance globale (singleton)
config = Config()


if __name__ == "__main__":
    # Test
    print(config)
    print(f"\nData config exists: {config.data.raw_data_exists}")
    print(f"Cat features: {config.preprocessing.cat_features}")
