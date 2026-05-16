"""
Module de logging centralisé avec structured logging support.
Fournit des loggers configurés et des décorateurs utilitaires.
"""

import json
import logging
import sys
from datetime import datetime
from functools import wraps
from typing import Any, Callable

from src.config import config


class JSONFormatter(logging.Formatter):
    """Formateur JSON pour structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Formate un log en JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Ajouter les extras
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        return json.dumps(log_data)


def get_logger(name: str) -> logging.Logger:
    """
    Récupère un logger configuré.

    Args:
        name: Nom du logger (généralement __name__)

    Returns:
        logging.Logger: Logger configuré
    """
    logger = logging.getLogger(name)

    # Configurer si pas encore fait
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)

        if config.logging.use_json:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                config.logging.format, datefmt=config.logging.date_format
            )

        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(config.logging.level)

    return logger


def log_execution(func: Callable) -> Callable:
    """
    Décorateur pour logger l'exécution d'une fonction.

    Exemple:
        @log_execution
        def my_function():
            pass
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        logger = get_logger(func.__module__)
        logger.info(f"Démarrage: {func.__name__}")

        try:
            result = func(*args, **kwargs)
            logger.info(f"Succès: {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"Erreur dans {func.__name__}: {str(e)}", exc_info=True)
            raise

    return wrapper


def log_performance(func: Callable) -> Callable:
    """
    Décorateur pour logger les performances d'une fonction.
    Affiche le temps d'exécution.
    """
    import time

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        logger = get_logger(func.__module__)
        start = time.time()

        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.info(f"{func.__name__} exécuté en {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"{func.__name__} échoué après {elapsed:.2f}s: {str(e)}")
            raise

    return wrapper


class LogContext:
    """Context manager pour ajouter du contexte aux logs."""

    def __init__(self, **context):
        self.context = context
        self.logger = get_logger(__name__)

    def __enter__(self):
        """Entre dans le contexte."""
        self.logger.info(f"Contexte: {self.context}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Quitte le contexte."""
        if exc_type:
            self.logger.error(f"Erreur dans contexte: {exc_val}")
            return False
        self.logger.info("Contexte fermé")
        return True


# Loggers prédéfinis
data_logger = get_logger("superstore.data")
model_logger = get_logger("superstore.model")
api_logger = get_logger("superstore.api")
training_logger = get_logger("superstore.training")
validation_logger = get_logger("superstore.validation")


__all__ = [
    "get_logger",
    "log_execution",
    "log_performance",
    "LogContext",
    "JSONFormatter",
    "data_logger",
    "model_logger",
    "api_logger",
    "training_logger",
    "validation_logger",
]
