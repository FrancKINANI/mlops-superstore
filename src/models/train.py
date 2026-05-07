# src/models/train.py
"""
Entraînement et tracking MLflow des modèles de classification.
Problème : prédire is_profitable (1 = transaction rentable, 0 = perte)
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, accuracy_score, classification_report
)
from sklearn.pipeline import Pipeline

import sys
import os
import git
import yaml
# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.preprocessing import run_preprocessing, build_preprocessor

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# HELPERS VERSIONING
# ─────────────────────────────────────────

def get_git_revision_hash() -> str:
    """Récupère le commit hash actuel."""
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha
    except Exception:
        return "unknown"

def get_dvc_data_version(path: str) -> str:
    """Récupère le MD5 du fichier de données depuis son fichier .dvc."""
    try:
        dvc_path = path + ".dvc"
        if os.path.exists(dvc_path):
            with open(dvc_path, 'r') as f:
                dvc_data = yaml.safe_load(f)
                return dvc_data['outs'][0]['md5']
        return "unknown"
    except Exception:
        return "unknown"


# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

MLFLOW_TRACKING_URI = "http://localhost:5000"  # ou ton URL ngrok si depuis Colab
EXPERIMENT_NAME     = "superstore-profitability"
DATA_PATH           = "data/raw/Superstore.csv"
PROCESSED_PATH      = "data/processed/Superstore_processed.csv"

TEST_SIZE           = 0.2
RANDOM_STATE        = 42


# ─────────────────────────────────────────
# MODÈLES À COMPARER
# ─────────────────────────────────────────

MODELS = {
    "LogisticRegression": {
        "model": LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
        "params": {
            "class_weight": "balanced",
            "max_iter": 1000,
            "solver": "lbfgs"
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "params": {
            "n_estimators": 100,
            "class_weight": "balanced",
            "max_depth": "None"
        }
    },
    "GradientBoosting": {
        "model": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=RANDOM_STATE
        ),
        "params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 4
        }
    }
}


# ─────────────────────────────────────────
# FONCTION D'ÉVALUATION
# ─────────────────────────────────────────

def evaluate(model, X_test, y_test) -> dict:
    """Calcule toutes les métriques de classification."""
    y_pred     = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    return {
        "f1_score":   f1_score(y_test, y_pred),
        "precision":  precision_score(y_test, y_pred),
        "recall":     recall_score(y_test, y_pred),
        "roc_auc":    roc_auc_score(y_test, y_pred_proba),
        "accuracy":   accuracy_score(y_test, y_pred)
    }


# ─────────────────────────────────────────
# BOUCLE D'ENTRAÎNEMENT + TRACKING MLFLOW
# ─────────────────────────────────────────

def train_all_models():

    # 1. Données
    X, y = run_preprocessing(DATA_PATH, PROCESSED_PATH)

    # 2. Split stratifié — essentiel pour dataset déséquilibré
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y           # garantit la proportion 80/20 dans train ET test
    )
    logger.info(f"Train : {X_train.shape[0]} | Test : {X_test.shape[0]}")

    # 3. Preprocessor (fit sur train uniquement — règle absolue)
    preprocessor = build_preprocessor()

    # 4. Configuration MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    best_f1    = 0
    best_model = None
    best_name  = ""

    # 5. Boucle sur les modèles
    for model_name, config in MODELS.items():
        logger.info(f"\nEntraînement : {model_name}")

        with mlflow.start_run(run_name=model_name):

            # ── Versioning : Git & DVC ──
            mlflow.set_tag("git_commit", get_git_revision_hash())
            mlflow.set_tag("data_version", get_dvc_data_version(DATA_PATH))

            # Pipeline complet : preprocessing + modèle
            full_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier',   config["model"])
            ])

            # Entraînement
            full_pipeline.fit(X_train, y_train)

            # Évaluation
            metrics = evaluate(full_pipeline, X_test, y_test)

            # Cross-validation F1 (plus robuste)
            cv_f1 = cross_val_score(
                full_pipeline, X_train, y_train,
                cv=5, scoring='f1'
            ).mean()

            # ── MLflow : log des paramètres ──
            mlflow.log_param("model_type",   model_name)
            mlflow.log_param("test_size",    TEST_SIZE)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("stratify",     True)
            mlflow.log_params(config["params"])

            # ── MLflow : log des métriques ──
            mlflow.log_metrics(metrics)
            mlflow.log_metric("cv_f1_mean", cv_f1)

            # ── MLflow : log du modèle ──
            mlflow.sklearn.log_model(
                full_pipeline,
                artifact_path="model",
                registered_model_name=f"superstore_{model_name}"
            )

            # Affichage console
            logger.info(f"  F1       : {metrics['f1_score']:.4f}")
            logger.info(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
            logger.info(f"  CV F1    : {cv_f1:.4f}")

            # Suivi du meilleur modèle
            if metrics['f1_score'] > best_f1:
                best_f1    = metrics['f1_score']
                best_model = full_pipeline
                best_name  = model_name

    logger.info(f"\n Meilleur modèle : {best_name} (F1 = {best_f1:.4f})")
    return best_name, best_f1


# ─────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────

if __name__ == "__main__":
    best_name, best_f1 = train_all_models()
    print(f"\nMeilleur modèle : {best_name}")
    print(f"F1-Score        : {best_f1:.4f}")