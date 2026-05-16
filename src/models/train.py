# src/models/train.py
"""
Entraînement et tracking MLflow des modèles de classification.
Problème : prédire is_profitable (1 = transaction rentable, 0 = perte)

Pipeline:
1. Chargement et preprocessing des données
2. Entraînement de plusieurs modèles
3. Évaluation et comparaison
4. Logging dans MLflow
5. Sauvegarde du preprocessor avec le modèle
"""

import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import git
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.config import config
from src.data.preprocessing import build_preprocessor, run_preprocessing
from src.logging_utils import get_logger, log_execution, log_performance

logger = get_logger(__name__)


# ─────────────────────────────────────────
# HELPERS VERSIONING
# ─────────────────────────────────────────


def get_git_revision_hash() -> str:
    """
    Récupère le commit hash actuel du repository Git.

    Returns:
        str: Commit SHA, ou "unknown" si erreur
    """
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha
    except Exception as e:
        logger.warning(f"Impossible de récupérer git commit: {e}")
        return "unknown"


def get_dvc_data_version(path: str) -> str:
    """
    Récupère le MD5 du fichier de données depuis son fichier .dvc.

    Args:
        path: Chemin du fichier de données

    Returns:
        str: MD5 hash, ou "unknown" si fichier .dvc manquant
    """
    try:
        dvc_path = Path(path).with_suffix(".dvc")
        if not dvc_path.exists():
            logger.warning(f"Fichier .dvc non trouvé: {dvc_path}")
            return "unknown"

        with open(dvc_path, "r") as f:
            dvc_data = yaml.safe_load(f)
            return dvc_data.get("outs", [{}])[0].get("md5", "unknown")
    except Exception as e:
        logger.warning(f"Impossible de récupérer version DVC: {e}")
        return "unknown"


def save_preprocessor(preprocessor: ColumnTransformer, path: Path = None) -> Path:
    """
    Sauvegarde le preprocessor sklearn pour réutilisation en inférence.

    IMPORTANT: Le preprocessor doit être fit sur les données d'entraînement!

    Args:
        preprocessor: ColumnTransformer sklearn
        path: Chemin de sauvegarde (défaut: config.model.preprocessor_save_path)

    Returns:
        Path: Chemin où le preprocessor a été sauvegardé
    """
    path = path or config.model.preprocessor_save_path / "preprocessor.pkl"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(path, "wb") as f:
            pickle.dump(preprocessor, f)
        logger.info(f"Preprocessor sauvegardé : {path}")
        return path
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du preprocessor: {e}")
        raise


# ─────────────────────────────────────────
# CONFIGURATION DES MODÈLES
# ─────────────────────────────────────────


def get_models_config() -> Dict[str, Dict[str, Any]]:
    """
    Retourne la configuration des modèles à entraîner.

    Returns:
        Dict: Configuration {nom_modèle: {model: instance, params: dict}}
    """
    return {
        "LogisticRegression": {
            "model": LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=config.data.random_state,
                solver="lbfgs",
            ),
            "params": {"class_weight": "balanced", "max_iter": 1000, "solver": "lbfgs"},
        },
        "RandomForest": {
            "model": RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=config.data.random_state,
                n_jobs=-1,
                max_depth=10,
            ),
            "params": {
                "n_estimators": 100,
                "class_weight": "balanced",
                "max_depth": 10,
                "n_jobs": -1,
            },
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                random_state=config.data.random_state,
                subsample=0.8,
            ),
            "params": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 4,
                "subsample": 0.8,
            },
        },
    }


# ─────────────────────────────────────────
# ÉVALUATION
# ─────────────────────────────────────────


def evaluate_model(
    model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, float]:
    """
    Évalue un modèle sur l'ensemble de test.

    Métriques:
    - F1-Score (harmonie précision/recall)
    - Precision
    - Recall
    - ROC-AUC
    - Accuracy
    - Confusion Matrix

    Args:
        model: Pipeline sklearn entraîné
        X_test: Features de test
        y_test: Labels de test

    Returns:
        Dict[str, float]: Dictionnaire des métriques

    Raises:
        ValueError: Si le modèle n'est pas entraîné
    """
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "f1_score": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "accuracy": accuracy_score(y_test, y_pred),
        }

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0

        logger.debug(f"Métriques calculées: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation: {e}")
        raise


# ─────────────────────────────────────────
# ENTRAÎNEMENT
# ─────────────────────────────────────────


@log_performance
def train_single_model(
    model_name: str,
    model_config: Dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    mlflow_run_name: str = None,
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Entraîne un modèle unique et log les résultats dans MLflow.

    Args:
        model_name: Nom du modèle
        model_config: Config {model: instance, params: dict}
        X_train, X_test: Features
        y_train, y_test: Labels
        preprocessor: ColumnTransformer déjà fit
        mlflow_run_name: Nom du run MLflow (défaut: model_name)

    Returns:
        Tuple[Pipeline, Dict]: (pipeline entraîné, métriques)
    """
    mlflow_run_name = mlflow_run_name or model_name

    logger.info(f"\n{'='*60}")
    logger.info(f"Entraînement: {model_name}")
    logger.info(f"{'='*60}")

    with mlflow.start_run(run_name=mlflow_run_name):
        # ── Versioning ──
        git_commit = get_git_revision_hash()
        data_version = get_dvc_data_version(str(config.data.raw_data_path))

        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("data_version", data_version)
        mlflow.set_tag("model_type", model_name)

        # ── Pipeline complet ──
        full_pipeline = Pipeline(
            [("preprocessor", preprocessor), ("classifier", model_config["model"])]
        )

        # ── Entraînement ──
        logger.info("Fit du pipeline en cours...")
        full_pipeline.fit(X_train, y_train)

        # ── Évaluation ──
        metrics = evaluate_model(full_pipeline, X_test, y_test)

        # ── Cross-validation ──
        logger.info("Cross-validation (5-fold) en cours...")
        cv_f1_scores = cross_val_score(
            full_pipeline, X_train, y_train, cv=5, scoring="f1", n_jobs=-1
        )
        cv_f1_mean = cv_f1_scores.mean()
        cv_f1_std = cv_f1_scores.std()

        # ── Log MLflow ──
        mlflow.log_params(model_config["params"])
        mlflow.log_metrics(metrics)
        mlflow.log_metric("cv_f1_mean", cv_f1_mean)
        mlflow.log_metric("cv_f1_std", cv_f1_std)

        # Log du modèle
        mlflow.sklearn.log_model(
            full_pipeline,
            artifact_path="model",
            registered_model_name=f"superstore_{model_name}",
        )

        # ── Affichage résultats ──
        logger.info(f"  F1-Score  : {metrics['f1_score']:.4f}")
        logger.info(f"  Precision : {metrics['precision']:.4f}")
        logger.info(f"  Recall    : {metrics['recall']:.4f}")
        logger.info(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
        logger.info(f"  Accuracy  : {metrics['accuracy']:.4f}")
        logger.info(f"  CV F1     : {cv_f1_mean:.4f} (±{cv_f1_std:.4f})")

        return full_pipeline, metrics


@log_execution
def train_all_models(
    input_path: str = None, output_path: str = None
) -> Tuple[str, float]:
    """
    Entraîne tous les modèles et retourne le meilleur.

    Pipeline:
    1. Charge et préprocess les données
    2. Split train/test stratifié
    3. Construit le preprocessor (fit sur train)
    4. Entraîne chaque modèle
    5. Évalue et compare
    6. Sauvegarde le preprocessor

    Args:
        input_path: Chemin données brutes (défaut: config)
        output_path: Chemin données traitées (défaut: config)

    Returns:
        Tuple[str, float]: (meilleur_modèle, meilleur_f1)
    """
    input_path = input_path or str(config.data.raw_data_path)
    output_path = output_path or str(config.data.processed_data_path)

    logger.info(f"Début de l'entraînement — Input: {input_path}")

    # ─ 1. DONNÉES ─
    logger.info("Chargement et preprocessing...")
    X, y = run_preprocessing(input_path, output_path)

    # ─ 2. SPLIT STRATIFIÉ ─
    logger.info(f"Split train/test (test_size={config.data.test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
        stratify=y,
    )
    logger.info(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    # ─ 3. PREPROCESSOR (fit sur train uniquement!) ─
    logger.info("Construction du preprocessor...")
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train, y_train)

    # ─ 4. MLFLOW ─
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    # ─ 5. ENTRAÎNEMENT ─
    models_config = get_models_config()
    best_f1 = 0
    best_model_name = None
    best_model = None

    for model_name, model_config in models_config.items():
        pipeline, metrics = train_single_model(
            model_name=model_name,
            model_config=model_config,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            preprocessor=preprocessor,
        )

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_model_name = model_name
            best_model = pipeline

    # ─ 6. SAUVEGARDE ─
    if config.model.save_preprocessor and best_model:
        logger.info("Sauvegarde du preprocessor...")
        save_preprocessor(preprocessor)

    logger.info(f"\n{'='*60}")
    logger.info("RÉSULTATS FINAUX")
    logger.info(f"{'='*60}")
    logger.info(f"Meilleur modèle : {best_model_name}")
    logger.info(f"F1-Score        : {best_f1:.4f}")
    logger.info(f"{'='*60}")

    return best_model_name, best_f1


# ─────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────

if __name__ == "__main__":
    best_name, best_f1 = train_all_models()
    print(f"\n✅ Meilleur modèle : {best_name} (F1 = {best_f1:.4f})")
