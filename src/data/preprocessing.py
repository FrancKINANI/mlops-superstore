"""
Pipeline de preprocessing pour le dataset Superstore.
Transforme les données brutes en features prêtes pour la modélisation.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import config
from src.logging_utils import get_logger, log_performance

logger = get_logger(__name__)


# ─────────────────────────────────────────
# HELPER: Standardisation des noms de colonnes
# ─────────────────────────────────────────


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise les noms de colonnes en snake_case.
    Remplace les espaces par des underscores et convertit en minuscules.

    Args:
        df: DataFrame avec colonnes à standardiser

    Returns:
        pd.DataFrame: DataFrame avec colonnes standardisées

    Example:
        >>> df = pd.DataFrame({'Order Date': [1, 2], 'Ship Mode': [3, 4]})
        >>> df_std = standardize_column_names(df)
        >>> df_std.columns
        Index(['order_date', 'ship_mode'], dtype='object')
    """
    df = df.copy()
    df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace("-", "_")
    logger.debug(f"Noms de colonnes standardisés: {list(df.columns)}")
    return df


# ─────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────


@log_performance
def load_data(path: str) -> pd.DataFrame:
    """
    Charge le dataset brut depuis le chemin spécifié.

    Args:
        path: Chemin vers le fichier CSV

    Returns:
        pd.DataFrame: Dataset chargé

    Raises:
        FileNotFoundError: Si le fichier n'existe pas
        pd.errors.ParserError: Si le fichier n'est pas un CSV valide
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier non trouvé : {path}")

    df = pd.read_csv(path, encoding="latin-1")
    df = standardize_column_names(df)

    logger.info(f"Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    logger.debug(f"Colonnes: {list(df.columns)}")
    return df


# ─────────────────────────────────────────
# 2. CRÉATION DE LA VARIABLE CIBLE
# ─────────────────────────────────────────


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée la variable cible is_profitable.

    Classification binaire:
    - 1: transaction profitable (profit > 0)
    - 0: transaction à perte ou nulle (profit ≤ 0)

    Args:
        df: DataFrame contenant la colonne 'profit'

    Returns:
        pd.DataFrame: DataFrame avec colonne 'is_profitable' ajoutée

    Raises:
        KeyError: Si la colonne 'profit' n'existe pas
    """
    df = df.copy()

    if "profit" not in df.columns:
        raise KeyError("Colonne 'profit' manquante")

    df["is_profitable"] = (df["profit"] > 0).astype(int)
    rate = df["is_profitable"].mean() * 100

    logger.info(f"Variable cible créée — Taux de rentabilité : {rate:.1f}%")
    logger.debug(f"Distribution: {df['is_profitable'].value_counts().to_dict()}")

    return df


# ─────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée de nouvelles features à partir des colonnes existantes.

    Features créées:
    - Temporelles: order_month, order_quarter, order_dayofweek depuis Order Date
    - Délai de livraison: shipping_delay (jours entre Order Date et Ship Date)
    - Tarifaires: unit_price (Sales / Quantity)

    Args:
        df: DataFrame contenant 'order_date' et 'ship_date'

    Returns:
        pd.DataFrame: DataFrame avec nouvelles features

    Raises:
        KeyError: Si les colonnes de date manquent
    """
    df = df.copy()

    # Vérifier les colonnes requises
    required = ["order_date", "ship_date", "sales", "quantity"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Colonnes manquantes : {missing}")

    # Conversion des dates
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["ship_date"] = pd.to_datetime(df["ship_date"], errors="coerce")

    # Vérifier s'il y a des dates invalides
    invalid_dates = df["order_date"].isna().sum() + df["ship_date"].isna().sum()
    if invalid_dates > 0:
        logger.warning(f"{invalid_dates} dates invalides détectées")

    # Features temporelles
    df["order_month"] = df["order_date"].dt.month
    df["order_quarter"] = df["order_date"].dt.quarter
    df["order_dayofweek"] = df["order_date"].dt.dayofweek

    # Délai de livraison (jours)
    df["shipping_delay"] = (df["ship_date"] - df["order_date"]).dt.days

    # Unit price = Sales / Quantity (avec gestion division par 0)
    df["unit_price"] = np.where(df["quantity"] > 0, df["sales"] / df["quantity"], 0)

    logger.info("Feature engineering terminé — nouvelles features créées")
    logger.debug(
        f"Features temporelles créées: {['order_month', 'order_quarter', 'order_dayofweek']}"
    )

    return df


# ─────────────────────────────────────────
# 4. SÉLECTION ET NETTOYAGE
# ─────────────────────────────────────────


# Utiliser la config centralisée
def get_features_config() -> Tuple[list, list, list]:
    """
    Récupère la configuration des features depuis config.py.

    Returns:
        Tuple[list, list, list]: (cols_to_drop, cat_features, num_features)
    """
    return (
        config.preprocessing.cols_to_drop,
        config.preprocessing.cat_features,
        config.preprocessing.num_features,
    )


def select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Sépare X (features) et y (cible).
    Supprime les colonnes non pertinentes et gère les valeurs manquantes.

    Args:
        df: DataFrame contenant 'is_profitable' et autres colonnes

    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X, y) avec X features et y cible

    Raises:
        KeyError: Si 'is_profitable' n'existe pas
    """
    df = df.copy()

    if "is_profitable" not in df.columns:
        raise KeyError("Colonne 'is_profitable' manquante")

    cols_to_drop, _, _ = get_features_config()

    # Supprimer les colonnes inutiles
    cols_to_drop_existing = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop_existing, errors="ignore")

    # Séparer X et y
    y = df["is_profitable"]
    X = df.drop(columns=["is_profitable"])

    # Gestion des valeurs manquantes
    missing_count = X.isnull().sum().sum()
    if missing_count > 0:
        logger.warning(f"{missing_count} valeurs manquantes détectées")
        X = X.fillna(X.mean(numeric_only=True))
        logger.info("Valeurs manquantes numériques imputées avec la moyenne")

    logger.info(f"Features sélectionnées : {X.shape[1]} colonnes, {X.shape[0]} lignes")
    logger.debug(f"Features: {list(X.columns)}")

    return X, y


# ─────────────────────────────────────────
# 5. PIPELINE SKLEARN
# ─────────────────────────────────────────


def build_preprocessor() -> ColumnTransformer:
    """
    Construit le ColumnTransformer sklearn.

    Transformations:
    - OneHotEncoding sur les features catégorielles
    - StandardScaler sur les features numériques

    Returns:
        ColumnTransformer: Pipeline de preprocessing sklearn

    Note:
        Doit être fit sur les données d'entraînement uniquement!
    """
    _, cat_features, num_features = get_features_config()

    numeric_pipeline = Pipeline([("scaler", StandardScaler())])

    categorical_pipeline = Pipeline(
        [
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=None),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_features),
            ("cat", categorical_pipeline, cat_features),
        ],
        remainder="drop",
    )

    logger.debug(
        f"Preprocessor construit avec {len(cat_features)} features catégorielles et {len(num_features)} numériques"
    )
    return preprocessor


# ─────────────────────────────────────────
# 6. FONCTION PRINCIPALE
# ─────────────────────────────────────────


@log_performance
def run_preprocessing(
    input_path: str = None, output_path: str = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Exécute le pipeline complet de preprocessing.

    Pipeline:
    1. Chargement des données brutes
    2. Création de la variable cible (is_profitable)
    3. Feature engineering (features temporelles, unit_price)
    4. Sélection et nettoyage des features

    Args:
        input_path: Chemin vers les données brutes (défaut: config.data.raw_data_path)
        output_path: Chemin pour sauvegarder les données traitées (défaut: config.data.processed_data_path)

    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X, y) prêts pour la modélisation

    Raises:
        FileNotFoundError: Si le fichier d'entrée n'existe pas
        KeyError: Si colonnes requises manquent
    """
    input_path = input_path or str(config.data.raw_data_path)
    output_path = output_path or str(config.data.processed_data_path)

    logger.info(f"Début du preprocessing — Input: {input_path}")

    # Pipeline
    df = load_data(input_path)
    df = create_target(df)
    df = engineer_features(df)
    X, y = select_features(df)

    # Sauvegarde des données traitées
    processed = X.copy()
    processed["is_profitable"] = y.values

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)

    logger.info(f"Données traitées sauvegardées : {output_path}")
    logger.info(f"Résumé final — X: {X.shape}, y: {y.shape}")

    return X, y


if __name__ == "__main__":
    X, y = run_preprocessing()
    print(f"\nX shape : {X.shape}")
    print(f"X columns : {list(X.columns)}")
    print(f"y distribution :\n{y.value_counts()}")
