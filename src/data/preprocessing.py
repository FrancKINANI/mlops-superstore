# src/data/preprocessing.py
"""
Pipeline de preprocessing pour le dataset Superstore.
Transforme les données brutes en features prêtes pour la modélisation.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Charge le dataset brut depuis le chemin spécifié."""
    df = pd.read_csv(path, encoding='latin-1')
    logger.info(f"Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


# ─────────────────────────────────────────
# 2. CRÉATION DE LA VARIABLE CIBLE
# ─────────────────────────────────────────

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée la variable cible is_profitable.
    1 = transaction profitable (Profit > 0)
    0 = transaction à perte ou nulle
    """
    df = df.copy()
    df['is_profitable'] = (df['Profit'] > 0).astype(int)
    rate = df['is_profitable'].mean() * 100
    logger.info(f"Variable cible créée — Taux de rentabilité : {rate:.1f}%")
    return df


# ─────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée de nouvelles features à partir des colonnes existantes.
    - Features temporelles depuis Order Date et Ship Date
    - Délai de livraison
    - Marge brute
    """
    df = df.copy()

    # Conversion des dates
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date']  = pd.to_datetime(df['Ship Date'])

    # Features temporelles
    df['order_month']    = df['Order Date'].dt.month
    df['order_quarter']  = df['Order Date'].dt.quarter
    df['order_dayofweek'] = df['Order Date'].dt.dayofweek

    # Délai de livraison (jours)
    df['shipping_delay'] = (df['Ship Date'] - df['Order Date']).dt.days

    # Marge brute = Sales / Quantity (prix unitaire moyen)
    df['unit_price'] = df['Sales'] / df['Quantity']

    logger.info("Feature engineering terminé — nouvelles features créées")
    return df


# ─────────────────────────────────────────
# 4. SÉLECTION ET NETTOYAGE
# ─────────────────────────────────────────

# Colonnes à supprimer (identifiants non prédictifs)
COLS_TO_DROP = [
    'Row ID', 'Order ID', 'Customer ID', 'Product ID',
    'Customer Name', 'Product Name', 'Order Date', 'Ship Date',
    'Profit', 'Country', 'City', 'State', 'Postal Code'
]

# Features catégorielles à encoder
CAT_FEATURES = ['Ship Mode', 'Segment', 'Region', 'Category', 'Sub-Category']

# Features numériques à scaler
NUM_FEATURES = [
    'Sales', 'Quantity', 'Discount',
    'order_month', 'order_quarter', 'order_dayofweek',
    'shipping_delay', 'unit_price'
]


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Sépare X (features) et y (cible).
    Supprime les colonnes non pertinentes.
    """
    df = df.drop(columns=COLS_TO_DROP, errors='ignore')
    y = df['is_profitable']
    X = df.drop(columns=['is_profitable'])
    logger.info(f"Features sélectionnées : {X.shape[1]} colonnes")
    return X, y


# ─────────────────────────────────────────
# 5. PIPELINE SKLEARN
# ─────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """
    Construit le ColumnTransformer sklearn :
    - OneHotEncoding sur les features catégorielles
    - StandardScaler sur les features numériques
    """
    numeric_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline,  NUM_FEATURES),
        ('cat', categorical_pipeline, CAT_FEATURES)
    ])

    return preprocessor


# ─────────────────────────────────────────
# 6. FONCTION PRINCIPALE
# ─────────────────────────────────────────

def run_preprocessing(input_path: str, output_path: str) -> tuple:
    """
    Exécute le pipeline complet de preprocessing.
    Retourne X, y prêts pour la modélisation.
    Sauvegarde les données traitées dans output_path.
    """
    df = load_data(input_path)
    df = create_target(df)
    df = engineer_features(df)
    X, y = select_features(df)

    # Sauvegarde des données traitées
    processed = X.copy()
    processed['is_profitable'] = y.values
    processed.to_csv(output_path, index=False)
    logger.info(f"Données traitées sauvegardées : {output_path}")

    return X, y


if __name__ == "__main__":
    X, y = run_preprocessing(
        input_path="data/raw/Superstore.csv",
        output_path="data/processed/Superstore_processed.csv"
    )
    print(f"\nX shape : {X.shape}")
    print(f"y distribution :\n{y.value_counts()}")