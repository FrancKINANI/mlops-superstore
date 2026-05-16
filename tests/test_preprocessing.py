# tests/test_preprocessing.py
"""
Tests unitaires — Pipeline MLOps Superstore
Couvre : Preprocessing, Feature Engineering, Détection de bugs ML
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.preprocessing import (
    load_data, create_target, engineer_features,
    select_features, build_preprocessor
)

# ─────────────────────────────────────────
# FIXTURE — Dataset partagé entre les tests
# ─────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df():
    """Charge le dataset une seule fois pour tous les tests."""
    return load_data("data/raw/Superstore.csv")

@pytest.fixture(scope="module")
def processed_df(raw_df):
    """Dataset après toutes les transformations."""
    df = create_target(raw_df)
    df = engineer_features(df)
    return df


# ─────────────────────────────────────────
# TESTS — CHARGEMENT
# ─────────────────────────────────────────

class TestLoading:

    def test_dataset_not_empty(self, raw_df):
        """Le dataset doit contenir des lignes."""
        assert len(raw_df) > 0, "Dataset vide"

    def test_expected_columns_present(self, raw_df):
        """Les colonnes critiques doivent être présentes."""
        required = ["Sales", "Profit", "Discount", "Category", "Region", "Segment"]
        missing = [c for c in required if c not in raw_df.columns]
        assert len(missing) == 0, f"Colonnes manquantes : {missing}"

    def test_no_fully_null_columns(self, raw_df):
        """Aucune colonne ne doit être entièrement vide."""
        fully_null = raw_df.columns[raw_df.isnull().all()].tolist()
        assert len(fully_null) == 0, f"Colonnes entièrement nulles : {fully_null}"


# ─────────────────────────────────────────
# TESTS — VARIABLE CIBLE
# ─────────────────────────────────────────

class TestTargetCreation:

    def test_target_column_created(self, processed_df):
        """La colonne is_profitable doit exister."""
        assert 'is_profitable' in processed_df.columns

    def test_target_is_binary(self, processed_df):
        """La cible doit être strictement binaire : 0 ou 1."""
        unique_values = set(processed_df['is_profitable'].unique())
        assert unique_values.issubset({0, 1}), \
            f"Valeurs inattendues dans la cible : {unique_values}"

    def test_target_distribution_reasonable(self, processed_df):
        """
        Le taux de classe positive doit être entre 50% et 95%.
        Un taux hors de cette plage signale un problème de définition.
        """
        rate = processed_df['is_profitable'].mean()
        assert 0.50 < rate < 0.95, \
            f"Distribution cible anormale : {rate:.2%}"

    def test_target_coherent_with_profit(self, processed_df):
        """
        is_profitable=1 doit correspondre exactement à Profit > 0.
        Vérifie la cohérence entre la cible et sa source.
        """
        expected = (processed_df["Profit"] > 0).astype(int)
        assert (processed_df["is_profitable"] == expected).all(), (
            "Incohérence entre is_profitable et Profit"
        )


# ─────────────────────────────────────────
# TESTS — FEATURE ENGINEERING
# ─────────────────────────────────────────

class TestFeatureEngineering:

    def test_temporal_features_created(self, processed_df):
        """Les features temporelles doivent être présentes."""
        expected = ['order_month', 'order_quarter', 'order_dayofweek', 'shipping_delay']
        missing  = [f for f in expected if f not in processed_df.columns]
        assert len(missing) == 0, f"Features temporelles manquantes : {missing}"

    def test_shipping_delay_non_negative(self, processed_df):
        """
        Le délai de livraison ne peut pas être négatif.
        Un délai négatif = erreur de saisie dans les données.
        """
        negative = (processed_df['shipping_delay'] < 0).sum()
        assert negative == 0, \
            f"{negative} lignes avec délai de livraison négatif"

    def test_order_month_valid_range(self, processed_df):
        """Les mois doivent être entre 1 et 12."""
        assert processed_df['order_month'].between(1, 12).all(), \
            "Valeurs de mois hors plage [1-12]"


# ─────────────────────────────────────────
# TESTS — DÉTECTION DE BUGS ML CLASSIQUES
# ─────────────────────────────────────────

class TestMLBugsDetection:

    def test_no_data_leakage_profit_in_features(self, processed_df):
        """
        BUG ML : Data Leakage.
        La colonne Profit ne doit PAS apparaître dans les features.
        Profit est la source directe de is_profitable — l'inclure
        revient à donner la réponse au modèle avant qu'il prédise.
        """
        X, y = select_features(processed_df)
        assert "Profit" not in X.columns, (
            "DATA LEAKAGE DÉTECTÉ : Profit présent dans les features"
        )

    def test_target_not_in_features(self, processed_df):
        """
        BUG ML : La variable cible ne doit pas être une feature.
        """
        X, y = select_features(processed_df)
        assert "is_profitable" not in X.columns, (
            "BUG : is_profitable présent dans les features"
        )

    def test_identifiers_removed(self, processed_df):
        """
        BUG ML : Les identifiants (Order ID, Customer ID...)
        ne doivent pas être des features — ils ne généralisent pas.
        """
        X, y = select_features(processed_df)
        leaking_ids = ["Order ID", "Customer ID", "Product ID", "Row ID"]
        found = [c for c in leaking_ids if c in X.columns]
        assert len(found) == 0, f"Identifiants présents dans les features : {found}"

    def test_stratified_split_preserves_distribution(self, processed_df):
        """
        BUG ML : Split non stratifié sur dataset déséquilibré.
        Vérifie que la proportion de classe 1 est similaire
        dans train et test après un split stratifié.
        """
        from sklearn.model_selection import train_test_split

        X, y = select_features(processed_df)

        _, _, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        train_rate = y_train.mean()
        test_rate  = y_test.mean()

        assert abs(train_rate - test_rate) < 0.02, \
            f"Split non stratifié : train={train_rate:.2%}, test={test_rate:.2%}"

    def test_correct_target_column(self, processed_df):
        """
        BUG ML : Mauvaise colonne cible.
        S'assurer que la cible est is_profitable et non Sales ou Profit.
        """
        X, y = select_features(processed_df)
        assert y.name == 'is_profitable', \
            f"Mauvaise colonne cible : {y.name}"
        assert set(y.unique()).issubset({0, 1}), \
            "La cible n'est pas binaire — mauvaise colonne sélectionnée"


# ─────────────────────────────────────────
# TESTS — PREPROCESSOR SKLEARN
# ─────────────────────────────────────────

class TestPreprocessor:

    def test_preprocessor_builds_without_error(self):
        """Le ColumnTransformer doit se construire sans erreur."""
        preprocessor = build_preprocessor()
        assert preprocessor is not None

    def test_preprocessor_fits_and_transforms(self, processed_df):
        """Le preprocessor doit transformer X sans erreur."""
        from sklearn.model_selection import train_test_split

        X, y = select_features(processed_df)
        X_train, X_test, _, _ = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        preprocessor = build_preprocessor()
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed  = preprocessor.transform(X_test)

        # Dimensions cohérentes
        assert X_train_transformed.shape[0] == len(X_train)
        assert X_test_transformed.shape[0]  == len(X_test)

        # Pas de NaN après transformation
        assert not np.isnan(X_train_transformed).any(), \
            "NaN détectés après preprocessing sur train"
        assert not np.isnan(X_test_transformed).any(), \
            "NaN détectés après preprocessing sur test"