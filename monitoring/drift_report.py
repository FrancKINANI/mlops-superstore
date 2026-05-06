# monitoring/drift_report.py
"""
Détection de Data Drift — Evidently AI
Compare la distribution référence (train) vs production (simulée)
"""

import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evidently import Dataset, DataDefinition
from evidently.presets import DataDriftPreset
from evidently import Report

from data.preprocessing import (
    load_data, create_target, engineer_features,
    select_features, NUM_FEATURES, CAT_FEATURES
)

# ─────────────────────────────────────────
# 1. DONNÉES DE RÉFÉRENCE (train)
# ─────────────────────────────────────────

def prepare_reference_data() -> pd.DataFrame:
    from sklearn.model_selection import train_test_split

    df = load_data("data/raw/Superstore.csv")
    df = create_target(df)
    df = engineer_features(df)
    X, y = select_features(df)

    X_train, _, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train


# ─────────────────────────────────────────
# 2. DONNÉES DE PRODUCTION (simulées)
# Simule une dégradation réelle :
# - remises anormalement élevées
# - concentration sur Furniture (peu rentable)
# ─────────────────────────────────────────

def simulate_production_drift(reference: pd.DataFrame) -> pd.DataFrame:
    prod = reference.copy()
    n    = len(prod)

    # Drift 1 : remises anormalement élevées en production
    prod['Discount'] = np.clip(
        prod['Discount'] + np.random.normal(0.15, 0.05, n), 0, 1
    )

    # Drift 2 : délai de livraison augmenté (problème logistique simulé)
    prod['shipping_delay'] = prod['shipping_delay'] + np.random.randint(2, 8, n)

    # Drift 3 : prix unitaires plus bas (promotions agressives)
    prod['unit_price'] = prod['unit_price'] * np.random.uniform(0.7, 0.9, n)

    return prod


# ─────────────────────────────────────────
# 3. GÉNÉRATION DU RAPPORT
# ─────────────────────────────────────────

def generate_drift_report():
    print("Préparation des données...")
    reference  = prepare_reference_data()
    production = simulate_production_drift(reference)

    print(f"Référence  : {reference.shape[0]} lignes")
    print(f"Production : {production.shape[0]} lignes")

    # Définition des colonnes pour Evidently
    definition = DataDefinition(
        numerical_columns   = NUM_FEATURES,
        categorical_columns = CAT_FEATURES,
    )

    # Datasets Evidently
    ref_dataset  = Dataset.from_pandas(reference,  data_definition=definition)
    prod_dataset = Dataset.from_pandas(production, data_definition=definition)

    # Rapport de drift
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=ref_dataset, current_data=prod_dataset)

    # Sauvegarde HTML
    os.makedirs("monitoring/reports", exist_ok=True)
    output_path = "monitoring/reports/drift_report.html"
    snapshot.save_html(output_path)
    print(f"\nRapport généré : {output_path}")
    print("Ouvre ce fichier dans ton navigateur pour la capture d'écran.")

    return output_path


if __name__ == "__main__":
    generate_drift_report()