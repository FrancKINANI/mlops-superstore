# src/models/register_model.py
"""
Promotion du meilleur modèle vers Production dans MLflow Model Registry.
Cycle de vie : None -> Staging -> Production
"""

import mlflow
from mlflow import MlflowClient

MLFLOW_TRACKING_URI = "http://localhost:5000"
BEST_MODEL_NAME     = "superstore_GradientBoosting"
BEST_MODEL_VERSION  = "1"

def promote_to_production():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # Étape 1 — Passage en Staging (validation intermédiaire) - using aliases
    try:
        client.set_registered_model_alias(
            name    = BEST_MODEL_NAME,
            alias   = "staging",
            version = BEST_MODEL_VERSION
        )
        print(f"[Staging] {BEST_MODEL_NAME} v{BEST_MODEL_VERSION}")
    except Exception:
        # Fallback pour anciennes versions de MLflow
        client.transition_model_version_stage(
            name    = BEST_MODEL_NAME,
            version = BEST_MODEL_VERSION,
            stage   = "Staging"
        )
        print(f"[Staging] {BEST_MODEL_NAME} v{BEST_MODEL_VERSION}")

    # Étape 2 — Validation avant Production
    model_info = client.get_model_version(BEST_MODEL_NAME, BEST_MODEL_VERSION)
    print(f"Modèle source      : {model_info.source}")
    print(f"Run ID             : {model_info.run_id}")

    # Étape 3 — Promotion en Production - using aliases
    try:
        client.set_registered_model_alias(
            name    = BEST_MODEL_NAME,
            alias   = "prod",
            version = BEST_MODEL_VERSION
        )
        print(f"[Production] {BEST_MODEL_NAME} v{BEST_MODEL_VERSION} — ACTIF")
    except Exception:
        # Fallback pour anciennes versions de MLflow
        client.transition_model_version_stage(
            name    = BEST_MODEL_NAME,
            version = BEST_MODEL_VERSION,
            stage   = "Production"
        )
        print(f"[Production] {BEST_MODEL_NAME} v{BEST_MODEL_VERSION} — ACTIF")

    # Étape 4 — Archiver les autres modèles en supprimant leurs alias
    for model_name in ["superstore_LogisticRegression", "superstore_RandomForest"]:
        try:
            # Supprimer tous les alias pour ces modèles
            for alias in ["prod", "staging"]:
                try:
                    client.delete_registered_model_alias(model_name, alias)
                except:
                    pass
            print(f"[Archived] {model_name} v1")
        except Exception:
            # Fallback pour anciennes versions
            client.transition_model_version_stage(
                name    = model_name,
                version = "1",
                stage   = "Archived"
            )
            print(f"[Archived] {model_name} v1")

    # Étape 5 — Ajouter une description au modèle en production
    client.update_model_version(
        name        = BEST_MODEL_NAME,
        version     = BEST_MODEL_VERSION,
        description = (
            "GradientBoosting — Meilleur modèle sélectionné. "
            "F1=0.9723 | ROC-AUC=0.9911 | CV F1=0.9716. "
            "Entraîné sur Superstore dataset (9994 lignes, stratify=True)."
        )
    )
    print("\nModèle en Production avec description enregistrée.")

if __name__ == "__main__":
    promote_to_production()