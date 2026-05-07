#!/usr/bin/env python3
"""
Récupère le modèle en stage "Production" depuis MLflow
Fallback sur le dernier modèle local si MLflow n'est pas accessible
"""

import os
import sys
import glob
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "superstore_GradientBoosting")
MLARTIFACTS_DIR = REPO_ROOT / "mlartifacts" / "1" / "models"


def is_mlflow_accessible():
    """Vérifie si MLflow est accessible"""
    try:
        import urllib.request
        urllib.request.urlopen(MLFLOW_TRACKING_URI, timeout=3)
        return True
    except Exception:
        return False


def get_production_model_from_mlflow():
    """Récupère le modèle Production depuis MLflow"""
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        
        if versions:
            print(f"✅ Modèle Production trouvé: version {versions[0].version}", file=sys.stderr)
            return True
        else:
            print(f"⚠️  Aucun modèle en stage 'Production' pour {MODEL_NAME}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"⚠️  Erreur MLflow: {e}", file=sys.stderr)
        return False


def get_latest_local_model():
    """Retourne le chemin du dernier modèle entraîné"""
    if not MLARTIFACTS_DIR.exists():
        print(f"❌ Erreur: {MLARTIFACTS_DIR} n'existe pas", file=sys.stderr)
        sys.exit(1)
    
    files = sorted(
        glob.glob(str(MLARTIFACTS_DIR / "*" / "artifacts" / "model.pkl")),
        key=lambda f: os.path.getmtime(f),
        reverse=True
    )
    
    if not files:
        print(f"❌ Aucun modèle trouvé dans {MLARTIFACTS_DIR}", file=sys.stderr)
        sys.exit(1)
    
    return files[0]


def main():
    print(f"📡 Récupération du modèle '{MODEL_NAME}' en stage 'Production'...", file=sys.stderr)
    print(f"   MLflow URI: {MLFLOW_TRACKING_URI}", file=sys.stderr)
    
    # Tenter de récupérer depuis MLflow
    if is_mlflow_accessible():
        print("   ✅ MLflow accessible", file=sys.stderr)
        if get_production_model_from_mlflow():
            model_path = get_latest_local_model()
            print(f"   📦 Chemin local: {model_path}", file=sys.stderr)
            print(model_path)
            return 0
    
    # Fallback: dernier modèle local
    print("   ⚠️  Fallback sur le dernier modèle local", file=sys.stderr)
    model_path = get_latest_local_model()
    print(f"   📦 Chemin local: {model_path}", file=sys.stderr)
    print(model_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
