import mlflow
import os

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "superstore_GradientBoosting"
MODEL_STAGE = "Production"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

try:
    print("Tentative de chargement LOCAL du modèle...")
    # On cherche dynamiquement un dossier contenant model.pkl
    import glob
    model_paths = glob.glob("mlartifacts/**/model.pkl", recursive=True)
    if model_paths:
        local_uri = os.path.dirname(model_paths[0])
        print(f"Chargement depuis : {local_uri}")
        model = mlflow.sklearn.load_model(local_uri)
        print("Modèle chargé avec succès LOCALEMENT !")
    else:
        print("Aucun model.pkl trouvé localement.")
except Exception as e:
    print(f"Erreur lors du chargement local : {e}")

try:
    print(f"\nTentative de chargement via REGISTRY : {MODEL_NAME}/{MODEL_STAGE}...")
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.sklearn.load_model(model_uri)
    print("Modèle chargé avec succès via Registry !")
except Exception as e:
    print(f"Erreur lors du chargement via Registry : {e}")
