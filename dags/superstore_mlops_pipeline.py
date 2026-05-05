# dags/superstore_mlops_pipeline.py
"""
DAG MLOps — Pipeline complet Superstore
Compatible Airflow 3.x
Orchestration : Validation → Preprocessing → Training → Registry
"""

from datetime import datetime, timedelta
import sys
import os

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

PROJECT_ROOT = "/home/franck/Documents/01_Cours/Data/IA/Projets/efm/ML/mlops-superstore"

DEFAULT_ARGS = {
    "owner":            "mlops-team",
    "depends_on_past":  False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=2),
    "email_on_failure": False,
}

# Airflow 3.x : schedule remplace schedule_interval
dag = DAG(
    dag_id            = "superstore_mlops_pipeline",
    default_args      = DEFAULT_ARGS,
    description       = "Pipeline MLOps complet — Superstore profitability prediction",
    schedule          = "@weekly",
    start_date        = datetime(2026, 5, 1),
    catchup           = False,
    tags              = ["mlops", "superstore", "classification"],
)

# ─────────────────────────────────────────
# TÂCHE 0 — DÉPART
# ─────────────────────────────────────────

start = EmptyOperator(
    task_id = "start_pipeline",
    dag     = dag,
)

# ─────────────────────────────────────────
# TÂCHE 1 — VALIDATION DES DONNÉES
# ─────────────────────────────────────────

def validate_data(**context):
    import pandas as pd

    data_path = os.path.join(PROJECT_ROOT, "data/raw/superstore.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset introuvable : {data_path}")

    df = pd.read_csv(data_path, encoding='latin-1')

    assert df.shape[0] > 1000, \
        f"Dataset trop petit : {df.shape[0]} lignes"

    required_cols = ['Sales', 'Profit', 'Discount', 'Category', 'Region', 'Segment']
    missing = [c for c in required_cols if c not in df.columns]
    assert len(missing) == 0, f"Colonnes manquantes : {missing}"

    profitable_rate = (df['Profit'] > 0).mean()
    assert 0.5 < profitable_rate < 1.0, \
        f"Distribution cible anormale : {profitable_rate:.2%}"

    print(f"Validation réussie — {df.shape[0]} lignes")
    print(f"Taux de rentabilité : {profitable_rate:.1%}")

    # XCom — passage de données entre tâches
    context['ti'].xcom_push(key='row_count',       value=df.shape[0])
    context['ti'].xcom_push(key='profitable_rate', value=round(profitable_rate, 4))

task_validate = PythonOperator(
    task_id         = "validate_data",
    python_callable = validate_data,
    dag             = dag,
)

# ─────────────────────────────────────────
# TÂCHE 2 — PREPROCESSING
# ─────────────────────────────────────────

def run_preprocessing_task(**context):
    sys.path.insert(0, PROJECT_ROOT)
    from src.data.preprocessing import run_preprocessing

    input_path  = os.path.join(PROJECT_ROOT, "data/raw/superstore.csv")
    output_path = os.path.join(PROJECT_ROOT, "data/processed/superstore_processed.csv")

    X, y = run_preprocessing(input_path, output_path)

    print(f"Preprocessing terminé — X: {X.shape}, y: {y.shape}")
    print(f"Classe 0 (pertes)    : {(y==0).sum()} ({(y==0).mean():.1%})")
    print(f"Classe 1 (rentables) : {(y==1).sum()} ({(y==1).mean():.1%})")

    context['ti'].xcom_push(key='n_features', value=X.shape[1])
    context['ti'].xcom_push(key='n_samples',  value=X.shape[0])

task_preprocess = PythonOperator(
    task_id         = "preprocess_data",
    python_callable = run_preprocessing_task,
    dag             = dag,
)

# ─────────────────────────────────────────
# TÂCHE 3 — ENTRAÎNEMENT
# ─────────────────────────────────────────

def run_training_task(**context):
    sys.path.insert(0, PROJECT_ROOT)
    from src.models.train import train_all_models

    best_name, best_f1 = train_all_models()

    print(f"Meilleur modèle : {best_name} (F1={best_f1:.4f})")

    # Seuil qualité minimum — bloque le pipeline si insuffisant
    assert best_f1 >= 0.85, \
        f"F1 insuffisant : {best_f1:.4f} < 0.85. Pipeline interrompu."

    context['ti'].xcom_push(key='best_model_name', value=best_name)
    context['ti'].xcom_push(key='best_f1',         value=best_f1)

task_train = PythonOperator(
    task_id         = "train_models",
    python_callable = run_training_task,
    dag             = dag,
)

# ─────────────────────────────────────────
# TÂCHE 4 — MODEL REGISTRY
# ─────────────────────────────────────────

def run_registry_task(**context):
    sys.path.insert(0, PROJECT_ROOT)
    from src.models.register_model import promote_to_production

    best_name = context['ti'].xcom_pull(
        task_ids = 'train_models',
        key      = 'best_model_name'
    )
    best_f1 = context['ti'].xcom_pull(
        task_ids = 'train_models',
        key      = 'best_f1'
    )

    print(f"Promotion en Production : {best_name} (F1={best_f1:.4f})")
    promote_to_production()

task_register = PythonOperator(
    task_id         = "register_model",
    python_callable = run_registry_task,
    dag             = dag,
)

# ─────────────────────────────────────────
# TÂCHE 5 — NOTIFICATION FINALE
# ─────────────────────────────────────────

def notify_success(**context):
    ti = context['ti']

    row_count       = ti.xcom_pull(task_ids='validate_data',  key='row_count')
    profitable_rate = ti.xcom_pull(task_ids='validate_data',  key='profitable_rate')
    best_name       = ti.xcom_pull(task_ids='train_models',   key='best_model_name')
    best_f1         = ti.xcom_pull(task_ids='train_models',   key='best_f1')

    print("=" * 50)
    print("   PIPELINE MLOPS TERMINÉ AVEC SUCCÈS")
    print("=" * 50)
    print(f"  Données      : {row_count} lignes")
    print(f"  Rentabilité  : {profitable_rate:.1%}")
    print(f"  Meilleur     : {best_name}")
    print(f"  F1-Score     : {best_f1:.4f}")
    print(f"  Statut       : Production ✅")
    print("=" * 50)

task_notify = PythonOperator(
    task_id         = "notify_success",
    python_callable = notify_success,
    dag             = dag,
)

# ─────────────────────────────────────────
# DÉPENDANCES — Ordre d'exécution
# ─────────────────────────────────────────

start >> task_validate >> task_preprocess >> task_train >> task_register >> task_notify