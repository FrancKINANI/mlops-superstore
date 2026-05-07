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
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.bash import BashOperator

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

    data_path = os.path.join(PROJECT_ROOT, "data/raw/Superstore.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset introuvable : {data_path}")

    df = pd.read_csv(data_path, encoding='latin-1')

    if df.shape[0] <= 1000:
        raise ValueError(f"Dataset trop petit : {df.shape[0]} lignes")

    required_cols = ['Sales', 'Profit', 'Discount', 'Category', 'Region', 'Segment']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    profitable_rate = (df['Profit'] > 0).mean()
    if not (0.5 < profitable_rate < 1.0):
        raise ValueError(
            f"Distribution cible anormale : {profitable_rate:.2%}"
        )

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
# TÂCHE 2 — Great Expectations
# ─────────────────────────────────────────


def run_great_expectations(**context):
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from src.data.data_validation import validate_raw_data

    result = validate_raw_data(
        os.path.join(PROJECT_ROOT, "data/raw/Superstore.csv")
    )

    if not result['success']:
        raise RuntimeError(
            f"Great Expectations échoué : {result['failed']} règles non respectées"
        )

    context['ti'].xcom_push(key='ge_passed', value=result['passed'])
    context['ti'].xcom_push(key='ge_total',  value=result['total'])
    print(f"GE Validation : {result['passed']}/{result['total']} ✅")

task_ge = PythonOperator(
    task_id         = "great_expectations_validation",
    python_callable = run_great_expectations,
    dag             = dag,
)



# ─────────────────────────────────────────
# TÂCHE 3 — DVC REPRO
# ─────────────────────────────────────────

task_dvc_repro = BashOperator(
    task_id = "dvc_repro",
    bash_command = f"cd {PROJECT_ROOT} && venv/bin/dvc repro",
    env = {
        **os.environ,
        "MLFLOW_TRACKING_URI": "http://localhost:5000",
    },
    dag = dag,
)

# ─────────────────────────────────────────
# TÂCHE 4 — NOTIFICATION FINALE
# ─────────────────────────────────────────

def notify_success(**context):
    print("=" * 50)
    print("   PIPELINE DVC / MLOPS TERMINÉ")
    print("=" * 50)
    print("  Consulter MLflow pour les métriques et artefacts.")
    print("  DVC a synchronisé les données et modèles.")
    print("=" * 50)

task_notify = PythonOperator(
    task_id         = "notify_success",
    python_callable = notify_success,
    dag             = dag,
)

# ─────────────────────────────────────────
# DÉPENDANCES — Ordre d'exécution
# ─────────────────────────────────────────

start >> task_validate >> task_ge >>  task_dvc_repro >> task_notify