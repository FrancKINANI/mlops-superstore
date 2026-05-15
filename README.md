# 💰 MLOps Superstore : Prédiction de Rentabilité

Ce projet implémente un pipeline MLOps de bout en bout pour prédire la rentabilité des transactions d'une chaîne de magasins (dataset "Superstore"). L'objectif est de transformer un processus d'analyse de données classique en un système industriel automatisé, robuste et monitoré.

---

## 📖 Sommaire
1. [Contexte et Objectifs](#-contexte-et-objectifs)
2. [Architecture du Système](#-architecture-du-système)
3. [Cycle de Vie du Modèle (Training Pipeline)](#-cycle-de-vie-du-modèle-training-pipeline)
4. [Service et Inférence (Serving)](#-service-et-inférence-serving)
5. [Observabilité et Monitoring](#-observabilité-et-monitoring)
6. [Guide de Démarrage Rapide](#-guide-de-démarrage-rapide)

---

## 🎯 Contexte et Objectifs

Dans le secteur du retail, comprendre pourquoi certaines transactions sont déficitaires est crucial. Ce projet classifie les transactions en deux catégories :
- **Rentable (1)** : Profit > 0
- **Non Rentable (0)** : Profit ≤ 0

### Valeur Métier
- **Optimisation des prix** : Identifier l'impact des remises (discounts) sur la rentabilité.
- **Aide à la décision** : Aider les gestionnaires à ajuster les modes d'expédition ou les segments clients ciblés.
- **Automatisation** : Passer d'une analyse ponctuelle à un service prédictif disponible 24/7.

---

## 🏗 Architecture du Système

Le projet repose sur une architecture modulaire respectant les principes du MLOps :

```text
[ Données Brutes ] ──> ( DVC ) ──> [ Preprocessing ] ──> ( MLflow ) ──> [ Modèle Production ]
                                                                             │
       ┌─────────────────────────────────────────────────────────────────────┘
       ▼
[ API FastAPI ] <───> [ UI Streamlit ]
       │
       └─> ( Prometheus ) ──> [ Dashboards Grafana ]
```

---

## ⚙️ Cycle de Vie du Modèle (Training Pipeline)

L'automatisation est assurée par une combinaison de **DVC**, **Airflow** et **MLflow**.

### 1. Gestion des Données (DVC)
Le dataset `Superstore.csv` est volumineux et ne doit pas être stocké sur Git. DVC (Data Version Control) assure le versionnage des données, permettant de lier chaque version du modèle à la version exacte des données utilisée.

### 2. Pipeline d'Entraînement
Le pipeline est défini dans `dvc.yaml` et peut être orchestré par Airflow :
- **Preprocessing** : Nettoyage, encodage des variables catégorielles et feature engineering (ex: calcul du délai de livraison, extraction du mois/trimestre).
- **Entraînement** : Test de plusieurs algorithmes (Random Forest, Gradient Boosting).
- **Tracking (MLflow)** : Chaque run enregistre les hyperparamètres, le F1-Score et la courbe ROC-AUC.

### 3. Registre de Modèles
Le meilleur modèle (actuellement **Gradient Boosting**) est promu au stade **"Production"** dans MLflow. L'API d'inférence charge automatiquement ce modèle sans intervention manuelle.

---

## 🚀 Service et Inférence (Serving)

Le modèle est exposé via deux interfaces complémentaires :

### 1. API Backend (FastAPI)
- **Endpoint `/predict`** : Reçoit les données JSON et retourne la prédiction avec un score de confiance.
- **Performance** : Utilise un système de cache pour le modèle afin de garantir des temps de réponse ultra-courts.
- **Documentation** : Swagger UI intégrée pour faciliter l'intégration par d'autres développeurs.

### 2. Frontend (Streamlit)
Une interface utilisateur intuitive permettant aux analystes métier de :
- Saisir manuellement les détails d'une transaction.
- Visualiser instantanément la probabilité de rentabilité.
- Faire des tests de type "What-if" (ex: "Que se passe-t-il si j'augmente la remise de 10% ?").

---

## 📊 Observabilité et Monitoring

Déployer un modèle ne suffit pas ; il faut s'assurer qu'il reste performant dans le temps.

- **Prometheus** : Collecte en temps réel les métriques de l'API (nombre de requêtes, taux d'erreur 5xx, latence P95).
- **Grafana** : Affiche des tableaux de bord pour visualiser la santé du système.
- **Détection de Dérive (Drift)** : Des scripts (via `evidently`) sont prévus pour comparer les données entrantes avec les données d'entraînement et alerter en cas de dégradation de la performance.

---

## 🚀 Guide de Démarrage Rapide

### Prérequis
- Python 3.12+
- Docker & Docker Compose (pour la stack complète)
- `uv` (recommandé pour la gestion des paquets)

### Installation et Lancement
Le script `./run_project.sh` est votre point d'entrée unique :

1.  **Setup** : `./run_project.sh setup`
2.  **Entraîner** : `./run_project.sh training`
3.  **Tester localement** : `./run_project.sh serving`
4.  **Déployer la stack complète** : `./run_project.sh docker`

### Accès aux Services
- **Interface Streamlit** : `http://localhost:8501`
- **API Swagger** : `http://localhost:8000/docs`
- **Grafana** : `http://localhost:3000` (admin/admin)
- **MLflow UI** : `http://localhost:5000`

---

## 🛠 Tech Stack
- **Langage** : Python
- **Gestion de paquets** : `uv`
- **ML** : Scikit-learn, Pandas, NumPy
- **MLOps** : DVC, MLflow, Airflow
- **Serving** : FastAPI, Streamlit, Uvicorn
- **DevOps/Monitoring** : Docker, Prometheus, Grafana

---

## 📝 License
Projet réalisé dans le cadre du cursus MLOps. Pour usage éducatif uniquement.
