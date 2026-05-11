# Makefile — MLOps Superstore Pipeline
# Commandes utiles pour développement et déploiement

.PHONY: help install env preprocess validate train test api deploy clean

# Variables
PYTHON := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest

# ─────────────────────────────────────────
# AIDE
# ─────────────────────────────────────────

help:
	@echo "🚀 MLOps Superstore — Commandes disponibles"
	@echo ""
	@echo "Setup:"
	@echo "  make install         Installe les dépendances"
	@echo "  make env             Configure les variables d'env"
	@echo ""
	@echo "Pipeline:"
	@echo "  make preprocess      Préprocess les données"
	@echo "  make validate        Valide les données"
	@echo "  make train           Entraîne les modèles"
	@echo "  make test            Lance les tests"
	@echo ""
	@echo "Déploiement:"
	@echo "  make api             Lance l'API"
	@echo "  make deploy          Déploie avec Docker"
	@echo ""
	@echo "Utilitaires:"
	@echo "  make clean           Nettoie les fichiers temporaires"
	@echo "  make logs            Affiche les logs"

# ─────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────

install:
	@echo "📦 Installation des dépendances..."
	@python -m venv $(VENV)
	@$(PIP) install --upgrade pip setuptools wheel
	@$(PIP) install -r requirements.txt
	@echo "✅ Installation complète!"

env:
	@echo "⚙️  Configuration de l'environnement..."
	@if [ ! -f .env ]; then cp .env.example .env && echo "✅ Fichier .env créé (à personnaliser)"; else echo "ℹ️  .env existe déjà"; fi

# ─────────────────────────────────────────
# PIPELINE DE DONNÉES
# ─────────────────────────────────────────

preprocess:
	@echo "🔄 Preprocessing des données..."
	@$(PYTHON) -m src.data.preprocessing
	@echo "✅ Preprocessing terminé!"

validate:
	@echo "✔️  Validation des données..."
	@$(PYTHON) -c "from src.data.data_validation import get_validation_status; status = get_validation_status(); print('✅ Validation réussie!' if status['overall_success'] else '❌ Validation échouée!')"
	@echo ""

pipeline: preprocess validate
	@echo "✅ Pipeline de données complété!"

# ─────────────────────────────────────────
# ENTRAÎNEMENT
# ─────────────────────────────────────────

train:
	@echo "🤖 Entraînement des modèles..."
	@echo "Assurez-vous que MLflow est lancé (mlflow server --host 127.0.0.1 --port 5000)"
	@$(PYTHON) -m src.models.train
	@echo "✅ Entraînement terminé!"

train-mlflow:
	@echo "🎯 Démarrage de MLflow..."
	@mlflow server --host 127.0.0.1 --port 5000 &
	@sleep 2
	@echo "✅ MLflow lancé sur http://localhost:5000"

# ─────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────

test:
	@echo "🧪 Exécution des tests..."
	@$(PYTEST) tests/ -v --tb=short
	@echo "✅ Tests terminés!"

test-api:
	@echo "🧪 Tests API uniquement..."
	@$(PYTEST) tests/test_api.py -v --tb=short

test-coverage:
	@echo "📊 Coverage des tests..."
	@$(PYTEST) tests/ --cov=src --cov-report=html --cov-report=term
	@echo "✅ Rapport HTML généré dans htmlcov/index.html"

# ─────────────────────────────────────────
# API
# ─────────────────────────────────────────

api:
	@echo "🚀 Démarrage de l'API (http://localhost:8000)..."
	@echo "Documentation: http://localhost:8000/docs"
	@$(PYTHON) -m src.api.main

docker-build:
	@echo "🐳 Construction de l'image Docker..."
	@docker build -t superstore-api:latest .
	@echo "✅ Image construite!"

docker-run:
	@echo "🐳 Exécution du container..."
	@docker run -p 8000:8000 --env-file .env superstore-api:latest
	@echo "API disponible sur http://localhost:8000"

deploy: docker-build docker-run
	@echo "✅ Déploiement complété!"

# ─────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────

clean:
	@echo "🧹 Nettoyage..."
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@rm -rf .pytest_cache .coverage htmlcov .mypy_cache .tox
	@echo "✅ Nettoyage terminé!"

logs:
	@echo "📋 Logs récents..."
	@tail -100 logs/*.log 2>/dev/null || echo "Aucun log trouvé"

check:
	@echo "🔍 Vérification du projet..."
	@$(PYTHON) -c "from src.config import config; print('✅ Config OK'); print(config)"
	@$(PYTHON) -c "from src.data.preprocessing import run_preprocessing; print('✅ Preprocessing importable')"
	@$(PYTHON) -c "from src.models.train import train_all_models; print('✅ Training importable')"
	@$(PYTHON) -c "from src.api.main import app; print('✅ API importable')"

# ─────────────────────────────────────────
# COMPLET
# ─────────────────────────────────────────

all: install env preprocess validate
	@echo "✅ Setup complet! Lancez 'make train' et 'make api' pour continuer."

.DEFAULT_GOAL := help
