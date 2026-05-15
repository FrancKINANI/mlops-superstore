#!/bin/bash
# run_project.sh — Orchestration du projet MLOps Superstore

# Couleurs
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

set -e

show_help() {
    echo -e "${BLUE}Usage: ./run_project.sh [phase]${NC}"
    echo ""
    echo "Phases disponibles :"
    echo -e "  ${GREEN}setup${NC}      : Installe l'environnement et les dépendances (uv)"
    echo -e "  ${GREEN}training${NC}   : Lance le pipeline complet (DVC + MLflow + Training)"
    echo -e "  ${GREEN}serving${NC}    : Lance l'API et l'UI Streamlit localement"
    echo -e "  ${GREEN}docker${NC}     : Lance la stack complète (API, UI, Monitoring) via Docker"
    echo -e "  ${GREEN}clean${NC}      : Nettoie les fichiers temporaires"
}

setup_env() {
    echo -e "${BLUE}📦 Installation de l'environnement...${NC}"
    make install
    make env
    echo -e "${GREEN}✓ Environnement prêt.${NC}"
}

run_training() {
    echo -e "${BLUE}🤖 Phase 1 : Entraînement et Gestion (MLflow + DVC)${NC}"
    
    # Vérifier si MLflow tourne
    if ! curl -s http://localhost:5000 > /dev/null; then
        echo -e "${YELLOW}⚠️  MLflow n'est pas lancé. Démarrage du serveur MLflow...${NC}"
        make train-mlflow &
        sleep 5
    fi
    
    echo -e "${BLUE}🚀 Exécution du pipeline DVC...${NC}"
    make pipeline  # Preprocess + Validate
    dvc repro
    
    echo -e "${GREEN}✅ Phase d'entraînement terminée. Meilleur modèle promu en Production.${NC}"
}

run_serving() {
    echo -e "${BLUE}🚀 Phase 2 : Inférence et Interface (Local)${NC}"
    echo -e "${YELLOW}Note: L'API doit être lancée pour que l'UI fonctionne.${NC}"
    
    echo -e "Démarrage de l'API en arrière-plan..."
    make api &
    API_PID=$!
    
    sleep 5
    
    echo -e "Démarrage de l'interface Streamlit..."
    make ui
    
    # Nettoyage à l'arrêt
    kill $API_PID
}

run_docker() {
    echo -e "${BLUE}🐳 Phase 2 : Inférence et Monitoring (Docker Compose)${NC}"
    make deploy
}

# Logique principale
case "$1" in
    setup)
        setup_env
        ;;
    training)
        run_training
        ;;
    serving)
        run_serving
        ;;
    docker)
        run_docker
        ;;
    clean)
        make clean
        ;;
    *)
        show_help
        ;;
esac
