#!/bin/bash
# Script de démarrage rapide — MLOps Superstore

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     🚀 DÉMARRAGE RAPIDE — MLOPS SUPERSTORE                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_ROOT"

# Couleurs
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ─────────────────────────────────────────
# 1. SETUP
# ─────────────────────────────────────────

echo -e "${BLUE}1️⃣  Configuration de l'environnement...${NC}"

if [ ! -d ".venv" ]; then
    echo "   Création du venv..."
    python -m venv .venv
fi

source .venv/bin/activate
echo "   ✓ Venv activé"

# Installer les dépendances
echo "   Installation des dépendances (cela peut prendre quelques minutes)..."
pip install -q -r requirements.txt 2>/dev/null || {
    echo -e "   ${YELLOW}⚠️  Certaines dépendances peuvent ne pas être installées${NC}"
    echo "   Cela peut être normal si mlflow n'est pas dans le venv"
}
echo -e "   ${GREEN}✓ Dépendances OK${NC}"

# Configuration .env
echo ""
if [ ! -f ".env" ]; then
    echo -e "${BLUE}2️⃣  Configuration de .env...${NC}"
    cp .env.example .env
    echo -e "   ${GREEN}✓ .env créé (personnalisez-le si nécessaire)${NC}"
else
    echo -e "${BLUE}2️⃣  .env existe déjà${NC}"
fi

# Charger les variables d'env
export $(cat .env | grep -v '^#' | xargs)

# ─────────────────────────────────────────
# 2. VALIDATION
# ─────────────────────────────────────────

echo ""
echo -e "${BLUE}3️⃣  Validation du projet...${NC}"
python validate_improvements.py | tail -20
echo ""

# ─────────────────────────────────────────
# 3. DONNÉES
# ─────────────────────────────────────────

echo -e "${BLUE}4️⃣  Pipeline de données...${NC}"

if [ ! -f "data/processed/Superstore_processed.csv" ]; then
    echo "   Preprocessing des données..."
    python -m src.data.preprocessing > /dev/null 2>&1
    echo -e "   ${GREEN}✓ Données prétraitées${NC}"
else
    echo -e "   ${GREEN}✓ Données déjà prétraitées${NC}"
fi

# ─────────────────────────────────────────
# 4. NEXT STEPS
# ─────────────────────────────────────────

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ CONFIGURATION COMPLÈTE!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo ""

echo "📝 Prochaines étapes:"
echo ""
echo "1️⃣  Pour l'entraînement (dans un autre terminal):"
echo -e "   ${YELLOW}mlflow server --host 127.0.0.1 --port 5000${NC}"
echo "   puis dans ce terminal:"
echo -e "   ${YELLOW}make train${NC}"
echo ""
echo "2️⃣  Pour l'API:"
echo -e "   ${YELLOW}make api${NC}"
echo "   Documentation: http://localhost:8000/docs"
echo ""
echo "3️⃣  Pour les tests:"
echo -e "   ${YELLOW}make test${NC}"
echo ""
echo "4️⃣  Pour Docker:"
echo -e "   ${YELLOW}make deploy${NC}"
echo ""
echo "📚 Pour plus d'infos:"
echo -e "   ${YELLOW}make help${NC}"
echo "   ou consulter ${BLUE}IMPROVEMENTS.md${NC}"
echo ""
