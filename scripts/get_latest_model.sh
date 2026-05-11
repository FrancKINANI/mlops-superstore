#!/usr/bin/env bash
# Script pour trouver le dernier modèle MLflow enregistré

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MLARTIFACTS_DIR="$REPO_ROOT/mlartifacts/1/models"

if [ ! -d "$MLARTIFACTS_DIR" ]; then
    echo "Erreur: $MLARTIFACTS_DIR n'existe pas"
    exit 1
fi

# Trouver le modèle le plus récemment modifié
LATEST_MODEL=$(find "$MLARTIFACTS_DIR" -name "model.pkl" -type f -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)

if [ -z "$LATEST_MODEL" ]; then
    echo "Erreur: Aucun modèle trouvé"
    exit 1
fi

echo "$LATEST_MODEL"
