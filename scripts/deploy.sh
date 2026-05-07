#!/usr/bin/env bash
# Script de déploiement : récupère le modèle Production et démarre le conteneur

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "🔍 Récupération du modèle en stage 'Production'..."
LATEST_MODEL=$(python3 "$REPO_ROOT/scripts/get_production_model.py" 2>&1 | tail -1)
# Convertir le chemin relatif en chemin absolu du conteneur
MODEL_PATH_CONTAINER="/app/$LATEST_MODEL"
echo "✅ Modèle trouvé: $LATEST_MODEL"
echo "📦 Chemin conteneur: $MODEL_PATH_CONTAINER"

echo "📝 Mise à jour du docker-compose.yml..."
MODEL_PATH_ESCAPED=$(printf '%s' "$MODEL_PATH_CONTAINER" | sed 's|/|\\/|g')
sed -i "s|MODEL_PATH=.*|MODEL_PATH=$MODEL_PATH_ESCAPED|" "$REPO_ROOT/docker-compose.yml"

echo "🐳 Reconstruction et démarrage du conteneur..."
docker compose down 2>/dev/null || true
docker compose up -d --build

echo "⏳ Attente du démarrage du service..."
sleep 5

echo "🏥 Test du health check..."
for i in {1..10}; do
  RESPONSE=$(curl -s http://localhost:8000/health)
  if echo "$RESPONSE" | grep -q "model_loaded.*true"; then
    echo "✅ Service prêt et modèle chargé!"
    echo "$RESPONSE" | jq .
    exit 0
  fi
  echo "  Tentative $i/10..."
  sleep 2
done

echo "❌ Le service n'a pas démarré correctement"
docker logs superstore-api | tail -20
exit 1
