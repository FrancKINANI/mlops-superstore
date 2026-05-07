## Déploiement automatique avec Docker

Ce projet utilise **Docker Compose** pour orchestrer trois services clés :
1. **superstore-api** : L'API FastAPI d'inférence.
2. **prometheus** : Collecteur de métriques de performance.
3. **grafana** : Visualisation des métriques en temps réel.

## Processus de déploiement

**Après avoir marqué un modèle comme "Production" dans MLflow:**

```bash
bash scripts/deploy.sh
# or
make deploy
```

Ce script :
1. **Récupère** le modèle en stage **"Production"** depuis MLflow (ou fallback sur le dernier modèle local)
2. **Met à jour** le `docker-compose.yml` avec le chemin du modèle
3. **Reconstruit** l'image Docker avec le nouveau modèle
4. **Démarre** le conteneur
5. **Vérifie** que le service est prêt

## Fichiers clés

- `scripts/deploy.sh` - Script de déploiement automatique
- `scripts/get_production_model.py` - Récupère le modèle en stage "Production" depuis MLflow
- `scripts/get_latest_model.sh` - Dernier modèle local disponible
- `docker-compose.yml` - Configuration du conteneur (MODEL_PATH mise à jour auto)
- `src/api/main.py` - API FastAPI qui charge le modèle via MODEL_PATH

## Processus MLflow → Production

1. **Entraîner un modèle** (dans MLflow)
2. **Marquer le modèle comme "Production"** dans MLflow UI ou via CLI:
   ```bash
   mlflow models transition-model-version-stage \
     --name "superstore_GradientBoosting" \
     --version 5 \
     --stage "Production"
   ```
3. **Déployer** :
   ```bash
   bash deploy.sh
   ```

## Vérification du déploiement

```bash
# Vérifier que le service est up et le modèle chargé
curl http://localhost:8000/health | jq .

# Tester une prédiction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Sales": 800,
    "Quantity": 2,
    "Discount": 0.0,
    "Ship_Mode": "Standard Class",
    "Segment": "Corporate",
    "Region": "West",
    "Category": "Technology",
    "Sub_Category": "Phones",
    "order_month": 11,
    "order_quarter": 4,
    "order_dayofweek": 1,
    "shipping_delay": 4,
    "unit_price": 400.0
  }'

# Voir les logs
docker logs superstore-api

# Arrêter le service
docker compose down
```

## Architecture

```
MLflow Model Registry
├── superstore_GradientBoosting (v1) → Staging
├── superstore_GradientBoosting (v2) → None
└── superstore_GradientBoosting (v3) → Production ← Récupéré par deploy.sh

                    ↓

mlartifacts/1/models/
└── m-f612a2259fc642ceb2d7071cb9334223/ (MODÈLE PRODUCTION)
    └── artifacts/model.pkl ← Embarqué dans l'image Docker

                    ↓

Docker Image (superstore-api:v1)
└── /app/mlartifacts/.../model.pkl ← Chargé par l'API
```

## Comportement du script

### Cas 1 : MLflow accessible et modèle Production existe
1. Vérifie la connexion à MLflow
2. Récupère le modèle en stage "Production"
3. Utilise le dernier modèle local correspondant
4. ✅ Déploie le modèle Production

### Cas 2 : MLflow inaccessible
1. Essai de connexion échoue
2. Fallback sur le **dernier modèle entraîné** (most recent by timestamp)
3. ✅ Déploie le dernier modèle disponible

### Cas 3 : Erreur
- Si aucun modèle n'existe → Erreur + exit(1)

## Notes

- **Modèle embarqué** : Le modèle est inclus dans l'image Docker (~11MB)
- **Image totale** : ~1.5GB (avec dépendances)
- **Chargement lazy** : Le modèle charge à la première requête `/predict`
- **Pas de dépendance runtime** : MLflow n'est pas requis pour exécuter l'API
- **Version Python** : 3.12 slim

## Troubleshooting

**Modèle ne charge pas :**
```bash
docker logs superstore-api | grep -E "ERROR|❌|Erreur"
```

**Port 8000 déjà utilisé :**
```bash
docker compose down
# Ou modifier le port dans docker-compose.yml
```

**Reconstruire sans cache :**
```bash
docker compose build --no-cache
docker compose up -d
```
