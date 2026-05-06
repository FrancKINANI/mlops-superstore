# --- Étape 1 : Builder ---
FROM python:3.12-slim AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ python3-dev git \
    && rm -rf /var/lib/apt/lists/*

# Créer un venv isolé
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

COPY requirements-api.txt .

# Installation des outils de base puis des dépendances
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-api.txt

# --- Étape 2 : Image Finale ---
FROM python:3.12-slim
WORKDIR /app

# Dépendances système minimales pour le runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copie du venv
COPY --from=builder /venv /venv

# Copie de tout le projet (en respectant .dockerignore)
COPY . .

# Configuration de l'environnement
ENV PATH="/venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

# Sécurité : utilisateur non-root
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Commande de lancement (uvicorn charge l'app depuis le dossier src)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]