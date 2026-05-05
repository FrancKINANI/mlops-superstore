# Dockerfile
FROM python:3.13

WORKDIR /app

# Dépendances système et Python
COPY requirements-api.txt .
RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && pip install --no-cache-dir -r requirements-api.txt \
    && apt-get purge -y gcc && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Code source
COPY src/ ./src/

# Port exposé
EXPOSE 8000

# Lancement
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]