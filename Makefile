# Makefile pour exécuter les étapes du pipeline MLOps

.PHONY: preprocess train register pipeline deploy api install

preprocess:
	@bash scripts/preprocess.sh

train:
	@bash scripts/train.sh

register:
	@bash scripts/register.sh

pipeline:
	@bash scripts/run_pipeline.sh

deploy:
	@bash scripts/deploy.sh

api:
	@docker compose up -d

install:
	@python -m venv .venv
	@.venv/bin/python -m pip install --upgrade pip
	@.venv/bin/pip install -r requirements.txt
