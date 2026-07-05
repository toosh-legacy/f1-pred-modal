# Convenience targets. On Windows without GNU make, run the underlying
# `uv run ...` commands directly (shown in each recipe).

.PHONY: install lint fmt test ingest features backtest train predict app clean

install:            ## Create env and install dev deps
	uv sync --extra dev

lint:               ## Ruff lint
	uv run ruff check .

fmt:                ## Ruff format + import sort
	uv run ruff format .
	uv run ruff check --fix .

test:               ## Run unit tests (skip network tests)
	uv run pytest -m "not network"

ingest:             ## Pull FastF1 + Jolpica data -> data/raw
	uv run f1pred ingest

features:           ## Build the feature table -> data/features
	uv run f1pred features

backtest:           ## Rolling-origin backtest, log to MLflow
	uv run f1pred backtest

train:              ## Train the final ranker on all data
	uv run f1pred train

predict:            ## Predict one race: make predict SEASON=2026 ROUND=11
	uv run f1pred predict --season $(SEASON) --round $(ROUND)

app:                ## Launch the Gradio demo locally
	uv run --extra app python app/app.py

clean:              ## Remove caches (keeps data/raw)
	rm -rf .pytest_cache .ruff_cache mlruns
