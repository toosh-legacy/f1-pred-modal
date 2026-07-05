# f1pred — Predicting 2026 Formula 1 Race Results

Learning-to-rank model that predicts the **finishing order** of Formula 1 races under the
2026 regulation reset, using [FastF1](https://docs.fastf1.dev/) timing data and historical
results. Winner, podium, and points-finish predictions all derive from a single ranked
output.

> **Why 2026 is interesting:** an all-new formula (new power units, active aero, lighter
> cars) resets competitive order with limited new-formula data — a genuine forecasting
> challenge, not curve-fitting a stable grid.

## Architecture

```
FastF1 API ──┐
Jolpica/Ergast├─► ingest ─► raw parquet ─► feature build ─► train / backtest ─► model registry
(results,     │   (cache)    (data/raw)     (data/features)  (LightGBM ranker)  (HF Hub + MLflow)
 schedule)    │                                                     │
News (v2) ────┘                                                     ▼
                                          Modal (weekly cron) ─► predict next race
                                                                   ▼
                                              Hugging Face Space (Gradio) ─► live demo
```

## Approach highlights

- **Learning-to-rank** (LightGBM `LGBMRanker`, LambdaMART) grouped per race.
- **Leakage-safe features** — every feature uses only pre-race information (rolling/shifted
  windows); invariants enforced by tests (`tests/test_leakage.py`).
- **Rolling-origin backtesting** — train on past rounds, predict the next; walk forward
  across seasons. No random splits. 2026-only performance reported separately.
- **Baselines** (grid-position, driver-form, constructor-Elo) prove the model adds signal.
- **MLOps** — `uv` env, MLflow tracking, `ruff` + `pytest` + GitHub Actions CI.

## Quickstart

```bash
uv sync --extra dev            # install
uv run pytest -m "not network" # unit tests
uv run f1pred ingest           # pull FastF1 + Jolpica -> data/raw
uv run f1pred features         # build feature table
uv run f1pred backtest         # rolling-origin evaluation
uv run f1pred predict --season 2026 --round 11
```

(GNU `make` targets mirror these: `make install`, `make test`, `make backtest`, …)

## Project layout

| Path | Purpose |
|---|---|
| `src/f1pred/ingest/` | FastF1 + Jolpica loaders, disk cache |
| `src/f1pred/features/` | Feature families + leakage guards |
| `src/f1pred/models/` | LightGBM ranker, baselines, calibration |
| `src/f1pred/eval/` | Rolling-origin backtest + ranking metrics |
| `src/f1pred/predict/` | Next-race inference |
| `modal_app/` | Modal serverless training + weekly cron |
| `app/` | Gradio demo for Hugging Face Spaces |
| `tests/` | Leakage, schema, and metric tests |

## Results (rolling-origin backtest, held-out seasons 2023–2026)

Mean over held-out races. Higher is better. The learned ranker beats the strong
grid-position baseline on rank correlation, NDCG@10, and top-3 overlap, and is strongest
in the **2026 regulation-reset season** — exactly where prior-year grid order is least
reliable.

| Model | Spearman | NDCG@3 | NDCG@10 | Top-3 overlap | Top-1 (winner) |
|---|---|---|---|---|---|
| **LGBM ranker** | **0.661** | 0.766 | **0.854** | **0.688** | 0.538 |
| Grid baseline | 0.655 | **0.771** | 0.850 | 0.675 | **0.603** |
| Driver-form baseline | 0.563 | 0.618 | 0.765 | 0.534 | 0.423 |
| Constructor-Elo baseline | 0.482 | 0.419 | 0.641 | 0.372 | 0.218 |

Ranker by season (NDCG@3): 2023 **0.80**, 2024 0.71, 2025 0.81, **2026 0.72**.

**Honest limitation:** grid position remains marginally better at calling the *exact*
winner — the model lacks race-pace / qualifying-gap features (documented next step).

## Status

Milestones 1–5 complete: scaffold, ingestion (2018–2026, 3,634 races-rows), leakage-safe
features, baselines + rolling-origin backtest, and the LightGBM ranker. Next: deployment
(HF Space + Modal), then NLP v2. Full roadmap in the project plan file.

## License

MIT
