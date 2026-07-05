# Data directory

All contents are **gitignored** (regenerated from source APIs). This file documents
provenance so the pipeline is reproducible.

## Layout

| Path | Contents | Produced by |
|---|---|---|
| `cache/fastf1/` | FastF1's own on-disk cache (session blobs) | `f1pred.ingest.cache` |
| `raw/results.parquet` | One row per driver per race: results, grid, status | `f1pred ingest` |
| `raw/laps.parquet` | Lap-level times (when telemetry enabled) | `f1pred ingest` |
| `features/features.parquet` | Leakage-safe feature table, keyed `(season, round, driver_id)` | `f1pred features` |
| `predictions/` | Per-round predicted finishing order (JSON/parquet) | `f1pred predict` |

## Sources & provenance

- **FastF1** (`fastf1` PyPI) — official F1 timing data. Full telemetry from 2018+.
  Pinned version recorded in `uv.lock`. Cached to `cache/fastf1/`.
- **Jolpica-F1** (`https://api.jolpi.ca/ergast/f1`) — maintained Ergast successor for
  historical results/standings and driver/constructor career history. Rate-limited;
  responses cached to `raw/`.
- **2026 entry/regulation CSV** (`data/reference/entries_2026.csv`, checked in) —
  hand-curated driver↔constructor↔power-unit mapping for the 2026 grid.

## Refresh

```bash
uv run f1pred ingest              # incremental; re-pulls only missing rounds
uv run f1pred features            # rebuild feature table from raw
```

## Leakage policy

Every feature is computed from data available **strictly before** the race it
describes (see `f1pred.features.leakage`). Schema and leakage invariants are enforced
in `tests/test_leakage.py` and the feature-build tests.
