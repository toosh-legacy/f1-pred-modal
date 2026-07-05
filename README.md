# f1pred

Predicting **2026 Formula 1 race results** with FastF1 data and LightGBM.

Notebook-driven: exploration and modeling live in `notebooks/`; reusable helpers live in
`src/f1pred/`.

## Layout

```
data/
  raw/           # ingested FastF1 data (gitignored)
  reference/     # hand-curated inputs (entries_2026.csv)
notebooks/       # where the work happens
src/f1pred/      # minimal helpers imported by notebooks
results/         # model outputs, predictions, figures
tests/           # slim
```

## Setup

```bash
uv sync --extra dev        # install core + dev (jupyter, pytest, ruff)
uv run jupyter lab         # start notebooks
uv run pytest              # run tests
```
