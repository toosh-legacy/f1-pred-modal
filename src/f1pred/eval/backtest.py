# ============================================================================
# eval/backtest.py  —  THE FAIR EXAM  (Step 16)  ⭐ "is the model actually good?"
# ----------------------------------------------------------------------------
# We can't test a model on races it already learned from — it would just recite
# the answers. So we "walk forward through time": train on OLD seasons, predict a
# NEWER season the model has never seen, then move forward. This mimics real life
# (you can only ever use the past to predict the future). This file runs that exam
# for our ranker AND the 3 baselines, then prints the comparison table.
# ============================================================================

"""Rolling-origin backtest: train on the past, predict held-out races, walk forward.

Two refit granularities:
  * ``per_season`` (default): for each held-out season S, train once on all seasons < S,
    then predict every race in S. Fast; the honest "train on history, test on an unseen
    season" story. In-season form still updates because features are computed causally.
  * ``per_race``: retrain before every target race on all strictly-earlier races. Gold
    standard, slower.

Never a random split — always time-ordered, group-aware (one race == one group).
"""

from __future__ import annotations

from collections.abc import Callable  # a type hint meaning "a function"

import numpy as np
import pandas as pd

from f1pred.config import get_config
from f1pred.eval import metrics as M  # our scorecards from Step 13 (nicknamed M)
from f1pred.logging_utils import get_logger
from f1pred.models.baselines import default_baselines  # the 3 baselines
from f1pred.models.ranker import RankerModel  # our real model
from f1pred.schema import GROUP_KEY

log = get_logger(__name__)

# A "factory" is a function that makes a fresh, untrained model each time we call it.
# We need fresh models because we retrain many times during the walk-forward.
ModelFactory = Callable[[], object]


# Helper: grade ONE race with all our scorecards, returning a dict of {metric: value}.
def _race_metrics(scores: np.ndarray, position: pd.Series, eval_at: list[int]) -> dict:
    pos = position.to_numpy(dtype=float)  # the real finishing positions
    row = {
        "top1": M.top1_hit(scores, pos),  # did we pick the winner?
        "spearman": M.spearman(scores, pos),  # overall order agreement
    }
    for k in eval_at:  # for each k in [1, 3, 10]
        row[f"ndcg@{k}"] = M.ndcg_at_k(scores, pos, k)  # ranking quality at top-k
        row[f"top{k}"] = M.topk_overlap(scores, pos, k)  # how many top-k we caught
    return row


# THE CORE: walk forward through seasons, training on the past, scoring the future.
def walk_forward(
    df: pd.DataFrame,
    factory: ModelFactory,  # makes a fresh model
    eval_seasons: list[int],  # which seasons to test on
    *,
    refit: str = "per_season",  # "per_season" (fast) or "per_race" (gold standard)
    eval_at: list[int] | None = None,
) -> pd.DataFrame:
    """Return one row of metrics per held-out race for the given model factory."""
    eval_at = eval_at or get_config().model.eval_at
    df = df.sort_values(["event_date", *GROUP_KEY]).reset_index(drop=True)  # time order
    rows: list[dict] = []  # collect one metrics-dict per race

    for season in eval_seasons:  # test each season, e.g. 2023, 2024, 2025, 2026
        season_races = df[df["season"] == season]  # just this season's rows
        if season_races.empty:  # no data for this season -> skip
            continue

        # "per_season" mode: train ONCE on all EARLIER seasons before testing this one.
        if refit == "per_season":
            train = df[df["season"] < season]  # everything before this season
            if train.empty:  # nothing to learn from (e.g. the very first season) -> skip
                continue
            model = factory().fit(train)  # make a fresh model and train it

        # Now go race by race within the season and score each one.
        for (s, rnd), race in season_races.groupby(GROUP_KEY, sort=True):
            # "per_race" mode: retrain on EVERYTHING before this exact race (more rigorous).
            if refit == "per_race":
                race_date = race["event_date"].iloc[0]  # this race's date
                train = df[df["event_date"] < race_date]  # all strictly-earlier races
                if train["season"].nunique() < 1 or len(train) < 40:  # too little history
                    continue
                model = factory().fit(train)  # retrain fresh
            scores = np.asarray(model.predict(race), dtype=float)  # the model's guesses
            m = _race_metrics(scores, race["position"], eval_at)  # grade them
            m.update({"season": int(s), "round": int(rnd), "n": len(race)})  # tag the race
            rows.append(m)

    return pd.DataFrame(rows)  # a table: one row per tested race, columns = metrics


# Helper: average each metric column down to a single number (the model's overall score).
def _summary(per_race: pd.DataFrame, metric_cols: list[str]) -> pd.Series:
    return per_race[metric_cols].mean()


# THE ENTRY POINT: run the exam for the ranker + all baselines and report. Runs on
# `uv run f1pred backtest`.
def run_backtest(refit: str = "per_season") -> dict[str, pd.DataFrame]:
    """Backtest the ranker + baselines, log a comparison table, persist per-race results."""
    cfg = get_config()
    cfg.paths.ensure()
    feats_path = cfg.paths.features_dir / "features.parquet"
    if not feats_path.exists():  # need features first
        raise FileNotFoundError(f"No features at {feats_path}; run `f1pred features` first.")
    df = pd.read_parquet(feats_path)  # load the feature table
    eval_seasons = cfg.backtest.eval_seasons  # seasons to test on
    eval_at = cfg.model.eval_at  # the k values for top-k / ndcg

    # Build a dictionary of {name: factory}. First our ranker, then each baseline.
    factories: dict[str, ModelFactory] = {"lgbm_ranker": RankerModel}
    for b in default_baselines():
        # lambda b=b: type(b)() makes a fresh copy of that baseline each call.
        factories[b.name] = lambda b=b: type(b)()

    # The list of metric columns we'll average and display.
    metric_cols = (
        ["top1", "spearman"] + [f"ndcg@{k}" for k in eval_at] + [f"top{k}" for k in eval_at]
    )

    # Run the walk-forward exam for every model and collect results.
    per_model: dict[str, pd.DataFrame] = {}
    summary_rows = {}
    for name, factory in factories.items():
        log.info("backtesting %s (refit=%s, eval_seasons=%s)", name, refit, eval_seasons)
        res = walk_forward(df, factory, eval_seasons, refit=refit, eval_at=eval_at)
        per_model[name] = res  # keep the per-race table
        summary_rows[name] = _summary(res, metric_cols)  # keep the averaged scores

    # Build and print the comparison table (models as rows, metrics as columns).
    summary = pd.DataFrame(summary_rows).T
    log.info(
        "\n=== Backtest summary (mean over held-out races) ===\n%s", summary.round(3).to_string()
    )

    # Also print how the ranker did SEASON BY SEASON (esp. the tricky 2026 reset year).
    ranker_res = per_model["lgbm_ranker"]
    if not ranker_res.empty:
        by_season = ranker_res.groupby("season")[["ndcg@3", "top1", "top3"]].mean()
        log.info("\n=== lgbm_ranker by season ===\n%s", by_season.round(3).to_string())

    # Persist results BEFORE optional tracking so a tracking failure never loses them.
    out = cfg.paths.predictions_dir / "backtest_per_race.parquet"
    # Stack every model's per-race table (tagged with the model name) and save it.
    pd.concat([r.assign(model=n) for n, r in per_model.items()], ignore_index=True).to_parquet(
        out, index=False
    )
    log.info("wrote per-race backtest metrics -> %s", out)
    summary.to_csv(cfg.paths.predictions_dir / "backtest_summary.csv")  # also a readable CSV

    _log_mlflow(summary, cfg)  # optionally record the run in MLflow (never fatal)
    return per_model


# Optional: log the results to MLflow (an experiment-tracking tool). Wrapped so that
# if MLflow isn't set up or errors, the backtest still succeeds.
def _log_mlflow(summary: pd.DataFrame, cfg) -> None:
    """Best-effort MLflow logging; never fatal to the backtest."""
    try:
        from pathlib import Path

        import mlflow

        # Resolve a file-store URI to an absolute path (relative file: URIs break on Windows).
        uri = cfg.tracking.mlflow_uri
        if uri.startswith("file:"):
            uri = (Path(uri[5:].lstrip("/")).resolve()).as_uri()
        mlflow.set_tracking_uri(uri)  # where to store the run
        mlflow.set_experiment(cfg.tracking.experiment)  # the experiment name
        with mlflow.start_run(run_name="backtest"):  # begin recording
            mlflow.log_params({"eval_seasons": str(cfg.backtest.eval_seasons), **cfg.model.params})
            # Record every metric for every model.
            for model_name, row in summary.iterrows():
                for metric, value in row.items():
                    mlflow.log_metric(f"{model_name}/{metric.replace('@', '_at_')}", float(value))
    except Exception as exc:  # noqa: BLE001 - tracking is optional, keep the run alive
        log.warning("MLflow logging skipped: %s", exc)
