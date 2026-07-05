# ============================================================================
# predict/infer.py  —  PREDICT ONE RACE  (Step 17)
# ----------------------------------------------------------------------------
# Runs on `uv run f1pred predict --season 2026 --round 8`. It trains the model on
# every race BEFORE the chosen race (so it's a fair prediction, no cheating), ranks
# that race's drivers, prints the predicted order next to the real order, and saves
# it as a JSON file. This is what a website/demo would call.
# ============================================================================

"""Predict the finishing order for a single race.

Trains the ranker on all races *strictly before* the target race (leakage-safe), then
ranks the target grid. Works for any race present in the feature table; the predicted
order is compared to the actual result when it is known.
"""

from __future__ import annotations

import json  # to save the prediction as a .json text file

import pandas as pd

from f1pred.config import get_config
from f1pred.logging_utils import get_logger
from f1pred.models.ranker import RankerModel

log = get_logger(__name__)


# noqa: A002 tells the linter it's fine to name an argument `round` even though Python
# has a built-in `round()` function — here it reads naturally as an F1 term.
def predict_race(season: int, round: int) -> pd.DataFrame:  # noqa: A002 - matches F1 vocabulary
    """Return a ranked prediction for (season, round); persist it as JSON."""
    cfg = get_config()
    cfg.paths.ensure()
    feats_path = cfg.paths.features_dir / "features.parquet"
    if not feats_path.exists():  # need features first
        raise FileNotFoundError(f"No features at {feats_path}; run `f1pred features` first.")

    df = pd.read_parquet(feats_path)  # load all features
    # Pick out just the rows for the race we want. "&" = AND (both conditions true).
    race = df[(df["season"] == season) & (df["round"] == round)]
    if race.empty:  # we don't have that race
        raise ValueError(
            f"Race {season} R{round} not in feature table. "
            "Ingest+build it first, or it has not happened yet."
        )

    race_date = race["event_date"].iloc[0]  # when this race happened
    train = df[df["event_date"] < race_date]  # ONLY races before it (fair training)
    if len(train) < 100:  # not enough history to train a decent model
        raise ValueError(
            f"Too little history before {season} R{round} to train ({len(train)} rows)."
        )

    model = RankerModel().fit(train)  # train fresh on the past
    scores = model.predict(race)  # score each driver in the target race

    # Build a small results table with the useful columns. .copy() to be safe.
    out = race[
        ["season", "round", "event_name", "driver_id", "driver_name", "constructor_id"]
    ].copy()
    out["score"] = scores  # attach the model's scores
    # Sort so the highest score (best predicted) is first; reset row numbers.
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    out["predicted_rank"] = out.index + 1  # 1st predicted, 2nd predicted, ... (index starts at 0)

    # Bring in the ACTUAL finishing position so we can compare prediction vs. reality.
    actual = race[["driver_id", "position"]].rename(columns={"position": "actual_position"})
    out = out.merge(actual, on="driver_id", how="left")

    # Print a readable table.
    event_name = out["event_name"].iloc[0]
    log.info("Predicted order — %s %s R%s:", season, event_name, round)
    show = out[["predicted_rank", "driver_id", "constructor_id", "actual_position"]]
    log.info("\n%s", show.to_string(index=False))

    # Save the full prediction as a JSON file (e.g. pred_2026_r08.json). :02d = 2 digits.
    dest = cfg.paths.predictions_dir / f"pred_{season}_r{round:02d}.json"
    # to_dict(orient="records") makes a list of {column: value} rows; json.dumps -> text.
    dest.write_text(json.dumps(out.to_dict(orient="records"), default=str, indent=2))
    log.info("wrote prediction -> %s", dest)
    return out
