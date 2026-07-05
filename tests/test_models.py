"""Model unit tests on synthetic feature data (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1pred.features.build import build_feature_table, feature_columns
from f1pred.models.baselines import ConstructorEloBaseline, GridBaseline
from f1pred.models.ranker import RankerModel, relevance_label


def _synthetic(n_races: int = 8) -> pd.DataFrame:
    drivers = [("a", "t1"), ("b", "t1"), ("c", "t2"), ("d", "t2")]
    rows = []
    base = pd.Timestamp("2024-03-02")
    for r in range(1, n_races + 1):
        date = base + pd.DateOffset(days=14 * (r - 1))
        # t1 is consistently faster than t2 -> learnable signal.
        order = ["a", "b", "c", "d"] if r % 2 else ["b", "a", "d", "c"]
        for pos, drv in enumerate(order, start=1):
            team = dict(drivers)[drv]
            rows.append(
                {
                    "season": 2024,
                    "round": r,
                    "event_name": "Test Grand Prix",
                    "event_date": date,
                    "driver_id": drv,
                    "driver_name": drv,
                    "constructor_id": team,
                    "grid": pos,
                    "position": float(pos),
                    "points": float(max(0, 26 - pos * 5)),
                    "status": "Finished",
                    "dnf": False,
                }
            )
    return build_feature_table(pd.DataFrame(rows), windows=[3])


def test_relevance_label_winner_highest():
    rel = relevance_label(pd.Series([1.0, 2.0, 20.0, np.nan]), max_grid=22)
    assert rel[0] > rel[1] > rel[2]
    assert rel[3] == 0  # DNF


def test_grid_baseline_orders_by_grid():
    df = _synthetic()
    race = df[df["round"] == df["round"].max()]
    scores = GridBaseline().predict(race)
    # Highest score must be pole sitter (grid == 1).
    assert race.iloc[np.argmax(scores)]["grid"] == 1


def test_constructor_elo_prefers_stronger_team():
    df = _synthetic(n_races=10)
    elo = ConstructorEloBaseline().fit(df)
    assert elo.ratings["t1"] > elo.ratings["t2"]


def test_ranker_fits_and_ranks():
    df = _synthetic(n_races=10)
    feats = feature_columns(df)
    assert feats
    model = RankerModel().fit(df)
    race = df[df["round"] == df["round"].max()]
    scores = model.predict(race)
    assert len(scores) == len(race)
    assert np.isfinite(scores).all()
