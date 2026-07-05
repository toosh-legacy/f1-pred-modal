# ============================================================================
# models/baselines.py  —  THE "DUMB" MODELS TO BEAT  (Step 14)
# ----------------------------------------------------------------------------
# Before trusting a fancy model, we must show it beats SIMPLE ideas. If our
# LightGBM model can't beat "just predict they finish where they started", it's
# useless. These three baselines are those simple ideas. They all share the same
# shape as the real model: a .fit() method (learn) and a .predict() method (guess),
# returning a score per driver (higher = predicted to finish better).
# ============================================================================

"""Baseline scorers. Each returns a per-row score (higher = predicted to finish better),
so they plug into the same evaluation path as the learned ranker.

Baselines exist to prove the ranker adds signal. They rely only on leakage-safe columns
already present in the feature table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# The PARENT class. The other baselines "inherit" from it (reuse its shape).
# Think of it as the template every baseline must follow.
class Baseline:
    """Fit/predict interface mirroring the ranker. Most baselines need no fitting."""

    name: str = "baseline"  # a label used in the results table

    # .fit() = "learn from training data". Most baselines don't need to learn anything,
    # so the default just returns itself unchanged.
    def fit(self, train: pd.DataFrame) -> Baseline:  # noqa: ARG002 - stateless by default
        return self

    # .predict() = "produce a score per driver". The parent leaves this unfinished
    # (raises an error) so each child MUST provide its own version.
    def predict(self, test: pd.DataFrame) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError


# BASELINE 1: predict that drivers finish exactly where they started (grid order).
# This is famously hard to beat in F1!
class GridBaseline(Baseline):
    """Predict finishing order == starting grid order. The classic hard-to-beat baseline."""

    name = "grid"

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        grid = pd.to_numeric(test["grid"], errors="coerce")  # starting positions as numbers
        # Grid 0 (pit-lane start) -> treat as back of grid.
        grid = grid.replace(0, grid.max() + 1)
        # Return NEGATIVE grid: pole (grid 1) gets the highest score (-1 > -20).
        return (-grid).to_numpy(dtype=float)


# BASELINE 2: predict by recent form (average finish over the last 5 races).
class DriverFormBaseline(Baseline):
    """Score by recent average finishing position (better recent form ranked higher)."""

    name = "driver_form"

    # __init__ is the "constructor" — it runs when you create the object. Here it just
    # remembers WHICH feature column to use (default: the 5-race rolling average).
    def __init__(self, col: str = "driver_pos_roll5"):
        self.col = col

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        val = pd.to_numeric(test.get(self.col), errors="coerce")  # each driver's recent avg
        # Missing form (debut) -> mid-pack fallback so the driver is not auto-last.
        val = val.fillna(val.median() if val.notna().any() else 10.0)
        return (-val).to_numpy(dtype=float)  # negative so lower avg = higher score


# BASELINE 3: an "Elo" rating for each TEAM (like chess ratings). Teams that keep
# beating expectations climb; teams that underperform drop. This one DOES learn.
class ConstructorEloBaseline(Baseline):
    """Elo rating over constructors, updated causally race-by-race on the training set.

    Each race, a team's expected score vs the field is compared to its actual mean finish;
    ratings drift toward teams that consistently out-finish expectation. At predict time a
    driver inherits their constructor's current rating (unknown teams get the base rating).
    """

    name = "constructor_elo"

    def __init__(self, k: float = 24.0, base: float = 1500.0):
        self.k = k  # how fast ratings move after each race
        self.base = base  # the starting rating for a brand-new team
        self.ratings: dict[str, float] = {}  # team_id -> current rating (learned in fit)

    def fit(self, train: pd.DataFrame) -> ConstructorEloBaseline:
        self.ratings = {}  # start fresh
        # Go through races in time order (oldest first) so ratings build up causally.
        for _, race in train.sort_values(["event_date", "season", "round"]).groupby(
            ["season", "round"], sort=False
        ):
            teams = race.groupby("constructor_id")["position"].mean()  # each team's avg finish
            if len(teams) < 2:  # need at least 2 teams to compare
                continue
            # Turn "average finish" into an actual result in 0-1 (best team = 1.0).
            ranks = teams.rank(ascending=True)
            actual = 1.0 - (ranks - 1) / (len(teams) - 1)
            cur = {t: self.ratings.get(t, self.base) for t in teams.index}  # current ratings
            mean_r = np.mean(list(cur.values()))  # average rating of the field this race
            for t in teams.index:
                # Elo's "expected score": how well we EXPECTED this team to do vs. the field.
                expected = 1.0 / (1.0 + 10 ** ((mean_r - cur[t]) / 400.0))
                # Update: if they did better than expected (actual > expected), rating rises.
                self.ratings[t] = cur[t] + self.k * (actual[t] - expected)
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        # Each driver's score = their team's learned rating (unknown team -> base rating).
        return test["constructor_id"].map(lambda t: self.ratings.get(t, self.base)).to_numpy(float)


# Convenience: get all three baselines as a list, ready to loop over.
def default_baselines() -> list[Baseline]:
    return [GridBaseline(), DriverFormBaseline(), ConstructorEloBaseline()]
