# ============================================================================
# models/ranker.py  —  THE REAL MODEL  (Step 15)
# ----------------------------------------------------------------------------
# This wraps LightGBM (a powerful, popular machine-learning library) in a friendly
# class called RankerModel with three simple actions:
#   .fit(data)     -> learn patterns from past races
#   .predict(race) -> score each driver for a race (higher = predicted better)
#   .save()/.load()-> store/restore the trained model on disk
#
# "Learning-to-rank" means the model learns to ORDER drivers within a race, rather
# than predict an exact finishing number. LightGBM's "LambdaMART" method is built
# for exactly this.
# ============================================================================

"""LightGBM learning-to-rank model (LambdaMART) for race finishing order.

The relevance label is derived from finishing position (winner = highest relevance) so
LightGBM's ``lambdarank`` objective optimizes the ranking directly. Groups are races.
"""

from __future__ import annotations

import pickle  # Python's tool for saving objects to a file and loading them back
from pathlib import Path

import numpy as np
import pandas as pd

from f1pred.config import get_config
from f1pred.features.build import feature_columns  # to know which columns are model inputs
from f1pred.logging_utils import get_logger
from f1pred.schema import GROUP_KEY  # = ["season", "round"], defines one race

log = get_logger(__name__)


# The model needs a "label" (the thing to learn) as small whole numbers where BIGGER =
# better. Finishing position is the opposite (1 = best), so we flip it here.
def relevance_label(position: pd.Series, max_grid: int = 22) -> np.ndarray:
    """Map finishing position -> integer graded relevance (winner highest).

    LightGBM lambdarank needs small non-negative integer labels. Winner -> max_grid-1,
    last -> 0. DNFs (NaN position) -> 0.
    """
    pos = pd.to_numeric(position, errors="coerce")  # ensure numbers
    rel = np.clip(max_grid - pos, 0, None)  # winner(1)->21, last(22)->0; clip stops negatives
    # Turn blanks (DNF) into 0, round to whole numbers, make them integers.
    return np.nan_to_num(rel, nan=0.0).round().astype(int)


# LightGBM needs to know how many rows belong to each race (each "group").
def _group_sizes(df: pd.DataFrame) -> list[int]:
    """Row counts per race, in the order rows appear (LightGBM's group format)."""
    return df.groupby(GROUP_KEY, sort=False).size().tolist()  # e.g. [20, 20, 22, ...]


# A "class" bundling the model plus its helper methods.
class RankerModel:
    """Thin wrapper around ``lightgbm.LGBMRanker`` with race-aware grouping."""

    name = "lgbm_ranker"  # label for the results table

    # Constructor: set up the model's knobs (params) and remember the feature list.
    def __init__(self, params: dict | None = None, features: list[str] | None = None):
        cfg = get_config()
        # Merge settings: start with config's params, add the random seed, then any
        # params passed in directly (which win). "**" unpacks a dict into another dict.
        self.params = {**cfg.model.params, "random_state": cfg.model.seed, **(params or {})}
        self.features = features  # which columns to use (decided at fit time if None)
        self.model = None  # the actual LightGBM model — empty until we .fit()

    # LEARN from training races.
    def fit(self, train: pd.DataFrame) -> RankerModel:
        import lightgbm as lgb  # heavy import, kept inside the method

        # Rows must be contiguous per race for LightGBM's group vector.
        train = train.sort_values([*GROUP_KEY]).reset_index(drop=True)
        # Decide the feature columns now if not already set.
        self.features = self.features or feature_columns(train)
        x = train[self.features]  # X = the inputs (the clues)
        y = relevance_label(train["position"])  # y = the answer to learn (as relevance)
        groups = _group_sizes(train)  # tells LightGBM where each race starts/ends

        self.model = lgb.LGBMRanker(**self.params)  # create the model with our knobs
        self.model.fit(x, y, group=groups)  # <-- THE ACTUAL LEARNING happens here
        return self  # return self so you can chain: RankerModel().fit(df)

    # GUESS scores for a race's drivers.
    def predict(self, test: pd.DataFrame) -> np.ndarray:
        if self.model is None:  # can't predict before training
            raise RuntimeError("RankerModel.predict called before fit().")
        return self.model.predict(test[self.features])  # higher score = predicted better

    # Which clues did the model rely on most? Great for understanding + the README.
    def feature_importance(self) -> pd.Series:
        if self.model is None:
            raise RuntimeError("no model")
        # Pair each feature name with its importance number, sorted biggest first.
        return pd.Series(self.model.feature_importances_, index=self.features).sort_values(
            ascending=False
        )

    # Save the trained model (plus its feature list + params) to a file.
    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)  # ensure the folder exists
        with open(path, "wb") as f:  # "wb" = write binary
            pickle.dump({"features": self.features, "params": self.params, "model": self.model}, f)

    # Load a previously saved model back into a RankerModel object.
    # @classmethod means you call it on the class: RankerModel.load("ranker.pkl").
    @classmethod
    def load(cls, path: Path) -> RankerModel:
        with open(path, "rb") as f:  # "rb" = read binary
            blob = pickle.load(f)
        obj = cls(params=blob["params"], features=blob["features"])  # rebuild the wrapper
        obj.model = blob["model"]  # put the trained model back in
        return obj


# Train the FINAL model on ALL data and save it. Runs on `uv run f1pred train`.
def train_final() -> RankerModel:
    """Train the ranker on the full feature table and save to models/ranker.pkl."""
    cfg = get_config()
    feats_path = cfg.paths.features_dir / "features.parquet"  # Station 2's output
    if not feats_path.exists():  # need features first
        raise FileNotFoundError(f"No features at {feats_path}; run `f1pred features` first.")
    df = pd.read_parquet(feats_path)  # load all features
    model = RankerModel().fit(df)  # train on everything
    out = cfg.paths.models_dir / "ranker.pkl"  # where to save
    model.save(out)  # save it
    log.info("trained final ranker on %d rows -> %s", len(df), out)
    # Print the 12 most important clues so we can see what the model cares about.
    log.info("top features:\n%s", model.feature_importance().head(12).to_string())
    return model
