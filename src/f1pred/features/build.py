# ============================================================================
# features/build.py  —  THE FEATURE CONDUCTOR  (Step 12)
# ----------------------------------------------------------------------------
# This runs when you type `uv run f1pred features`. It loads the raw results,
# runs ALL the feature builders from Steps 8-11 in order, adds a cleaned starting-
# position clue, and saves the finished feature table. It also defines which
# columns count as "features" (clues the model uses) vs. "labels/ids" (things the
# model must NOT use, like the answer or the driver's name).
# ============================================================================

"""Assemble the full leakage-safe feature table from raw results.

Chains the feature families (driver -> constructor -> track -> 2026 regs) and writes
``data/features/features.parquet`` keyed by (season, round, driver_id). ``grid`` is kept
as a direct feature: starting position is set at qualifying, i.e. before the race, so it
is legitimately pre-race information.
"""

from __future__ import annotations

import pandas as pd

from f1pred.config import get_config

# Import each feature builder we wrote in Steps 8-11.
from f1pred.features.constructor import add_constructor_form
from f1pred.features.driver import add_driver_form
from f1pred.features.regs2026 import add_reg_features
from f1pred.features.track import add_track_features
from f1pred.logging_utils import get_logger
from f1pred.schema import KEY_COLUMNS, TARGET_COLUMN

log = get_logger(__name__)

# A "set" (note the {curly braces}) of columns the model must NEVER use as input.
# These are ids, names, and the answer itself (position/points/dnf/status), plus the
# raw ambiguous grid. Everything ELSE that's numeric becomes a feature.
NON_FEATURE_COLUMNS = {
    "season",
    "round",
    "event_name",
    "event_date",
    "driver_id",
    "driver_name",
    "constructor_id",
    "circuit_key",
    "position",  # THE ANSWER — using it would be cheating
    "points",  # part of the outcome
    "status",
    "dnf",  # part of the outcome
    # Raw grid is ambiguous (0 == pit-lane start); the model uses `grid_start` instead.
    "grid",
}


# Given a table, return the list of column names that ARE features (model inputs).
def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model-input columns: everything numeric that is not identity/target/leaky."""
    # Keep a column only if: it's NOT in the "never use" set AND it holds numbers.
    return [
        c
        for c in df.columns
        if c not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    ]


# The PURE builder: raw results in -> full feature table out. No file reading/writing,
# so it's easy to test. `windows=None` means "use the windows from config".
def build_feature_table(results: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """Pure transform: raw canonical results -> feature table. No I/O."""
    windows = windows or get_config().features.form_windows
    df = results.copy()  # work on a copy

    # --- Cleaned starting-position clue ---
    # Raw grid uses 0 for a pit-lane start, which would look like "better than pole" to
    # the model. We replace 0/blank with "back of the grid" so lower always means better.
    grid = pd.to_numeric(df["grid"], errors="coerce")  # ensure numbers
    # "back of grid" = one worse than the biggest real grid number (or 21 if none exist).
    back = float(grid[grid > 0].max() + 1) if (grid > 0).any() else 21.0
    # .where(grid > 0, back) keeps grid where it's positive, else uses `back`.
    df["grid_start"] = grid.where(grid > 0, back).fillna(back)

    # --- Run each feature family in order; each returns a bigger table ---
    df = add_driver_form(df, windows)  # Step 8 clues
    df = add_constructor_form(df, windows)  # Step 9 clues
    df = add_track_features(df)  # Step 10 clues
    df = add_reg_features(df)  # Step 11 clues
    # Sort the final table by race then finishing position (neat + needed for ranking).
    df = df.sort_values(["season", "round", TARGET_COLUMN]).reset_index(drop=True)
    return df


# The wrapper that reads the file, builds features, and saves the result.
def build_features() -> pd.DataFrame:
    """Read raw results, build features, write parquet, return the table."""
    cfg = get_config()
    cfg.paths.ensure()  # make sure folders exist
    raw_path = cfg.paths.raw_dir / "results.parquet"  # Station 1's output
    if not raw_path.exists():  # can't build features without raw data
        raise FileNotFoundError(f"No raw results at {raw_path}; run `f1pred ingest` first.")

    results = pd.read_parquet(raw_path)  # load the raw results
    df = build_feature_table(results, cfg.features.form_windows)  # build all features

    out_path = cfg.paths.features_dir / "features.parquet"  # where to save
    df.to_parquet(out_path, index=False)  # SAVE the feature table
    feats = feature_columns(df)  # count how many feature columns we made
    log.info(
        "wrote %d rows x %d features -> %s (keys=%s)",
        len(df),
        len(feats),
        out_path,
        KEY_COLUMNS,
    )
    return df
