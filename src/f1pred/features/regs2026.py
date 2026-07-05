# ============================================================================
# features/regs2026.py  —  SPECIAL 2026 RULE-CHANGE FLAGS  (Step 11)
# ----------------------------------------------------------------------------
# 2026 is a big "regulation reset" in F1: new engines, a brand-new team (Cadillac),
# a rebranded team (Audi), engine-supplier swaps, etc. FastF1's results don't tell
# us these things, so we keep them in a small hand-typed file:
#   data/reference/entries_2026.csv
# This file reads that CSV and turns it into simple 0/1 "flag" columns the model
# can use — but ONLY for 2026 rows (older seasons get zeros).
# ============================================================================

"""2026 regulation-context features from the curated reference table.

Joins ``data/reference/entries_2026.csv`` (power unit + team/PU-change status) onto each
row by a normalized constructor key. Unmatched constructors fall back to neutral defaults
and are logged, so a missing/renamed team never silently corrupts the join.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from f1pred.config import REPO_ROOT  # to locate the reference CSV
from f1pred.logging_utils import get_logger

log = get_logger(__name__)

# The default location of our hand-typed 2026 team info.
DEFAULT_REF = REPO_ROOT / "data" / "reference" / "entries_2026.csv"

# The possible "engine situation" categories in the CSV. Each becomes its own 0/1 column.
_PU_STATUS_FLAGS = ["new_manufacturer", "supplier_change", "customer"]  # PU = Power Unit (engine)
# The possible "team situation" categories.
_TEAM_STATUS_FLAGS = ["new_team", "rebranded"]


# Helper: normalize a team name to a consistent key (same idea as circuit_key).
def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


# Read the CSV into a table and clean its key column.
def load_reference(path: Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else DEFAULT_REF  # use given path or the default
    # pd.read_csv reads a comma-separated file. comment="#" ignores lines starting with #.
    ref = pd.read_csv(path, comment="#")
    ref["constructor_key"] = ref["constructor_key"].map(_norm)  # normalize the team keys
    return ref


def add_reg_features(df: pd.DataFrame, path: Path | None = None) -> pd.DataFrame:
    """Add 2026 regulation flags. Non-target seasons get zeros (regs not applicable)."""
    df = df.copy()  # work on a copy so we don't change the caller's table
    ref = load_reference(path)  # load the 2026 info table
    df["_ckey"] = df["constructor_id"].map(_norm)  # a normalized key to join on

    # .merge glues the reference info onto our rows, matching team keys.
    # how="left" keeps all our rows even if a team isn't in the reference (they get blanks).
    merged = df.merge(
        ref, left_on="_ckey", right_on="constructor_key", how="left", suffixes=("", "_ref")
    )

    # Safety check: find any 2026 team that DIDN'T match the reference (power_unit is blank).
    # If we find some, warn — it means our CSV is missing/misnamed a team.
    unmatched = merged.loc[
        (merged["season"] == 2026) & (merged["power_unit"].isna()), "constructor_id"
    ].unique()
    if len(unmatched):
        log.warning("2026 constructors unmatched in reference table: %s", sorted(unmatched))

    # Turn each engine-status category into a 0/1 column.
    # (merged["pu_status"] == flag) gives True/False; .astype(int) turns it into 1/0.
    for flag in _PU_STATUS_FLAGS:
        merged[f"reg_pu_{flag}"] = (merged["pu_status"] == flag).astype(int)
    # Same for the team-status categories.
    for flag in _TEAM_STATUS_FLAGS:
        merged[f"reg_team_{flag}"] = (merged["team_status"] == flag).astype(int)

    # These 2026 rules don't apply to older seasons, so zero the flags everywhere except 2026.
    reg_cols = [c for c in merged.columns if c.startswith("reg_")]  # all the flag columns
    non_2026 = merged["season"] != 2026  # True for every row that is NOT 2026
    merged.loc[non_2026, reg_cols] = 0  # set those rows' flags to 0

    # Drop the temporary/text helper columns we no longer need; keep the numeric flags.
    return merged.drop(
        columns=["_ckey", "constructor_key", "power_unit", "pu_status", "team_status"]
    )
