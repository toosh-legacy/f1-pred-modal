# ============================================================================
# features/leakage.py  —  THE ANTI-CHEATING TOOLS  (Step 7)  ⭐ MOST IMPORTANT
# ----------------------------------------------------------------------------
# THE #1 RULE of this whole project:
#   When predicting race R, we may ONLY use information from races BEFORE R.
# If a "clue" (feature) accidentally includes race R's own result, the model is
# cheating — it looks amazing in testing but fails in real life. That mistake is
# called "leakage" (future information "leaks" into the past).
#
# This file gives us two safe tools to build clues from PAST races only. The magic
# ingredient is `.shift(1)`, which drops the current race from every calculation.
# ============================================================================

"""Leakage guards used by feature building and tests.

The cardinal rule: a feature row for race R may only use information observable
strictly *before* R starts. These helpers make that rule enforceable in code.
"""

from __future__ import annotations

import pandas as pd


# A safety check: make sure the rows are sorted oldest -> newest before we roll.
def assert_time_ordered(df: pd.DataFrame, time_col: str = "event_date") -> None:
    """Raise if rows are not sorted non-decreasing by ``time_col`` per group."""
    # .is_monotonic_increasing is True if the dates never go backwards.
    if not df[time_col].is_monotonic_increasing:
        # "raise" stops the program with an error message. Better to fail loudly here
        # than to silently compute wrong (leaky) features.
        raise AssertionError(f"{time_col} is not monotonically increasing; sort before rolling.")


# TOOL 1: a "rolling average of the last N races, NOT counting the current one".
def shifted_rolling(
    df: pd.DataFrame,  # the table
    group_cols: list[str],  # what to group by, e.g. ["driver_id"] (per driver)
    value_col: str,  # which column to average, e.g. "position"
    window: int,  # how many past races, e.g. 5
    agg: str = "mean",  # how to combine them: "mean", "sum", etc.
    order_col: str = "event_date",  # what to sort by (time)
) -> pd.Series:
    """Rolling aggregate over prior rows only (shifted by 1) within each group.

    Because the current row is excluded via ``.shift(1)``, the result for race R
    never sees R's own ``value_col`` — the core leakage guarantee. Returns a Series
    aligned to ``df``'s index.
    """
    # 1) Sort so each group's rows are in time order (oldest first).
    ordered = df.sort_values([*group_cols, order_col])
    # 2) Group by driver (or team), and look only at the value column we care about.
    grouped = ordered.groupby(group_cols, sort=False)[value_col]
    # 3) The key line. For each group's series `s`:
    #      .shift(1)                  -> move everything down one row = DROP the current race
    #      .rolling(window, ...)      -> take a sliding window of the last `window` races
    #      .agg(agg)                  -> combine them (e.g. average)
    #    min_periods=1 means "give an answer even if there are fewer than `window` past races".
    rolled = grouped.transform(lambda s: s.shift(1).rolling(window, min_periods=1).agg(agg))
    # 4) Put the results back in the ORIGINAL row order of df (we sorted a copy above).
    return rolled.reindex(df.index)


# TOOL 2: a "career-so-far average" — like Tool 1 but using ALL past races, not just N.
def expanding_prior(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    agg: str = "mean",
    order_col: str = "event_date",
) -> pd.Series:
    """Expanding aggregate over all prior rows (career-to-date), excluding current."""
    ordered = df.sort_values([*group_cols, order_col])  # sort in time order
    grouped = ordered.groupby(group_cols, sort=False)[value_col]  # per driver/team
    # Same idea as above, but .expanding() grows to include EVERY past race (career total),
    # while .shift(1) still drops the current race so there is no leakage.
    rolled = grouped.transform(lambda s: s.shift(1).expanding(min_periods=1).agg(agg))
    return rolled.reindex(df.index)  # restore original order
