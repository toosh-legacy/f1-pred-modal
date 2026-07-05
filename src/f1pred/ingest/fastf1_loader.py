# ============================================================================
# ingest/fastf1_loader.py  —  DOWNLOAD & CLEAN ONE RACE  (Step 5)
# ----------------------------------------------------------------------------
# FastF1 gives us race results, but with ITS column names (like "GridPosition").
# We want OUR column names (like "grid", from schema.py). This file does that
# translation for a single race.
#
# It is split in two on purpose:
#   * to_canonical(...)        = PURE translation, no internet. Easy to test.
#   * load_session_results(...) = the part that actually downloads, then calls the above.
# Splitting "the risky internet part" from "the pure logic part" is a common,
# very useful pattern.
# ============================================================================

"""Load FastF1 race sessions into the canonical results schema (see f1pred.schema).

The network-touching bit (``load_session_results``) is a thin shell around the pure
``to_canonical`` transform, which is unit-tested without any network access.
"""

from __future__ import annotations

import pandas as pd  # pandas = tables (DataFrames). "pd" is the standard nickname.

from f1pred.logging_utils import get_logger
from f1pred.schema import RESULT_COLUMNS  # the exact columns our output must have

log = get_logger(__name__)

# A "dictionary" mapping FastF1's column name -> our column name.
# Read it as: "rename DriverId to driver_id, FullName to driver_name, ..."
_COLUMN_MAP = {
    "DriverId": "driver_id",
    "FullName": "driver_name",
    "TeamId": "constructor_id",
    "GridPosition": "grid",
    "Position": "position",
    "Points": "points",
    "Status": "status",
}


# A private helper (leading "_") used only inside this file.
def _is_classified(classified_position: pd.Series) -> pd.Series:
    """True where the driver was classified as a finisher (numeric position code).

    FastF1's ``ClassifiedPosition`` is a string: digits for classified finishers,
    letters for retirements/DSQ/DNS ('R', 'D', 'W', 'E', 'N', ...).
    """
    # A pd.Series is ONE column of a table. Here's the chain, left to right:
    #   .astype(str)          -> make sure every value is text
    #   .str.fullmatch(r"\d+")-> True if the text is ALL digits (e.g. "1", "12")
    #   .fillna(False)        -> if the check gave a blank, treat it as False
    # Result: a column of True/False, True = "this driver finished and was classified".
    return classified_position.astype(str).str.fullmatch(r"\d+").fillna(False)


# The PURE translation function. The "*" in the arguments means everything after it
# must be passed by name (season=..., round=...), which prevents mix-ups.
def to_canonical(
    results: pd.DataFrame,  # FastF1's results table for one race
    *,
    season: int,  # e.g. 2026
    round: int,  # e.g. 8
    event_name: str,  # e.g. "Austrian Grand Prix"
    event_date: pd.Timestamp,  # the race date
) -> pd.DataFrame:
    """Transform a FastF1 ``session.results`` frame into the canonical results schema.

    Pure function: no I/O. ``results`` must expose the FastF1 result columns plus
    ``ClassifiedPosition``.
    """
    # .rename() applies our column-name map. .copy() makes a separate copy so we
    # don't accidentally change FastF1's original table.
    df = results.rename(columns=_COLUMN_MAP).copy()
    # Add the race-identifying columns. The same value is written to every row.
    df["season"] = int(season)
    df["round"] = int(round)
    df["event_name"] = str(event_name)
    df["event_date"] = pd.to_datetime(event_date)  # ensure it's a proper date type

    # Create our "dnf" (Did Not Finish) column. "~" means NOT, so:
    #   dnf = NOT classified  ->  True when the driver retired/crashed/was disqualified.
    df["dnf"] = ~_is_classified(results["ClassifiedPosition"])

    # Make sure these three columns are real numbers. errors="coerce" turns anything
    # unparseable into "NaN" (Not a Number = a blank) instead of crashing.
    for col in ("grid", "position", "points"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # A pit-lane / no-grid start comes through as 0.0; keep it, it is meaningful.

    # Keep ONLY our agreed columns, in the agreed order. reset_index makes the row
    # numbers a clean 0,1,2,... again.
    out = df[RESULT_COLUMNS].reset_index(drop=True)
    return out


# The function that actually touches the internet, then reuses to_canonical above.
def load_session_results(
    season: int,
    round: int,
    session_code: str = "R",  # "R" = Race (default), "Q" = Qualifying
) -> pd.DataFrame:
    """Load one session's results from FastF1 into canonical schema (hits network/cache)."""
    import fastf1  # heavy import, kept inside the function

    # Ask FastF1 for this specific session (year, round number, session type).
    session = fastf1.get_session(season, round, session_code)
    # Actually download it. We turn OFF the heavy extras (laps/telemetry/weather/messages)
    # because we only need the final results here — this makes it much faster.
    session.load(laps=False, telemetry=False, weather=False, messages=False)
    event = session.event  # info about the event (name, date, etc.)
    # Hand FastF1's results table to our pure translator and return the clean result.
    return to_canonical(
        session.results,
        season=season,
        round=round,
        event_name=str(event["EventName"]),
        event_date=pd.to_datetime(event["EventDate"]),
    )
