# ============================================================================
# schema.py  —  THE AGREED COLUMN NAMES  (Reading-guide Step 3)
# ----------------------------------------------------------------------------
# A "schema" is the agreed shape of our data table: which columns exist and what
# they mean. Every file imports these names instead of typing "position" or
# "driver_id" by hand, so if we ever rename a column we change it in ONE place.
#
# Remember: our main table has ONE ROW PER DRIVER PER RACE.
# ============================================================================

"""Canonical column names for the ingested race-results table.

Keeping these in one place lets ingestion, features, and schema tests agree on a
contract. The raw results parquet (one row per driver per race) must contain at
least RESULT_COLUMNS; feature building adds columns on top.
"""

from __future__ import annotations

# These two columns together identify WHICH race a row belongs to.
# (season 2026, round 8) = the 2026 Austrian Grand Prix, for example.
KEY_COLUMNS = ["season", "round"]

# These identify the driver and their team in that row.
ENTITY_COLUMNS = ["driver_id", "constructor_id"]

# The full list of columns our cleaned results table must have, with a note on each.
# The "# comment" after each string just explains that column — it's not code.
RESULT_COLUMNS = [
    "season",  # int, e.g. 2026
    "round",  # int, 1-based within season (round 1 = first race of the year)
    "event_name",  # str, e.g. "Austrian Grand Prix"
    "event_date",  # datetime (the race date) — used to sort races in time order
    "driver_id",  # str, a stable code for the driver, e.g. "max_verstappen"
    "driver_name",  # str, human-readable name
    "constructor_id",  # str, a stable code for the team, e.g. "red_bull"
    "grid",  # number, starting position (0 = started from the pit lane)
    "position",  # number, finishing position (this is what we try to predict!)
    "points",  # number, championship points scored
    "status",  # str, e.g. "Finished", "+1 Lap", "Accident"
    "dnf",  # True/False, "Did Not Finish" — we calculate this ourselves
]

# TARGET_COLUMN = the thing the model learns to predict = the finishing position.
TARGET_COLUMN = "position"

# GROUP_KEY = what defines one "group" for ranking. In learning-to-rank we rank
# drivers WITHIN a race, so one race (season + round) is one group.
GROUP_KEY = KEY_COLUMNS  # one ranking group == one race
