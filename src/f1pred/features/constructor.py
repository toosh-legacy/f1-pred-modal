# ============================================================================
# features/constructor.py  —  CLUES ABOUT EACH TEAM'S CAR  (Step 9)
# ----------------------------------------------------------------------------
# A driver is only as good as their car. This file builds TEAM clues: recent
# points (car pace) and how often the car breaks down (reliability). We compute
# these once per team-per-race, then copy the value onto BOTH of that team's
# drivers (a fast car helps both drivers equally).
# ============================================================================

"""Constructor (car) pace & reliability features — leakage-safe, prior races only.

Constructor form is computed at the *team-race* grain (one value per team per race,
built from both cars) then broadcast back to each driver row, so a strong car lifts
both its drivers equally.
"""

from __future__ import annotations

import pandas as pd

from f1pred.features.leakage import shifted_rolling  # our anti-cheating rolling tool


def add_constructor_form(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Add rolling constructor pace/reliability features."""
    # Sort oldest -> newest for the rolling tool.
    df = df.sort_values(["event_date", "season", "round"]).reset_index(drop=True)

    # STEP 1: Shrink the driver-level table down to ONE ROW PER TEAM PER RACE.
    # .groupby(...).agg(...) means "for each group, summarize these columns":
    #   team_points  = SUM of both drivers' points (total the team scored that race)
    #   team_best_pos = the BEST (minimum) finishing position of the team's cars
    #   team_dnf_rate = the AVERAGE dnf flag = fraction of the team's cars that broke
    team_race = (
        df.groupby(["season", "round", "event_date", "constructor_id"], as_index=False)
        .agg(
            team_points=("points", "sum"),
            team_best_pos=("position", "min"),
            team_dnf_rate=("dnf", "mean"),
        )
        .sort_values(["event_date", "season", "round"])
        .reset_index(drop=True)
    )

    # STEP 2: On that team-per-race table, build rolling "recent car form" clues.
    for w in windows:  # for each window 3, 5, 10
        # Team's average total points over the last w races (car speed).
        team_race[f"constructor_points_roll{w}"] = shifted_rolling(
            team_race, ["constructor_id"], "team_points", w, "mean"
        )
        # Team's average best-finish over the last w races.
        team_race[f"constructor_bestpos_roll{w}"] = shifted_rolling(
            team_race, ["constructor_id"], "team_best_pos", w, "mean"
        )
        # Team's average breakdown rate over the last w races (reliability).
        team_race[f"constructor_dnf_rate{w}"] = shifted_rolling(
            team_race, ["constructor_id"], "team_dnf_rate", w, "mean"
        )

    # STEP 3: Figure out which columns are the new team CLUES to copy back.
    # We keep columns that start with "constructor_" BUT NOT the id column itself
    # (constructor_id also starts with "constructor_", and we don't want to copy the key).
    feature_cols = [
        c for c in team_race.columns if c.startswith("constructor_") and c != "constructor_id"
    ]
    # STEP 4: .merge() glues the team clues back onto the original driver rows,
    # matching on (season, round, constructor_id). how="left" keeps every driver row.
    merged = df.merge(
        team_race[["season", "round", "constructor_id", *feature_cols]],
        on=["season", "round", "constructor_id"],
        how="left",
    )
    return merged
