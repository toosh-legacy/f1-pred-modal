# ============================================================================
# features/driver.py  —  CLUES ABOUT EACH DRIVER'S RECENT FORM  (Step 8)
# ----------------------------------------------------------------------------
# Turns raw results into DRIVER clues, e.g. "average finishing position in the
# last 5 races" or "how often they beat their teammate". Every clue is built with
# the anti-cheating tools from leakage.py, so it only uses PAST races.
# ============================================================================

"""Driver-form features. Every feature uses only races *before* the current one.

Input: canonical results (one row per driver per race). Output: same rows plus
``driver_*`` feature columns. All rolling/expanding aggregates are shifted by one race
via ``f1pred.features.leakage`` so the current race never leaks into its own features.
"""

from __future__ import annotations

import pandas as pd

# Import our two safe tools from Step 7.
from f1pred.features.leakage import expanding_prior, shifted_rolling


# Private helper: for each row, did this driver beat their teammate in THAT race?
# (This describes the current race, so add_driver_form below only uses it AFTER
#  shifting it into the past — never for the race being predicted.)
def _beat_teammate(df: pd.DataFrame) -> pd.Series:
    """Per race: 1.0 if the driver out-finished their best teammate, else 0.0.

    Computed on the *current* race outcome; callers must shift it before use as a
    feature (see ``add_driver_form``). Solo entries (no teammate) -> NaN.
    """
    # Group rows by (race, team): the members of one group are teammates in one race.
    grp = df.groupby(["season", "round", "constructor_id"])["position"]
    team_min = grp.transform("min")  # the best (lowest) finishing position in the team
    team_size = grp.transform("count")  # how many cars the team had classified
    # The team's SECOND-best position (so a driver isn't compared against themselves).
    # nsmallest(2).max() = of the two smallest positions, take the larger one.
    team_second = grp.transform(
        lambda s: s.nsmallest(2).max() if s.notna().sum() >= 2 else float("nan")
    )
    # This driver's teammate-to-beat: if the driver IS the team's best, their bar is the
    # team's second-best; otherwise the bar is the team's best.
    teammate_best = df["position"].where(df["position"] != team_min, team_second)
    # beat = 1.0 if this driver finished ahead of that bar, else 0.0.
    beat = (df["position"] < teammate_best).astype(float)
    # If the team had fewer than 2 cars, "beat teammate" is undefined -> NaN (blank).
    beat[team_size < 2] = float("nan")
    return beat


# The main function: adds all the driver clue columns and returns the bigger table.
def add_driver_form(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Add rolling driver-form and career-to-date features."""
    # Sort oldest -> newest so the rolling tools work correctly.
    df = df.sort_values(["event_date", "season", "round"]).reset_index(drop=True)
    # Compute the raw per-race "beat teammate" flag (we'll turn it into a rate below).
    df["_beat_teammate"] = _beat_teammate(df)

    # Loop over each window size (3, 5, 10). f"...{w}" builds column names like
    # "driver_pos_roll5". This creates several clues per window.
    for w in windows:
        # Average finishing position over the last w races.
        df[f"driver_pos_roll{w}"] = shifted_rolling(df, ["driver_id"], "position", w, "mean")
        # Average starting grid position over the last w races.
        df[f"driver_grid_roll{w}"] = shifted_rolling(df, ["driver_id"], "grid", w, "mean")
        # Average points over the last w races.
        df[f"driver_points_roll{w}"] = shifted_rolling(df, ["driver_id"], "points", w, "mean")
        # How often they Did Not Finish over the last w races (0.0-1.0 rate).
        df[f"driver_dnf_rate{w}"] = shifted_rolling(df, ["driver_id"], "dnf", w, "mean")
        # How often they beat their teammate over the last w races.
        df[f"driver_beat_teammate_rate{w}"] = shifted_rolling(
            df, ["driver_id"], "_beat_teammate", w, "mean"
        )

    # Career-to-date clues (use ALL past races, not just a window):
    # How many races the driver had BEFORE this one. cumcount() counts 0,1,2,... per driver.
    df["driver_career_races"] = (
        df.sort_values(["driver_id", "event_date"]).groupby("driver_id").cumcount()
    ).reindex(df.index)
    # Their average finishing position across their whole career so far.
    df["driver_career_avg_pos"] = expanding_prior(df, ["driver_id"], "position", "mean")
    # Their total career points so far.
    df["driver_career_points"] = expanding_prior(df, ["driver_id"], "points", "sum")

    # Drop the temporary helper column (it described the current race; not a safe feature).
    return df.drop(columns=["_beat_teammate"])
