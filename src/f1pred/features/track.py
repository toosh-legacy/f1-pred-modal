# ============================================================================
# features/track.py  —  CLUES ABOUT THIS SPECIFIC CIRCUIT  (Step 10)
# ----------------------------------------------------------------------------
# Some drivers/teams are especially strong at certain tracks (e.g. street circuits
# vs. fast circuits). This file builds "how did this driver/team do HERE in the
# past?" clues, using only previous visits to the same circuit.
# ============================================================================

"""Track-context features: how a driver/team has historically gone at this circuit.

A ``circuit_key`` is derived from the event name (stable enough across seasons for
returning venues). Past-at-circuit aggregates use only prior visits (expanding, shifted).
"""

from __future__ import annotations

import re  # "regular expressions" = a mini-language for find/replace on text

import pandas as pd

from f1pred.features.leakage import expanding_prior  # career-so-far tool, reused here per track


# Turn a messy event name into a clean, consistent key so the SAME circuit matches
# across seasons. "Austrian Grand Prix" -> "austrian".
def circuit_key(event_name: str) -> str:
    """Normalize an event name to a circuit key (lowercase, alnum+underscore)."""
    # str(...).lower() -> lowercase text. Then re.sub replaces any run of non-letter/
    # non-digit characters (spaces, punctuation) with a single "_". .strip("_") trims
    # leading/trailing underscores.
    key = re.sub(r"[^a-z0-9]+", "_", str(event_name).lower()).strip("_")
    # Remove a trailing "_grand_prix" so "australian_grand_prix" becomes "australian".
    return re.sub(r"_grand_prix$", "", key)


def add_track_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add driver/constructor past-performance-at-this-circuit features."""
    df = df.sort_values(["event_date", "season", "round"]).reset_index(drop=True)  # time order
    # .map(circuit_key) runs our function on every event name, making a new column.
    df["circuit_key"] = df["event_name"].map(circuit_key)

    # Average finishing position for this DRIVER at this circuit, across past visits only.
    # Notice we group by BOTH driver_id and circuit_key -> "this driver, at this track".
    df["track_driver_avg_pos"] = expanding_prior(
        df, ["driver_id", "circuit_key"], "position", "mean"
    )
    # Same idea for the TEAM at this circuit.
    df["track_constructor_avg_pos"] = expanding_prior(
        df, ["constructor_id", "circuit_key"], "position", "mean"
    )
    return df
