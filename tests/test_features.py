"""Feature-pipeline tests, focused on the leakage guarantee.

The decisive test: shuffle the *outcome* of the final race and confirm the feature
values for every earlier race are unchanged. If any feature peeked at future results,
those values would move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1pred.features.build import build_feature_table, feature_columns
from f1pred.features.track import circuit_key


def _synthetic_results(n_races: int = 6) -> pd.DataFrame:
    """Two teams, four drivers, N races on two alternating circuits."""
    drivers = [
        ("alpha", "team_a"),
        ("bravo", "team_a"),
        ("charlie", "team_b"),
        ("delta", "team_b"),
    ]
    rows = []
    base = pd.Timestamp("2025-03-02")
    for r in range(1, n_races + 1):
        date = base + pd.DateOffset(days=14 * (r - 1))
        circuit = "Alpha Grand Prix" if r % 2 else "Bravo Grand Prix"
        # Deterministic finishing order that rotates each race.
        order = [(i + r) % 4 for i in range(4)]
        for pos, di in enumerate(order, start=1):
            drv, team = drivers[di]
            rows.append(
                {
                    "season": 2025,
                    "round": r,
                    "event_name": circuit,
                    "event_date": date,
                    "driver_id": drv,
                    "driver_name": drv.title(),
                    "constructor_id": team,
                    "grid": pos,
                    "position": float(pos),
                    "points": float(max(0, 26 - pos * 5)),
                    "status": "Finished",
                    "dnf": False,
                }
            )
    return pd.DataFrame(rows)


def test_build_produces_features_and_keys():
    df = build_feature_table(_synthetic_results(), windows=[3])
    feats = feature_columns(df)
    assert feats, "expected some feature columns"
    assert not df.duplicated(["season", "round", "driver_id"]).any()
    # First race for every driver has no prior -> rolling features are NaN.
    first = df[df["round"] == 1]
    assert first["driver_pos_roll3"].isna().all()


def test_no_future_leakage_when_last_race_changes():
    base = _synthetic_results(n_races=6)
    feats_base = build_feature_table(base, windows=[3])

    # Corrupt only the LAST race's outcomes (reverse finishing order + flip DNFs).
    altered = base.copy()
    last = altered["round"] == altered["round"].max()
    altered.loc[last, "position"] = altered.loc[last, "position"].values[::-1]
    altered.loc[last, "dnf"] = True
    feats_alt = build_feature_table(altered, windows=[3])

    # Compare features for all races EXCEPT the last one; they must be identical.
    key = ["season", "round", "driver_id"]
    cols = feature_columns(feats_base)
    a = feats_base[feats_base["round"] < 6].sort_values(key).reset_index(drop=True)
    b = feats_alt[feats_alt["round"] < 6].sort_values(key).reset_index(drop=True)
    for c in cols:
        assert np.allclose(a[c].fillna(-999).to_numpy(), b[c].fillna(-999).to_numpy()), (
            f"feature '{c}' leaked future information from the final race"
        )


def test_circuit_key_normalization():
    assert circuit_key("Australian Grand Prix") == "australian"
    assert circuit_key("São Paulo Grand Prix") == "s_o_paulo"
    assert circuit_key("Miami Grand Prix") == "miami"


def test_reg_flags_zero_outside_2026():
    df = build_feature_table(_synthetic_results(), windows=[3])
    reg_cols = [c for c in df.columns if c.startswith("reg_")]
    assert reg_cols, "expected regulation flag columns"
    # Synthetic data is 2025 -> all regulation flags must be zero.
    assert (df[reg_cols] == 0).all().all()
