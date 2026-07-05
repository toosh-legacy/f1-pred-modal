import numpy as np
import pandas as pd

from f1pred.features.leakage import expanding_prior, shifted_rolling


def _toy():
    # One driver, five races in chronological order, finishing 1..5.
    return pd.DataFrame(
        {
            "driver_id": ["x"] * 5,
            "event_date": pd.to_datetime(
                ["2026-03-01", "2026-03-08", "2026-03-15", "2026-03-22", "2026-03-29"]
            ),
            "position": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


def test_shifted_rolling_excludes_current_row():
    df = _toy()
    roll = shifted_rolling(df, ["driver_id"], "position", window=2, agg="mean")
    # First race has no prior -> NaN. Second sees only race1 (=1). Third sees races 1,2 -> 1.5.
    assert np.isnan(roll.iloc[0])
    assert roll.iloc[1] == 1.0
    assert roll.iloc[2] == 1.5
    assert roll.iloc[3] == 2.5  # mean of races 2,3 = (2+3)/2


def test_no_leakage_feature_never_equals_own_target():
    df = _toy()
    roll = shifted_rolling(df, ["driver_id"], "position", window=3, agg="mean")
    valid = ~roll.isna()
    # A leaking feature would sometimes equal the same-row position; shifted must not.
    assert not (roll[valid].values == df["position"].values[valid]).any()


def test_expanding_prior_is_career_to_date():
    df = _toy()
    exp = expanding_prior(df, ["driver_id"], "position", agg="mean")
    assert np.isnan(exp.iloc[0])
    assert exp.iloc[3] == 2.0  # mean of positions 1,2,3
