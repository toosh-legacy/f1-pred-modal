import pandas as pd
import pytest

from f1pred.ingest.fastf1_loader import to_canonical
from f1pred.schema import RESULT_COLUMNS


def _fake_fastf1_results():
    """Minimal stand-in for a FastF1 session.results frame."""
    return pd.DataFrame(
        {
            "DriverId": ["russell", "verstappen", "leclerc"],
            "FullName": ["George Russell", "Max Verstappen", "Charles Leclerc"],
            "TeamId": ["mercedes", "red_bull", "ferrari"],
            "GridPosition": [1.0, 20.0, 4.0],
            "Position": [1.0, 6.0, 3.0],
            "ClassifiedPosition": ["1", "6", "R"],  # Leclerc retired
            "Points": [25.0, 8.0, 0.0],
            "Status": ["Finished", "Finished", "Accident"],
        }
    )


def test_to_canonical_schema_and_types():
    out = to_canonical(
        _fake_fastf1_results(),
        season=2026,
        round=1,
        event_name="Australian Grand Prix",
        event_date=pd.Timestamp("2026-03-08"),
    )
    # Exact canonical schema, in order.
    assert list(out.columns) == RESULT_COLUMNS
    assert len(out) == 3
    assert (out["season"] == 2026).all()
    assert out["event_name"].iloc[0] == "Australian Grand Prix"
    assert pd.api.types.is_datetime64_any_dtype(out["event_date"])


def test_dnf_derived_from_classified_position():
    out = to_canonical(
        _fake_fastf1_results(),
        season=2026,
        round=1,
        event_name="X",
        event_date=pd.Timestamp("2026-03-08"),
    )
    dnf = {d: bool(v) for d, v in zip(out["driver_id"], out["dnf"], strict=False)}
    assert dnf["russell"] is False
    assert dnf["verstappen"] is False  # classified P6
    assert dnf["leclerc"] is True  # 'R' = retired


def test_no_duplicate_driver_per_race():
    out = to_canonical(
        _fake_fastf1_results(),
        season=2026,
        round=1,
        event_name="X",
        event_date=pd.Timestamp("2026-03-08"),
    )
    assert not out.duplicated(["season", "round", "driver_id"]).any()


@pytest.mark.network
def test_load_real_session_smoke():
    from f1pred.ingest.cache import enable_cache
    from f1pred.ingest.fastf1_loader import load_session_results

    enable_cache()
    df = load_session_results(2024, 1, "R")
    assert list(df.columns) == RESULT_COLUMNS
    assert len(df) == 20
    winner = df.sort_values("position").iloc[0]
    assert "verstappen" in winner["driver_id"]  # 2024 Bahrain winner
