# ============================================================================
# ingest/run.py  —  THE DOWNLOAD LOOP  (Step 6)
# ----------------------------------------------------------------------------
# This is what runs when you type `uv run f1pred ingest`. It loops over every
# season and every race, calls fastf1_loader for each one, stacks all the results
# into a single big table, and saves it as data/raw/results.parquet.
#
# Two smart behaviours:
#   * Incremental: it skips races it already downloaded (re-running is fast).
#   * Safe: it skips races that haven't happened yet, or that error, without crashing.
# ============================================================================

"""Ingestion orchestrator: pull Race results across seasons into data/raw/results.parquet.

Incremental and resumable — rounds already present in the parquet are skipped, and
future/unraced rounds are not attempted. Run via ``f1pred ingest``.
"""

from __future__ import annotations

# datetime = date+time tools. UTC = a standard timezone (so "now" is unambiguous).
from datetime import UTC, datetime

import pandas as pd

from f1pred.config import get_config
from f1pred.ingest.cache import enable_cache  # Step 4: turn on download caching
from f1pred.ingest.fastf1_loader import load_session_results  # Step 5: download one race
from f1pred.logging_utils import get_logger
from f1pred.schema import RESULT_COLUMNS

log = get_logger(__name__)


# Helper: load the existing results file if we have one, else an empty table.
def _existing(results_path) -> pd.DataFrame:
    if results_path.exists():  # is there already a saved file?
        return pd.read_parquet(results_path)  # yes -> load it
    # No file yet -> return an EMPTY table that still has the right columns.
    return pd.DataFrame(columns=RESULT_COLUMNS)


# Helper: figure out which rounds of a season have already been raced (date <= now).
def _raced_rounds(season: int, until: datetime) -> list[tuple[int, str, pd.Timestamp]]:
    """(round, event_name, event_date) for events whose race date is on/before ``until``."""
    import fastf1

    # Ask FastF1 for the season's calendar (all events). Skip pre-season testing.
    sched = fastf1.get_event_schedule(season, include_testing=False)
    rounds: list[tuple[int, str, pd.Timestamp]] = []  # we'll collect (round, name, date) here
    # .iterrows() loops over the calendar one event at a time. "_" = we ignore the row number.
    for _, ev in sched.iterrows():
        rnd = int(ev["RoundNumber"])  # the round number, e.g. 8
        if rnd < 1:  # round 0 is testing/placeholder -> skip
            continue
        date = pd.to_datetime(ev["EventDate"])  # the event's date
        # Timezones can cause comparison errors; strip any timezone so we compare plainly.
        cmp_date = date.tz_localize(None) if date.tzinfo else date
        # Only keep the event if its date is on or before "now" (i.e. it already happened).
        if cmp_date <= pd.Timestamp(until).tz_localize(None):
            rounds.append((rnd, str(ev["EventName"]), date))
    return rounds


# The MAIN function. `seasons=None` means "if not told, use the seasons from config".
def run_ingest(seasons: list[int] | None = None) -> pd.DataFrame:
    """Load Race results for the given seasons (default: config), write parquet, return it."""
    cfg = get_config()  # grab our settings
    cfg.paths.ensure()  # make sure the data folders exist
    enable_cache()  # turn on FastF1's download cache

    seasons = seasons or cfg.seasons  # use given seasons, else the config's list
    now = datetime.now(UTC)  # the current date/time (to know which races have happened)
    out_path = cfg.paths.raw_dir / "results.parquet"  # where we'll save the big table

    existing = _existing(out_path)  # what we've already downloaded before
    # Build a set of (season, round) pairs we already have, for a fast "do we have it?" check.
    have = set(zip(existing["season"], existing["round"], strict=False))
    new_frames: list[pd.DataFrame] = []  # newly downloaded races go here

    # Outer loop: each season. Inner loop: each already-raced round in that season.
    for season in seasons:
        for rnd, name, _date in _raced_rounds(season, now):
            if (season, rnd) in have:  # already downloaded? skip to save time.
                continue
            # "try/except" = attempt something risky; if it fails, handle it instead of crashing.
            try:
                df = load_session_results(season, rnd, "R")  # download the Race results
            except Exception as exc:  # noqa: BLE001 - skip unraced/broken rounds, keep going
                # If a race won't load (not run yet, data missing...), warn and move on.
                log.warning("skip %s R%s (%s): %s", season, rnd, name, exc)
                continue
            log.info("loaded %s R%s %s (%d drivers)", season, rnd, name, len(df))
            new_frames.append(df)  # add this race to our pile

    # Combine the old data + all the newly downloaded races into one table.
    # (We only include `existing` if it actually has rows, to avoid a pandas warning.)
    frames = ([existing] if len(existing) else []) + new_frames
    # pd.concat stacks tables on top of each other. If there's nothing new, keep existing.
    combined = pd.concat(frames, ignore_index=True) if frames else existing
    # Sort nicely: by season, then round, then finishing position.
    combined = combined.sort_values(["season", "round", "position"]).reset_index(drop=True)
    combined.to_parquet(out_path, index=False)  # SAVE the big table to disk
    log.info("wrote %d rows -> %s", len(combined), out_path)
    return combined  # also return it in case the caller wants it directly
