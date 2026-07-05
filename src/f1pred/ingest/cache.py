# ============================================================================
# ingest/cache.py  —  "SAVE DOWNLOADS SO WE DON'T REPEAT THEM"  (Step 4)
# ----------------------------------------------------------------------------
# FastF1 downloads race data from the internet. Downloading is slow, so FastF1
# can save ("cache") each download to a folder and reuse it next time. This tiny
# file just tells FastF1 which folder to use. Call enable_cache() ONCE before any
# FastF1 download.
# ============================================================================

"""FastF1 disk-cache setup. Call ``enable_cache()`` before any FastF1 access."""

from __future__ import annotations

from pathlib import Path  # smart file/folder paths

from f1pred.config import get_config  # to find the cache folder from our settings
from f1pred.logging_utils import get_logger  # for a status message

# Every file makes its own logger tagged with its name (__name__ = "f1pred.ingest.cache").
log = get_logger(__name__)


def enable_cache(cache_dir: Path | None = None) -> Path:
    """Enable FastF1's on-disk cache (idempotent). Returns the cache directory."""
    # We import fastf1 INSIDE the function (not at the top) because it's a heavy
    # library. Importing it only when needed keeps the rest of the code fast to load.
    import fastf1

    # If the caller didn't pass a folder, use the one from our settings (config.py).
    # "a or b" means: use a if it's given, otherwise fall back to b.
    cache_dir = cache_dir or get_config().paths.fastf1_cache
    cache_dir = Path(cache_dir)  # make sure it's a Path object
    # Create the folder (and any missing parents). Harmless if it already exists.
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Tell FastF1 to use this folder for all its caching. (It wants a string, not a Path.)
    fastf1.Cache.enable_cache(str(cache_dir))
    log.info("FastF1 cache enabled at %s", cache_dir)  # print where the cache is
    return cache_dir  # hand the folder back in case the caller wants it
