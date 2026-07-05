# ============================================================================
# logging_utils.py  —  TIDY STATUS MESSAGES  (Reading-guide Step 2)
# ----------------------------------------------------------------------------
# "Logging" just means printing progress messages while the program runs, e.g.
#   12:00:05 INFO  f1pred.ingest.run | loaded 2026 R8 (22 drivers)
# We use logging instead of plain print() because it AUTOMATICALLY adds the time
# and which file the message came from. This file hands every other file a ready
# "log" object to use.
# ============================================================================

"""Small logging helper so every module logs consistently."""

from __future__ import annotations

import logging  # Python's built-in logging system

# A module-level flag. We only want to set up logging ONCE for the whole program.
# Starts False, becomes True after the first setup.
_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    # "global" lets us change the module-level _CONFIGURED variable from inside here.
    global _CONFIGURED
    if not _CONFIGURED:  # only true the very first time this runs
        # basicConfig sets the global style for ALL log messages.
        logging.basicConfig(
            level=logging.INFO,  # show INFO and above (hide low-level DEBUG spam)
            # The format string: time, level (INFO/WARNING), file name, then the message.
            format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
            datefmt="%H:%M:%S",  # show time as Hours:Minutes:Seconds
        )
        _CONFIGURED = True  # remember we've done setup so we skip it next time
    # Return a logger tagged with the caller's name (usually the file's name).
    # Other files do:  log = get_logger(__name__)   then   log.info("hello")
    return logging.getLogger(name)
