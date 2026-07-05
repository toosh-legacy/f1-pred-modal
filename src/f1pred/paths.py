"""Project paths, resolved relative to the repo root so notebooks work from anywhere."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
REFERENCE = DATA / "reference"
RESULTS = ROOT / "results"


def ensure_dirs() -> None:
    """Create the writable output dirs if they don't exist yet."""
    for d in (RAW, RESULTS):
        d.mkdir(parents=True, exist_ok=True)
