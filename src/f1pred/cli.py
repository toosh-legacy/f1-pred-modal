# ============================================================================
# cli.py  —  THE FRONT DESK / SWITCHBOARD  (Step 18, read LAST)
# ----------------------------------------------------------------------------
# CLI = "Command-Line Interface". When you type `uv run f1pred ingest`, THIS file
# reads the word "ingest" and calls the right function (run_ingest). It's the glue
# between your keyboard and the four stations. Read it last, now that you know all
# the stations it points to.
# ============================================================================

"""Command-line entrypoint. Subcommands are wired to pipeline stages.

Usage:
    python -m f1pred.cli ingest [--seasons 2023 2024]
    python -m f1pred.cli features
    python -m f1pred.cli backtest
    python -m f1pred.cli train
    python -m f1pred.cli predict --season 2026 --round 11

Stages import their heavy deps lazily so `--help` stays fast and unit tests can
import this module without a full scientific stack installed.
"""

from __future__ import annotations

import argparse  # Python's built-in tool for reading command-line words/options
from collections.abc import Sequence  # a type hint for "a list of strings"


# Build the "parser" — the thing that understands the words you type after `f1pred`.
def _build_parser() -> argparse.ArgumentParser:
    # prog="f1pred" is the program name shown in help. description=__doc__ reuses the
    # docstring at the top of this file as the help text.
    p = argparse.ArgumentParser(prog="f1pred", description=__doc__)
    # "subparsers" let us have sub-commands like `f1pred ingest`, `f1pred train`, etc.
    # dest="command" stores which one you picked; required=True means you must pick one.
    sub = p.add_subparsers(dest="command", required=True)

    # --- define the `ingest` command and its optional --seasons list ---
    ing = sub.add_parser("ingest", help="Pull FastF1 + Jolpica data to raw parquet.")
    # nargs="*" means --seasons can take zero or more numbers, e.g. --seasons 2024 2025.
    ing.add_argument("--seasons", type=int, nargs="*", default=None)

    # --- commands that take no extra options ---
    sub.add_parser("features", help="Build the leakage-safe feature table.")
    sub.add_parser("backtest", help="Run the rolling-origin backtest.")
    sub.add_parser("train", help="Train the final ranker on all available data.")

    # --- the `predict` command needs a season and round (both required) ---
    pred = sub.add_parser("predict", help="Predict finishing order for one race.")
    pred.add_argument("--season", type=int, required=True)
    pred.add_argument("--round", type=int, required=True)

    return p


# The main function that runs when you use the command line. argv is the list of typed
# words; None means "read them from the real command line".
def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)  # figure out what the user asked for

    # Based on which command was chosen, import + call the matching station function.
    # NOTE: we import INSIDE each branch (not at the top) so that, e.g., running
    # `predict` doesn't load the ingestion code. This keeps startup fast.
    if args.command == "ingest":
        from f1pred.ingest.run import run_ingest

        run_ingest(seasons=args.seasons)  # Station 1
    elif args.command == "features":
        from f1pred.features.build import build_features

        build_features()  # Station 2
    elif args.command == "backtest":
        from f1pred.eval.backtest import run_backtest

        run_backtest()  # Station 3 (test)
    elif args.command == "train":
        from f1pred.models.ranker import train_final

        train_final()  # Station 3 (train)
    elif args.command == "predict":
        from f1pred.predict.infer import predict_race

        predict_race(season=args.season, round=args.round)  # Station 4
    else:  # pragma: no cover - argparse enforces choices
        raise SystemExit(2)  # exit with an error code if somehow no command matched
    return 0  # 0 = success


# This special check means: "only run main() if this file is executed directly."
# It lets `python -m f1pred.cli ...` work. SystemExit passes the return code to the OS.
if __name__ == "__main__":
    raise SystemExit(main())
