import pytest

from f1pred.cli import _build_parser


def test_parser_requires_command():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_predict_args():
    args = _build_parser().parse_args(["predict", "--season", "2026", "--round", "11"])
    assert args.command == "predict"
    assert args.season == 2026
    assert args.round == 11


def test_ingest_seasons_optional():
    args = _build_parser().parse_args(["ingest", "--seasons", "2024", "2025"])
    assert args.seasons == [2024, 2025]
