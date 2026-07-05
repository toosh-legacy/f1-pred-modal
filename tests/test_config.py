from pathlib import Path

from f1pred.config import Config, load_config


def test_load_config_defaults():
    cfg = load_config()
    assert cfg.target_season == 2026
    assert 2026 in cfg.seasons
    # Paths get resolved to absolute under the repo root.
    assert cfg.paths.raw_dir.is_absolute()
    assert cfg.paths.raw_dir.name == "raw"


def test_env_override(monkeypatch):
    monkeypatch.setenv("F1PRED_TARGET_SEASON", "2027")
    cfg = load_config()
    assert cfg.target_season == 2027


def test_paths_resolve_relative_to_root():
    p = Config().paths.resolve(Path("/tmp/root"))
    assert p.raw_dir == Path("/tmp/root/data/raw")
