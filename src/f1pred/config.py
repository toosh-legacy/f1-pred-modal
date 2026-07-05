# ============================================================================
# config.py  —  THE SETTINGS DESK  (Reading-guide Step 1)
# ----------------------------------------------------------------------------
# Every other file in the project starts by asking THIS file "what are the
# settings?". Settings live in one place (the file config.yaml) so we never
# scatter magic numbers like "2026" or file paths all over the code.
#
# This file's job: read config.yaml -> turn it into a tidy Python object -> hand
# that object to whoever asks via get_config().
# ============================================================================

"""Typed configuration loaded from config.yaml with env-var overrides.

Precedence (low -> high): defaults in this file < config.yaml < F1PRED_* env vars.
Load once via ``get_config()``; it is cached for the process.
"""

# This special import makes type hints (the ": Path" bits) simpler. Always at the
# top of our files. You can safely ignore it as a beginner.
from __future__ import annotations

# --- Import tools we need from Python's standard library and from libraries ---
from functools import lru_cache  # lru_cache = "remember the answer so we don't redo work"
from pathlib import Path  # Path = a smart way to handle file/folder locations
from typing import ClassVar  # used to mark a variable as "belongs to the class, not a setting"

# pydantic is a library that checks data has the right types (e.g. seasons must be
# a list of integers). It turns messy config into safe, validated Python objects.
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,  # a special model that can also read environment variables
    PydanticBaseSettingsSource,  # a "source" of settings (used below)\
    SettingsConfigDict,  # holds options for how settings are read
    YamlConfigSettingsSource,  # lets pydantic read our config.yaml file
)

# __file__ is THIS file's location. .resolve() makes it a full absolute path.
# .parents[2] walks up 3 folders:  config.py -> f1pred -> src -> (repo root).
# So REPO_ROOT is the top folder of the whole project.
REPO_ROOT = Path(__file__).resolve().parents[2]
# The config.yaml sits right in that top folder.
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"  # the "/" joins folder + filename


# A "class" is a blueprint. This blueprint groups all our folder locations together.
# BaseModel (from pydantic) means "validate the types of these fields."
class Paths(BaseModel):
    # Each line below is a setting with a DEFAULT value. If config.yaml doesn't
    # mention it, this default is used. The ": Path" says the value is a folder path.
    data_root: Path = Path("data")  # top data folder
    raw_dir: Path = Path("data/raw")  # Station 1 saves downloaded results here
    features_dir: Path = Path("data/features")  # Station 2 saves features here
    fastf1_cache: Path = Path("data/cache/fastf1")  # FastF1's download cache
    models_dir: Path = Path("models")  # the trained model is saved here
    predictions_dir: Path = Path("data/predictions")  # predictions + backtest scores

    # A "method" is a function that belongs to a class. This one turns relative
    # paths like "data/raw" into full paths like "C:/.../f1-pred-modal/data/raw".
    def resolve(self, root: Path) -> Paths:
        """Make every path absolute relative to the repo root."""
        # self.model_dump() gives a dict of {name: value} for every field above.
        # For each one: if it's not already absolute, glue the repo root in front.
        return Paths(
            **{
                name: (root / value) if not value.is_absolute() else value
                for name, value in self.model_dump().items()
            }
        )

    # Create the folders on disk if they don't exist yet. Called before we save files.
    def ensure(self) -> None:
        for value in self.model_dump().values():  # loop over every folder path
            # parents=True: also create missing parent folders.
            # exist_ok=True: don't error if it already exists.
            Path(value).mkdir(parents=True, exist_ok=True)


# The next few small classes each group one section of config.yaml.


class IngestConfig(BaseModel):  # settings for Station 1 (downloading)
    sessions: list[str] = ["Q", "R"]  # which sessions to load: Q=Qualifying, R=Race
    load_telemetry: bool = False  # whether to download heavy lap-by-lap data (off by default)
    jolpica_base_url: str = "https://api.jolpi.ca/ergast/f1"  # a backup historical data API


class FeaturesConfig(BaseModel):  # settings for Station 2 (feature building)
    # "windows" = how many past races to average over. 3, 5, and 10 give short,
    # medium, and long-term form.
    form_windows: list[int] = [3, 5, 10]


class ModelConfig(BaseModel):  # settings for the model
    seed: int = 42  # a fixed "random seed" so results are repeatable each run
    params: dict = Field(default_factory=dict)  # the LightGBM knobs (filled from config.yaml)
    eval_at: list[int] = [1, 3, 10]  # measure quality at top-1, top-3, top-10


class BacktestConfig(BaseModel):  # settings for the fair test
    min_train_season: int = 2019  # earliest season used for training
    eval_seasons: list[int] = [2023, 2024, 2025, 2026]  # seasons we test the model on


class TrackingConfig(BaseModel):  # settings for experiment tracking (MLflow)
    mlflow_uri: str = "file:./mlruns"  # where MLflow saves run history
    experiment: str = "f1pred"  # the name of our experiment group


# This is the MAIN settings object that bundles ALL the little ones above.
# BaseSettings (not just BaseModel) means it can ALSO read environment variables.
class Config(BaseSettings):
    """Root config.

    Precedence (high -> low): constructor kwargs > F1PRED_ env vars > config.yaml >
    field defaults. The YAML source sits *below* env so an env var always wins, which
    is what deployment expects.
    """

    # model_config here configures HOW settings are read (it is not an F1 setting!).
    model_config = SettingsConfigDict(
        env_prefix="F1PRED_",  # env vars must start with F1PRED_ to count
        env_nested_delimiter="__",  # F1PRED_MODEL__SEED sets model.seed
        extra="ignore",  # ignore unknown keys instead of crashing
    )

    # ClassVar means "this belongs to the class itself, it is NOT a setting field."
    # load_config() sets this to tell the YAML reader which file to open.
    yaml_path: ClassVar[Path] = DEFAULT_CONFIG_PATH

    # The actual top-level settings. Each has a default; config.yaml usually overrides them.
    seasons: list[int] = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
    target_season: int = 2026  # the season we ultimately want to predict
    paths: Paths = Paths()  # the folder locations (the class above)
    ingest: IngestConfig = IngestConfig()  # nesting: config.ingest.sessions, etc.
    features: FeaturesConfig = FeaturesConfig()
    model: ModelConfig = ModelConfig()
    backtest: BacktestConfig = BacktestConfig()
    tracking: TrackingConfig = TrackingConfig()

    # This method tells pydantic the ORDER in which to look for each setting.
    # Earlier in the returned tuple = higher priority (wins ties).
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,  # values passed directly in code
        env_settings: PydanticBaseSettingsSource,  # values from F1PRED_ env vars
        dotenv_settings: PydanticBaseSettingsSource,  # values from a .env file
        file_secret_settings: PydanticBaseSettingsSource,  # values from secret files
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Build the source that reads our config.yaml file.
        yaml_source = YamlConfigSettingsSource(settings_cls, yaml_file=cls.yaml_path)
        # Order = priority: code args first, then env vars, then YAML, then secrets.
        # So an env var like F1PRED_TARGET_SEASON=2027 beats the value in config.yaml.
        return (init_settings, env_settings, yaml_source, file_secret_settings)


# A plain function (not inside a class) that builds a ready-to-use Config object.
def load_config(path: Path | str | None = None) -> Config:
    """Load config from ``path`` (default config.yaml) with env overrides, resolve paths."""
    # Tell the class which YAML file to read (default = the main config.yaml).
    Config.yaml_path = Path(path) if path else DEFAULT_CONFIG_PATH
    cfg = Config()  # <- this line actually reads YAML + env vars and validates everything
    # Turn "data/raw" style paths into full absolute paths under the repo root.
    cfg.paths = cfg.paths.resolve(REPO_ROOT)
    return cfg  # hand back the finished settings object


# @lru_cache(maxsize=1) means "run this once, then remember the result forever."
# So every file that calls get_config() shares the SAME settings object (fast + consistent).
@lru_cache(maxsize=1)
def get_config() -> Config:
    return load_config()
