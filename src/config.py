"""Central configuration: load config.yaml + .env, resolve paths, seed RNGs.

Secrets are read only from the environment (.env via python-dotenv) and are
never written to disk or logged.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

# Project root = the directory that contains this `src/` package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _resolve(path_str: str) -> Path:
    """Resolve a config path relative to the project root (abs paths pass through)."""
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


class Config(dict):
    """Thin dict wrapper exposing resolved paths and the project root."""

    def __init__(self, data: dict, path: Path):
        super().__init__(data)
        self.path = path
        self.root = PROJECT_ROOT

    # ── resolved paths ──────────────────────────────────────────────────
    @property
    def db_path(self) -> Path:
        return _resolve(self["paths"]["database"])

    @property
    def data_raw(self) -> Path:
        return _resolve(self["paths"]["data_raw"])

    @property
    def data_processed(self) -> Path:
        return _resolve(self["paths"]["data_processed"])

    @property
    def reports_dir(self) -> Path:
        return _resolve(self["paths"]["reports"])

    @property
    def figures_dir(self) -> Path:
        return _resolve(self["paths"]["figures"])

    def ensure_dirs(self) -> None:
        for p in (self.data_raw, self.data_processed, self.reports_dir, self.figures_dir):
            p.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load .env + config.yaml, create working dirs, and seed RNGs."""
    load_dotenv(PROJECT_ROOT / ".env")  # load secrets if a .env exists
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    cfg = Config(data, cfg_path)
    cfg.ensure_dirs()
    seed_everything(int(data.get("seed", 42)))
    return cfg


def seed_everything(seed: int = 42) -> None:
    """Fix all RNG seeds we control for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_api_key(name: str, required: bool = False) -> str | None:
    """Fetch an API key from the environment. Raises if required and missing."""
    val = os.getenv(name)
    if required and not val:
        raise RuntimeError(
            f"Missing required API key '{name}'. Copy .env.example to .env and set it. "
            f"Secrets must never be hard-coded or committed."
        )
    return val


def save_config(cfg: Config) -> None:
    """Persist the config back to disk (used to record the discovered actor id)."""
    plain = {k: v for k, v in cfg.items()}
    with open(cfg.path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(plain, fh, sort_keys=False, default_flow_style=False, allow_unicode=True)
