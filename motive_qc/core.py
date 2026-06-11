"""Core types, config, and shared constants."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr
import yaml

__version__ = "0.5.0"

LOGGER = logging.getLogger("motive_raw_qc")

CSV_TABLE_EXCLUDE = frozenset({"gap_summary_by_marker"})

REQUIRED_CONFIG_SECTIONS = [
    "project",
    "paths",
    "time",
    "parsing",
    "markers",
    "marker_groups",
    "gaps",
    "quality_labels",
    "outputs",
    "reporting",
]

SEVERITY_ORDER = [
    ("single_frame", None),
    ("tiny", "tiny_gap"),
    ("minor", "minor_gap"),
    ("moderate", "moderate_gap"),
    ("large", "large_gap"),
    ("severe", "severe_gap"),
]


class ConfigValidationError(Exception):
    """Invalid or incomplete configuration."""


class MotiveCSVParseError(Exception):
    """CSV cannot be parsed safely."""


class SchemaValidationError(Exception):
    """Expected columns or axes are missing."""


class QCValidationError(Exception):
    """Scientific validation failed."""


@dataclass
class QCMessage:
    severity: str
    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)
    suggested_action: str | None = None


@dataclass
class QCResult:
    layer_name: str
    status: str
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: dict[str, Path] = field(default_factory=dict)
    files_written: list[Path] = field(default_factory=list)
    messages: list[QCMessage] = field(default_factory=list)
    exception: Exception | None = None
    session: "MotiveSession | None" = None


@dataclass
class MotiveSession:
    metadata: dict[str, Any]
    frames: pd.Index
    time_seconds: pd.Series
    coordinates: xr.DataArray
    valid_marker_frame: xr.DataArray
    marker_inventory: pd.DataFrame
    validation_messages: list[QCMessage] = field(default_factory=list)


def deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigValidationError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ConfigValidationError("Config root must be a mapping.")

    missing = [section for section in REQUIRED_CONFIG_SECTIONS if section not in config]
    if missing:
        raise ConfigValidationError(f"Missing required config sections: {', '.join(missing)}")

    if overrides:
        config = deep_merge(config, overrides)

    validate_config(config)
    config["_config_path"] = str(config_path.resolve())
    return config


def validate_config(config: dict[str, Any]) -> None:
    paths = config["paths"]
    if not paths.get("input_csv"):
        raise ConfigValidationError("paths.input_csv is required.")
    if not paths.get("output_dir"):
        raise ConfigValidationError("paths.output_dir is required.")

    gaps = config["gaps"]
    if not gaps.get("thresholds_seconds"):
        raise ConfigValidationError("gaps.thresholds_seconds is required.")

    time_cfg = config["time"]
    if time_cfg.get("frame_rate_hz_override") is None and not time_cfg.get(
        "infer_frame_rate_from_file", True
    ):
        raise ConfigValidationError(
            "Either time.infer_frame_rate_from_file must be true or "
            "time.frame_rate_hz_override must be set."
        )


def resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def base_dir_from_config(config: dict[str, Any]) -> Path:
    return Path(config.get("_base_dir", Path(config["_config_path"]).parent))


def output_dir_from_config(config: dict[str, Any]) -> Path:
    return resolve_path(base_dir_from_config(config), config["paths"]["output_dir"])


def plot_dir_from_config(config: dict[str, Any]) -> Path:
    from motive_qc.output_tiers import resolve_run_output_dir

    run_dir = resolve_run_output_dir(config)
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def stop_after_layer(config: dict[str, Any]) -> int:
    return int(config.get("reporting", {}).get("stop_after_layer", 2))
