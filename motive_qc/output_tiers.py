"""Output tier definitions and run directory resolution."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from motive_qc.core import resolve_path

ESSENTIAL_TABLES = frozenset(
    {
        "session_summary",
        "gap_events",
        "window_quality_summary",
        "window_quality_0p5s",
        "artifact_events",
        "artifact_session_summary",
        "qc_intervals",
        "analysis_mask_summary",
    }
)

ESSENTIAL_PLOTS = frozenset(
    {
        "gap_timeline",
        "window_quality_timeline",
        "artifact_timeline",
        "artifact_velocity_histogram",
    }
)


def resolve_run_output_dir(config: dict[str, Any]) -> Path:
    """Resolve per-run output directory; sets config['_run_output_dir']."""
    if config.get("_run_output_dir"):
        return Path(config["_run_output_dir"])

    base_dir = Path(config.get("_base_dir", Path(config["_config_path"]).parent))
    root = resolve_path(base_dir, config["paths"]["output_dir"])
    paths_cfg = config.get("paths", {})

    if paths_cfg.get("use_timestamp_subfolder", True):
        session_id = str(config.get("project", {}).get("session_id", "session"))
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = root / f"{safe_id}_{ts}"
    else:
        run_dir = root

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tables").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    config["_run_output_dir"] = str(run_dir.resolve())
    return run_dir


def should_write_table(name: str, config: dict[str, Any]) -> bool:
    tier = config.get("outputs", {}).get("tier", "essential")
    if name == "artifact_candidates":
        return bool(config.get("outputs", {}).get("write_frame_level_artifacts", False))
    if tier == "full":
        return True
    return name in ESSENTIAL_TABLES


def should_write_plot(name: str, config: dict[str, Any]) -> bool:
    tier = config.get("outputs", {}).get("tier", "essential")
    if tier == "full":
        return True
    if name in ESSENTIAL_PLOTS:
        return True
    return name.startswith("artifact_velocity_histogram__")
