"""Discover Motive CSV sessions under data/{subject_id}/ topology."""

from __future__ import annotations

import copy
import fnmatch
import re
from pathlib import Path
from typing import Any

import pandas as pd

from motive_qc.core import LOGGER, resolve_path
from motive_qc.marker_meta import read_csv_header

SESSION_FILENAME_RE = re.compile(r"^(\d+)_(T\d+_P\d+_R\d+)_", re.IGNORECASE)


def parse_session_from_filename(file_name: str) -> tuple[str | None, str | None, bool]:
    match = SESSION_FILENAME_RE.match(file_name)
    if match:
        return match.group(1), match.group(2), True
    return None, None, False


def _paths_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("paths", {})


def _batch_config(config: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "continue_on_error": True,
        "sort_by": ["subject_id", "session_id"],
        "progress_bar": True,
    }
    return {**defaults, **config.get("batch", {})}


def data_root_from_config(config: dict[str, Any]) -> Path:
    base_dir = Path(config.get("_base_dir", Path(config["_config_path"]).parent))
    root = _paths_config(config).get("data_root", "data")
    return resolve_path(base_dir, root)


def _is_excluded(rel_posix: str, exclude_globs: list[str]) -> bool:
    for pattern in exclude_globs:
        if fnmatch.fnmatch(rel_posix, pattern) or fnmatch.fnmatch(rel_posix, f"**/{pattern}"):
            return True
        if fnmatch.fnmatch(Path(rel_posix).name, pattern):
            return True
    return False


def discover_subjects(config: dict[str, Any]) -> list[str]:
    """Return subject folder names under data_root that contain at least one CSV."""
    data_root = data_root_from_config(config)
    if not data_root.is_dir():
        return []
    exclude = _paths_config(config).get("exclude_globs", ["archive/**"])
    subjects: list[str] = []
    for child in sorted(data_root.iterdir()):
        if not child.is_dir():
            continue
        rel = child.relative_to(data_root).as_posix()
        if _is_excluded(rel, exclude) or _is_excluded(f"{rel}/**", exclude):
            continue
        if any(child.glob("*.csv")):
            subjects.append(child.name)
    return subjects


def discover_sessions(
    config: dict[str, Any],
    subject_ids: list[str] | None = None,
    session_filter: list[str] | None = None,
) -> pd.DataFrame:
    """Scan data/{subject_id}/*.csv and return a catalog DataFrame."""
    data_root = data_root_from_config(config)
    paths_cfg = _paths_config(config)
    exclude_globs = paths_cfg.get("exclude_globs", ["archive/**", "**/*raw test*"])
    include_template = paths_cfg.get("include_globs", ["{subject_id}/*.csv"])

    if subject_ids is None:
        subject_ids = discover_subjects(config)

    rows: list[dict[str, Any]] = []
    for subject_id in subject_ids:
        subject_dir = data_root / subject_id
        if not subject_dir.is_dir():
            LOGGER.warning("Subject folder not found: %s", subject_dir)
            continue
        patterns = [p.replace("{subject_id}", subject_id) for p in include_template]
        csv_files: set[Path] = set()
        for pattern in patterns:
            if "*" in pattern:
                csv_files.update(data_root.glob(pattern))
            else:
                csv_files.update(subject_dir.glob("*.csv"))

        for csv_path in sorted(csv_files):
            if csv_path.suffix.lower() != ".csv":
                continue
            try:
                rel = csv_path.relative_to(data_root).as_posix()
            except ValueError:
                rel = csv_path.name
            if _is_excluded(rel, exclude_globs):
                continue

            file_name = csv_path.name
            parsed_subject, session_id, parse_ok = parse_session_from_filename(file_name)
            subj = parsed_subject or subject_id
            if not parse_ok:
                session_id = re.sub(r"[^\w\-]+", "_", csv_path.stem)[:40]
                LOGGER.warning(
                    "Could not parse session_id from %s; using %s", file_name, session_id
                )
            if session_filter and session_id not in session_filter:
                continue

            size_mb = csv_path.stat().st_size / (1024 * 1024)
            rows.append(
                {
                    "csv_path": str(csv_path.resolve()),
                    "file_name": file_name,
                    "subject_id": subj,
                    "session_id": session_id,
                    "parse_ok": parse_ok,
                    "file_size_mb": round(size_mb, 3),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    sort_by = _batch_config(config).get("sort_by", ["subject_id", "session_id"])
    sort_cols = [c for c in sort_by if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def validate_csv_header(path: Path) -> tuple[bool, str]:
    """Quick header validation before full pipeline run."""
    try:
        if not path.exists():
            return False, f"File not found: {path}"
        if path.stat().st_size == 0:
            return False, "File is empty"
        read_csv_header(path)
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def apply_session_to_config(config: dict[str, Any], session_row: pd.Series | dict) -> dict[str, Any]:
    """Deep-copy config and override paths/project for one discovered session."""
    if isinstance(session_row, pd.Series):
        row = session_row.to_dict()
    else:
        row = session_row

    base_dir = Path(config.get("_base_dir", Path(config["_config_path"]).parent))
    out = copy.deepcopy(config)
    csv_path = Path(row["csv_path"])
    if not csv_path.is_absolute():
        csv_path = (base_dir / csv_path).resolve()

    try:
        rel_input = csv_path.relative_to(base_dir).as_posix()
    except ValueError:
        rel_input = str(csv_path)

    out.setdefault("paths", {})["input_csv"] = rel_input
    out.setdefault("project", {})["subject_id"] = str(row["subject_id"])
    out["project"]["session_id"] = str(row["session_id"])
    out.pop("_run_output_dir", None)
    out["_base_dir"] = str(base_dir)
    return out
