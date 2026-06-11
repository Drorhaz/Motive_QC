#!/usr/bin/env python3
"""Motive raw marker QC pipeline — Layers 1-2 only."""

from __future__ import annotations

import argparse
import copy
import csv
import logging
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml

__version__ = "0.2.0"

LOGGER = logging.getLogger("motive_raw_qc")

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


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ConfigValidationError(Exception):
  """Invalid or incomplete configuration."""


class MotiveCSVParseError(Exception):
  """CSV cannot be parsed safely."""


class SchemaValidationError(Exception):
  """Expected columns or axes are missing."""


class QCValidationError(Exception):
  """Scientific validation failed."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
  merged = copy.deepcopy(base)
  for key, value in override.items():
    if isinstance(value, dict) and isinstance(merged.get(key), dict):
      merged[key] = _deep_merge(merged[key], value)
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
    config = _deep_merge(config, overrides)

  _validate_config(config)
  config["_config_path"] = str(config_path.resolve())
  return config


def _validate_config(config: dict[str, Any]) -> None:
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


def _resolve_path(base_dir: Path, value: str | Path) -> Path:
  path = Path(value)
  if not path.is_absolute():
    path = (base_dir / path).resolve()
  return path


# ---------------------------------------------------------------------------
# Metadata and header parsing
# ---------------------------------------------------------------------------


def _parse_metadata_row(row: list[str]) -> dict[str, str]:
  metadata: dict[str, str] = {}
  idx = 0
  while idx < len(row) - 1:
    key = row[idx].strip()
    value = row[idx + 1].strip() if idx + 1 < len(row) else ""
    if key:
      metadata[key] = value
      idx += 2
    else:
      idx += 1
  return metadata


def _find_header_rows(rows: list[list[str]]) -> dict[str, int]:
  labels = {
    "type": "Type",
    "name": "Name",
    "id": "ID",
    "parent": "Parent",
    "axis": "Frame",
  }
  found: dict[str, int] = {}
  for row_idx, row in enumerate(rows[:20]):
    if not row:
      continue
    label = _header_row_label(row)
    for key, expected in labels.items():
      if label == expected and key not in found:
        found[key] = row_idx
    if label == "Frame" and "axis" not in found:
      found["axis"] = row_idx
    if "channel" not in found and any(cell.strip() == "Position" for cell in row[2:6]):
      found["channel"] = row_idx
  required = ["type", "name", "axis"]
  missing = [key for key in required if key not in found]
  if missing:
    raise MotiveCSVParseError(
      f"Could not locate required header rows: {', '.join(missing)}"
    )
  return found


def _is_unlabeled_marker(name: str) -> bool:
  return bool(re.match(r"(?i)^unlabeled(\s+\d+)?$", name.strip()))


def _parse_subject_prefix(marker_name: str) -> tuple[str | None, str]:
  if ":" in marker_name:
    prefix, short = marker_name.split(":", 1)
    return prefix, short
  return None, marker_name


def _assign_body_region(marker_name: str, marker_groups: dict[str, Any]) -> str:
  short_name = _parse_subject_prefix(marker_name)[1]
  if _is_unlabeled_marker(marker_name):
    return "unlabeled"
  for group_name, group_cfg in marker_groups.items():
    if group_name == "unclassified":
      continue
    keywords = group_cfg.get("keywords", [])
    for keyword in keywords:
      if keyword and keyword.lower() in short_name.lower():
        return group_name
  return "unclassified"


def _header_row_label(row: list[str]) -> str:
  if not row:
    return ""
  if row[0].strip():
    return row[0].strip()
  if len(row) > 1:
    return row[1].strip()
  return ""


def _read_csv_header(path: Path) -> tuple[dict[str, str], list[list[str]], int]:
  with path.open("r", encoding="utf-8-sig", newline="") as handle:
    reader = csv.reader(handle)
    metadata_row = next(reader)
    metadata = _parse_metadata_row(metadata_row)
    header_rows: list[list[str]] = []
    data_start_idx = 1
    for row in reader:
      data_start_idx += 1
      if not any(cell.strip() for cell in row):
        continue
      label = _header_row_label(row)
      if label == "Frame":
        header_rows.append(row)
        break
      header_rows.append(row)
    else:
      raise MotiveCSVParseError("Could not find axis header row starting with Frame.")

  return metadata, header_rows, data_start_idx


def _build_marker_columns(
  header_rows: list[list[str]],
  header_map: dict[str, int],
  config: dict[str, Any],
  messages: list[QCMessage],
) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
  type_row = header_rows[header_map["type"]]
  name_row = header_rows[header_map["name"]]
  axis_row = header_rows[header_map["axis"]]
  channel_idx = header_map.get("channel")
  channel_row = header_rows[channel_idx] if channel_idx is not None else []

  accepted_types = set(config["parsing"].get("accepted_marker_types", ["Marker"]))
  markers: dict[str, dict[str, Any]] = {}
  non_marker_types: dict[str, set[str]] = {
    "rigid_body": set(),
    "skeleton": set(),
    "quaternion": set(),
    "other": set(),
  }

  max_cols = max(len(type_row), len(name_row), len(axis_row))
  for col in range(2, max_cols):
    marker_name = name_row[col].strip() if col < len(name_row) else ""
    if not marker_name:
      continue
    marker_type = type_row[col].strip() if col < len(type_row) else ""
    axis = axis_row[col].strip() if col < len(axis_row) else ""
    channel = channel_row[col].strip() if col < len(channel_row) else ""

    if marker_type and marker_type not in accepted_types:
      lowered = marker_type.lower()
      if "rigid" in lowered or "body" in lowered:
        non_marker_types["rigid_body"].add(marker_type)
      elif "skeleton" in lowered or "bone" in lowered:
        non_marker_types["skeleton"].add(marker_type)
      elif "quaternion" in lowered or lowered == "rotation":
        non_marker_types["quaternion"].add(marker_type)
      else:
        non_marker_types["other"].add(marker_type)
      continue

    if channel and channel.lower() not in ("position", ""):
      if "quaternion" in channel.lower() or channel.lower() == "rotation":
        non_marker_types["quaternion"].add(channel)
        continue

    if axis not in ("X", "Y", "Z"):
      continue

    if marker_name not in markers:
      prefix, short_name = _parse_subject_prefix(marker_name)
      is_unlabeled = _is_unlabeled_marker(marker_name)
      markers[marker_name] = {
        "marker_name": marker_name,
        "marker_short_name": short_name,
        "subject_or_asset_prefix": prefix,
        "is_labeled": not is_unlabeled,
        "is_unlabeled": is_unlabeled,
        "marker_type_raw": marker_type or "Marker",
        "body_region_group": _assign_body_region(marker_name, config["marker_groups"]),
        "axes": {},
      }
    markers[marker_name]["axes"][axis] = col

  marker_records: list[dict[str, Any]] = []
  duplicate_names: list[str] = []
  for marker_name, info in sorted(markers.items()):
    axes = info["axes"]
    has_x, has_y, has_z = "X" in axes, "Y" in axes, "Z" in axes
    if len(axes) > 3:
      duplicate_names.append(marker_name)
    parse_status = "ok"
    if not (has_x and has_y and has_z):
      parse_status = "missing_axis"
    marker_records.append(
      {
        "marker_name": marker_name,
        "marker_short_name": info["marker_short_name"],
        "subject_or_asset_prefix": info["subject_or_asset_prefix"],
        "is_labeled": info["is_labeled"],
        "is_unlabeled": info["is_unlabeled"],
        "marker_type_raw": info["marker_type_raw"],
        "body_region_group": info["body_region_group"],
        "x_column_source": axes.get("X"),
        "y_column_source": axes.get("Y"),
        "z_column_source": axes.get("Z"),
        "has_x": has_x,
        "has_y": has_y,
        "has_z": has_z,
        "parse_status": parse_status,
      }
    )

  if duplicate_names and config["parsing"].get("fail_on_duplicate_marker_names", True):
    raise SchemaValidationError(
      f"Duplicate marker axis definitions detected for: {', '.join(duplicate_names)}"
    )

  if non_marker_types["rigid_body"]:
    messages.append(
      QCMessage(
        "WARNING",
        "RIGID_BODY_COLUMNS",
        "Rigid-body columns detected and excluded from raw marker QC.",
        {"types": sorted(non_marker_types["rigid_body"])},
      )
    )
  if non_marker_types["skeleton"]:
    messages.append(
      QCMessage(
        "WARNING",
        "SKELETON_COLUMNS",
        "Skeleton/bone columns detected and excluded from raw marker QC.",
        {"types": sorted(non_marker_types["skeleton"])},
      )
    )
  if non_marker_types["quaternion"]:
    messages.append(
      QCMessage(
        "WARNING",
        "QUATERNION_COLUMNS",
        "Quaternion/rotation columns detected and excluded from raw marker QC.",
        {"types": sorted(non_marker_types["quaternion"])},
      )
    )

  return marker_records, non_marker_types


def _filter_markers(marker_records: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
  markers_cfg = config["markers"]
  exclude = set(markers_cfg.get("exclude_markers", []))
  include_only = set(markers_cfg.get("include_only_markers", []))
  filtered: list[dict[str, Any]] = []
  for record in marker_records:
    name = record["marker_name"]
    if name in exclude:
      continue
    if include_only and name not in include_only:
      continue
    if record["is_unlabeled"] and not markers_cfg.get("include_unlabeled_markers", True):
      continue
    if record["is_labeled"] and not markers_cfg.get("include_labeled_markers", True):
      continue
    filtered.append(record)
  return filtered


def _metadata_float(metadata: dict[str, str], key: str) -> float | None:
  value = metadata.get(key)
  if value is None or value == "":
    return None
  try:
    return float(value)
  except ValueError:
    return None


def _metadata_int(metadata: dict[str, str], key: str) -> int | None:
  value = metadata.get(key)
  if value is None or value == "":
    return None
  try:
    return int(float(value))
  except ValueError:
    return None


# ---------------------------------------------------------------------------
# Layer 1
# ---------------------------------------------------------------------------


def run_layer1_parse(config: dict[str, Any], verbose: bool = False) -> QCResult:
  messages: list[QCMessage] = []
  base_dir = Path(config.get("_base_dir", Path(config["_config_path"]).parent))

  input_path = _resolve_path(base_dir, config["paths"]["input_csv"])
  if not input_path.exists():
    raise FileNotFoundError(f"Input CSV not found: {input_path}")

  if verbose:
    LOGGER.info("Parsing %s", input_path)

  metadata_row, header_rows, data_start_idx = _read_csv_header(input_path)
  header_map = _find_header_rows(header_rows)
  marker_records, non_marker_types = _build_marker_columns(
    header_rows, header_map, config, messages
  )
  marker_records = _filter_markers(marker_records, config)

  if not marker_records:
    raise SchemaValidationError("No marker XYZ triplets identified in CSV header.")

  if config["parsing"].get("require_marker_xyz_triplets", True):
    bad = [m["marker_name"] for m in marker_records if m["parse_status"] != "ok"]
    if bad and config["parsing"].get("fail_on_missing_xyz_axis", True):
      raise SchemaValidationError(
        f"Markers missing complete XYZ triplets: {', '.join(bad[:10])}"
        + (" ..." if len(bad) > 10 else "")
      )

  marker_inventory = pd.DataFrame(marker_records)
  marker_names = marker_inventory["marker_name"].tolist()
  axis_row = header_rows[header_map["axis"]]

  usecols = ["Frame", "Time (Seconds)"]
  col_indices: dict[str, tuple[str, int]] = {}
  for _, row in marker_inventory.iterrows():
    name = row["marker_name"]
    for axis in ("X", "Y", "Z"):
      source = row[f"{axis.lower()}_column_source"]
      header_name = axis_row[int(source)] if source is not None else axis
      col_key = f"{name}|{axis}"
      usecols.append(col_key)
      col_indices[col_key] = (name, int(source))

  # Read data with pandas using column indices
  all_cols = pd.read_csv(
    input_path,
    skiprows=data_start_idx - 1,
    header=0,
    encoding="utf-8-sig",
    low_memory=False,
  )
  if "Frame" not in all_cols.columns:
    raise MotiveCSVParseError("No Frame column found in Motive CSV data section.")

  frame_col = all_cols["Frame"]
  if frame_col.isna().all():
    raise MotiveCSVParseError("Frame column is empty.")

  frames = pd.Index(frame_col.astype(int).tolist(), name="frame")
  if config["parsing"].get("fail_on_duplicate_frames", True) and frame_col.duplicated().any():
    dups = frame_col[frame_col.duplicated()].unique()[:10]
    raise QCValidationError(f"Duplicate frame numbers detected: {list(dups)}")

  if not frame_col.is_monotonic_increasing and config["parsing"].get(
    "fail_on_non_monotonic_frames", True
  ):
    raise QCValidationError("Frame numbers are not monotonically increasing.")

  expected_frames = pd.Index(range(int(frames.min()), int(frames.max()) + 1))
  missing_frames = expected_frames.difference(frames)
  frame_continuity_status = "continuous"
  if len(missing_frames) > 0:
    frame_continuity_status = "missing_frames"
    msg = (
      f"Missing {len(missing_frames)} frame numbers between "
      f"{int(frames.min())} and {int(frames.max())}."
    )
    if config["parsing"].get("fail_on_missing_frames", False):
      raise QCValidationError(msg)
    messages.append(
      QCMessage(
        "WARNING",
        "MISSING_FRAMES",
        msg,
        {"missing_count": int(len(missing_frames)), "first_missing": int(missing_frames[0])},
      )
    )

  if "Time (Seconds)" in all_cols.columns:
    time_seconds = pd.to_numeric(all_cols["Time (Seconds)"], errors="coerce")
    time_column_status = "ok"
  else:
    time_seconds = pd.Series(np.nan, index=range(len(frames)))
    time_column_status = "missing"
    messages.append(
      QCMessage("WARNING", "MISSING_TIME_COLUMN", "Time (Seconds) column not found.")
    )

  capture_rate = _metadata_float(metadata_row, "Capture Frame Rate")
  export_rate = _metadata_float(metadata_row, "Export Frame Rate")
  override_rate = config["time"].get("frame_rate_hz_override")
  effective_rate = override_rate or export_rate or capture_rate
  frame_rate_status = "ok"

  if effective_rate is None:
    raise QCValidationError(
      "Effective frame rate could not be determined. "
      "Set time.frame_rate_hz_override in config or ensure CSV metadata includes Export Frame Rate."
    )
  if override_rate is not None:
    frame_rate_status = "missing_used_override"
  elif export_rate is None and capture_rate is not None:
    frame_rate_status = "missing_used_override"
    messages.append(
      QCMessage(
        "WARNING",
        "EXPORT_RATE_MISSING",
        "Export Frame Rate missing; using Capture Frame Rate.",
      )
    )
  if (
    capture_rate is not None
    and export_rate is not None
    and abs(capture_rate - export_rate) > 1e-6
  ):
    if config["time"].get("require_capture_export_rate_match", True):
      raise QCValidationError(
        f"Capture/export frame rate mismatch: {capture_rate} vs {export_rate} Hz."
      )
    frame_rate_status = "mismatch"
    messages.append(
      QCMessage(
        "WARNING",
        "FRAME_RATE_MISMATCH",
        f"Capture/export frame rate mismatch: {capture_rate} vs {export_rate} Hz.",
      )
    )

  if time_column_status == "ok" and len(time_seconds) > 1:
    diffs = time_seconds.diff().iloc[1:]
    expected_dt = 1.0 / effective_rate
    tolerance = config["time"].get("allow_time_column_tolerance_seconds", 0.0005)
    if not np.allclose(diffs, expected_dt, atol=tolerance, rtol=0.01, equal_nan=False):
      time_column_status = "inconsistent_with_frame_rate"
      messages.append(
        QCMessage(
          "WARNING",
          "TIME_INCONSISTENT",
          "Time column differences are inconsistent with effective frame rate.",
          {"expected_dt": expected_dt, "median_dt": float(np.nanmedian(diffs))},
        )
      )

  n_frames = len(frames)
  n_markers = len(marker_names)
  coord_array = np.full((n_frames, n_markers, 3), np.nan, dtype=float)
  convert_blanks = config["parsing"].get("convert_blank_cells_to_nan", True)
  fail_non_numeric = config["parsing"].get("fail_on_non_numeric_coordinate_values", False)
  partial_axis_invalid = 0

  raw_header = list(all_cols.columns)
  for marker_idx, marker_name in enumerate(marker_names):
    row = marker_inventory.loc[marker_inventory["marker_name"] == marker_name].iloc[0]
    for axis_idx, axis in enumerate(("X", "Y", "Z")):
      source = int(row[f"{axis.lower()}_column_source"])
      if source >= len(raw_header):
        raise SchemaValidationError(
          f"Column index {source} out of range for marker {marker_name} axis {axis}."
        )
      series = pd.to_numeric(all_cols.iloc[:, source], errors="coerce")
      if fail_non_numeric:
        raw = all_cols.iloc[:, source].astype(str).str.strip()
        non_empty = raw != ""
        bad = non_empty & series.isna()
        if bad.any():
          raise QCValidationError(
            f"Non-numeric coordinate values for {marker_name} {axis}."
          )
      coord_array[:, marker_idx, axis_idx] = series.to_numpy()

  valid = np.isfinite(coord_array).all(axis=2)
  finite_axes = np.isfinite(coord_array).sum(axis=2)
  partial_axis_invalid = int(((finite_axes > 0) & (finite_axes < 3)).sum())
  if partial_axis_invalid > 0:
    messages.append(
      QCMessage(
        "WARNING",
        "PARTIAL_AXIS_MISSING",
        f"{partial_axis_invalid} marker-frames have only some XYZ axes present.",
        {"count": partial_axis_invalid},
      )
    )

  coordinates = xr.DataArray(
    coord_array,
    dims=["frame", "marker", "axis"],
    coords={"frame": frames.values, "marker": marker_names, "axis": ["X", "Y", "Z"]},
    name="coordinates",
  )
  valid_marker_frame = xr.DataArray(
    valid,
    dims=["frame", "marker"],
    coords={"frame": frames.values, "marker": marker_names},
    name="valid_marker_frame",
  )

  length_units = metadata_row.get("Length Units")
  coordinate_space = metadata_row.get("Coordinate Space")
  rotation_type = metadata_row.get("Rotation Type")
  if not length_units:
    messages.append(
      QCMessage("WARNING", "UNITS_MISSING", "Length units not found in CSV metadata.")
    )

  contains_marker_xyz = True
  contains_rigid = bool(non_marker_types["rigid_body"])
  contains_skeleton = bool(non_marker_types["skeleton"])
  contains_quaternion = bool(non_marker_types["quaternion"])
  if contains_rigid or contains_skeleton or contains_quaternion:
    raw_data_status = "ambiguous"
  else:
    raw_data_status = "consistent_with_marker_xyz"

  n_errors = sum(1 for m in messages if m.severity == "ERROR")
  n_warnings = sum(1 for m in messages if m.severity == "WARNING")
  validation_status = "pass" if n_errors == 0 and n_warnings == 0 else (
    "fail" if n_errors > 0 else "pass_with_warnings"
  )

  session_metadata = {
    "input_file": str(input_path),
    "file_stem": input_path.stem,
    "file_name": input_path.name,
    "motive_version": config["project"]["motive_version"],
    "capture_frame_rate_hz": capture_rate,
    "export_frame_rate_hz": export_rate,
    "effective_frame_rate_hz": float(effective_rate),
    "total_frames_metadata": _metadata_int(metadata_row, "Total Exported Frames")
    or _metadata_int(metadata_row, "Total Frames in Take"),
    "total_frames_observed": n_frames,
    "duration_seconds": float((n_frames - 1) / effective_rate) if n_frames > 1 else 0.0,
    "start_frame": int(frames.min()),
    "end_frame": int(frames.max()),
    "rotation_type": rotation_type,
    "length_units": length_units,
    "coordinate_space": coordinate_space,
    "axis_convention": metadata_row.get("Axis", "unknown"),
    "raw_data_status": raw_data_status,
    "frame_rate_status": frame_rate_status,
    "frame_continuity_status": frame_continuity_status,
    "time_column_status": time_column_status,
    "n_marker_triplets_total": len(marker_records),
    "n_labeled_markers": int(marker_inventory["is_labeled"].sum()),
    "n_unlabeled_markers": int(marker_inventory["is_unlabeled"].sum()),
    "contains_marker_xyz": contains_marker_xyz,
    "contains_rigid_body_columns": contains_rigid,
    "contains_skeleton_columns": contains_skeleton,
    "contains_quaternion_columns": contains_quaternion,
    "validation_status": validation_status,
    "n_errors": n_errors,
    "n_warnings": n_warnings,
    "partial_axis_invalid_count": partial_axis_invalid,
    "project_name": config["project"]["project_name"],
    "subject_id": config["project"]["subject_id"],
    "session_id": config["project"]["session_id"],
  }

  session = MotiveSession(
    metadata=session_metadata,
    frames=frames,
    time_seconds=time_seconds.reset_index(drop=True),
    coordinates=coordinates,
    valid_marker_frame=valid_marker_frame,
    marker_inventory=marker_inventory,
    validation_messages=messages,
  )

  session_summary = _build_layer1_session_summary(session)
  status = validation_status
  return QCResult(
    layer_name="layer1",
    status=status,
    tables={"session_summary": session_summary, "marker_inventory": marker_inventory},
    messages=messages,
    session=session,
  )


def _build_layer1_session_summary(session: MotiveSession) -> pd.DataFrame:
  md = session.metadata
  row = {
    "file_name": md["file_name"],
    "input_file": md["input_file"],
    "project_name": md["project_name"],
    "subject_id": md["subject_id"],
    "session_id": md["session_id"],
    "motive_version": md["motive_version"],
    "capture_frame_rate_hz": md["capture_frame_rate_hz"],
    "export_frame_rate_hz": md["export_frame_rate_hz"],
    "effective_frame_rate_hz": md["effective_frame_rate_hz"],
    "frame_rate_status": md["frame_rate_status"],
    "total_frames_metadata": md["total_frames_metadata"],
    "total_frames_observed": md["total_frames_observed"],
    "frame_start": md["start_frame"],
    "frame_end": md["end_frame"],
    "frame_continuity_status": md["frame_continuity_status"],
    "duration_seconds": md["duration_seconds"],
    "time_column_status": md["time_column_status"],
    "length_units": md["length_units"],
    "coordinate_space": md["coordinate_space"],
    "axis_convention": md["axis_convention"],
    "rotation_type": md["rotation_type"],
    "n_marker_triplets_total": md["n_marker_triplets_total"],
    "n_labeled_markers": md["n_labeled_markers"],
    "n_unlabeled_markers": md["n_unlabeled_markers"],
    "contains_marker_xyz": md["contains_marker_xyz"],
    "contains_rigid_body_columns": md["contains_rigid_body_columns"],
    "contains_skeleton_columns": md["contains_skeleton_columns"],
    "contains_quaternion_columns": md["contains_quaternion_columns"],
    "raw_data_status": md["raw_data_status"],
    "validation_status": md["validation_status"],
    "n_errors": md["n_errors"],
    "n_warnings": md["n_warnings"],
  }
  layer2_cols = [
    "total_marker_frames_all",
    "missing_marker_frames_all",
    "missing_percent_all",
    "total_marker_frames_labeled",
    "missing_marker_frames_labeled",
    "missing_percent_labeled",
    "total_marker_frames_unlabeled",
    "missing_marker_frames_unlabeled",
    "missing_percent_unlabeled",
    "n_gaps_total_all",
    "n_gaps_total_labeled",
    "n_gaps_ge_0p2s_labeled",
    "n_gaps_ge_0p5s_labeled",
    "n_gaps_ge_1p0s_labeled",
    "longest_gap_marker_labeled",
    "longest_gap_seconds_labeled",
    "raw_qc_preprocessing_status",
    "raw_qc_status_reason",
  ]
  for col in layer2_cols:
    row[col] = "not_computed"
  return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Layer 2
# ---------------------------------------------------------------------------


def _seconds_to_key(seconds: float) -> str:
  return f"{seconds:.3f}".rstrip("0").rstrip(".").replace(".", "p") + "s"


def _gap_threshold_labels(config: dict[str, Any]) -> list[tuple[str, float]]:
  thresholds = config["gaps"]["thresholds_seconds"]
  ordered = [
    ("tiny", thresholds.get("tiny_gap", 0.025)),
    ("minor", thresholds.get("minor_gap", 0.1)),
    ("moderate", thresholds.get("moderate_gap", 0.2)),
    ("large", thresholds.get("large_gap", 0.5)),
    ("severe", thresholds.get("severe_gap", 1.0)),
  ]
  return ordered


def _crossed_thresholds(duration_seconds: float, config: dict[str, Any]) -> list[str]:
  use_ge = config["gaps"].get("use_greater_equal_thresholds", True)
  crossed = []
  for label, threshold in _gap_threshold_labels(config):
    if use_ge:
      if duration_seconds >= threshold:
        crossed.append(label)
    else:
      if duration_seconds > threshold:
        crossed.append(label)
  return crossed


def _severity_label(duration_frames: int, duration_seconds: float, config: dict[str, Any]) -> str:
  if duration_frames == 1:
    return "single_frame"
  crossed = _crossed_thresholds(duration_seconds, config)
  if not crossed:
    return "tiny"
  return crossed[-1]


def _recommended_status(severity: str, touches_edge: bool) -> str:
  if severity in ("severe", "large") or (touches_edge and severity in ("moderate", "large", "severe")):
    return "potential_exclusion"
  if severity in ("moderate", "large"):
    return "caution"
  return "document"


def _count_gaps_ge(gap_durations_seconds: list[float], threshold: float, config: dict[str, Any]) -> int:
  use_ge = config["gaps"].get("use_greater_equal_thresholds", True)
  if use_ge:
    return sum(1 for value in gap_durations_seconds if value >= threshold)
  return sum(1 for value in gap_durations_seconds if value > threshold)


def _marker_quality_label(
  missing_percent: float,
  n_large_gaps: int,
  config: dict[str, Any],
) -> tuple[str, str]:
  labels = config["quality_labels"]["marker"]
  clean = labels["clean"]
  minor = labels["minor_issue"]
  caution = labels["caution"]
  poor = labels["poor"]

  if (
    missing_percent > poor["missing_percent_above"]
    or n_large_gaps > poor["large_gaps_above"]
  ):
    return "poor", "Missing percent or large-gap count exceeds poor thresholds."
  if missing_percent <= clean["max_missing_percent"] and n_large_gaps <= clean["max_large_gaps"]:
    return "clean", "Within clean thresholds."
  if missing_percent <= minor["max_missing_percent"] and n_large_gaps <= minor["max_large_gaps"]:
    return "minor_issue", "Within minor-issue thresholds."
  if missing_percent <= caution["max_missing_percent"] and n_large_gaps <= caution["max_large_gaps"]:
    return "caution", "Within caution thresholds."
  return "poor", "Exceeded caution thresholds."


def _detect_gaps_for_marker(
  valid: np.ndarray,
  frames: np.ndarray,
  frame_rate: float,
  marker_name: str,
  inventory_row: pd.Series,
  config: dict[str, Any],
  gap_id_start: int,
) -> tuple[list[dict[str, Any]], int]:
  gaps: list[dict[str, Any]] = []
  gap_id = gap_id_start
  n_frames = len(valid)
  idx = 0
  while idx < n_frames:
    if valid[idx]:
      idx += 1
      continue
    start_idx = idx
    while idx < n_frames and not valid[idx]:
      idx += 1
    end_idx = idx - 1
    duration_frames = end_idx - start_idx + 1
    duration_seconds = duration_frames / frame_rate
    if duration_frames == 1 and not config["gaps"].get("report_single_frame_gaps", True):
      continue
    crossed = _crossed_thresholds(duration_seconds, config)
    severity = _severity_label(duration_frames, duration_seconds, config)
    start_frame = int(frames[start_idx])
    end_frame = int(frames[end_idx])
    prev_valid = int(frames[start_idx - 1]) if start_idx > 0 else None
    next_valid = int(frames[end_idx + 1]) if end_idx < n_frames - 1 else None
    touches_edge = start_idx == 0 or end_idx == n_frames - 1
    gap_id += 1
    gaps.append(
      {
        "gap_id": f"G{gap_id:06d}",
        "marker_name": marker_name,
        "is_labeled": bool(inventory_row["is_labeled"]),
        "is_unlabeled": bool(inventory_row["is_unlabeled"]),
        "body_region_group": inventory_row["body_region_group"],
        "gap_start_frame": start_frame,
        "gap_end_frame": end_frame,
        "gap_start_time_seconds": float(start_frame / frame_rate),
        "gap_end_time_seconds": float(end_frame / frame_rate),
        "duration_frames": duration_frames,
        "duration_seconds": float(duration_seconds),
        "thresholds_crossed": ";".join(crossed),
        "severity_label": severity,
        "prev_valid_frame": prev_valid,
        "next_valid_frame": next_valid,
        "touches_start_or_end": touches_edge,
        "recommended_status": _recommended_status(severity, touches_edge),
      }
    )
  return gaps, gap_id


def run_layer2_gaps(session: MotiveSession, config: dict[str, Any], verbose: bool = False) -> QCResult:
  messages = list(session.validation_messages)
  frame_rate = float(session.metadata["effective_frame_rate_hz"])
  frames = session.coordinates.coords["frame"].values
  valid_da = session.valid_marker_frame
  inventory = session.marker_inventory.set_index("marker_name")
  thresholds = config["gaps"]["thresholds_seconds"]

  all_gaps: list[dict[str, Any]] = []
  gap_id = 0
  marker_rows: list[dict[str, Any]] = []

  for marker in session.coordinates.coords["marker"].values:
    valid = valid_da.sel(marker=marker).values.astype(bool)
    inv = inventory.loc[marker]
    gaps, gap_id = _detect_gaps_for_marker(
      valid, frames, frame_rate, marker, inv, config, gap_id
    )
    all_gaps.extend(gaps)

    n_total = len(valid)
    n_valid = int(valid.sum())
    n_missing = n_total - n_valid
    missing_percent = 100.0 * n_missing / n_total if n_total else 0.0
    gap_durations_frames = [g["duration_frames"] for g in gaps]
    gap_durations_seconds = [g["duration_seconds"] for g in gaps]
    longest_gap_frames = max(gap_durations_frames) if gaps else 0
    longest_gap_seconds = max(gap_durations_seconds) if gaps else 0.0
    n_single = sum(1 for value in gap_durations_frames if value == 1)
    n_large = _count_gaps_ge(gap_durations_seconds, thresholds["large_gap"], config)

    missing_idx = np.where(~valid)[0]
    first_missing = int(frames[missing_idx[0]]) if len(missing_idx) else None
    last_missing = int(frames[missing_idx[-1]]) if len(missing_idx) else None

    quality_label, quality_reason = _marker_quality_label(
      missing_percent, n_large, config
    )

    marker_rows.append(
      {
        "marker_name": marker,
        "is_labeled": bool(inv["is_labeled"]),
        "is_unlabeled": bool(inv["is_unlabeled"]),
        "body_region_group": inv["body_region_group"],
        "n_total_frames": n_total,
        "n_valid_frames": n_valid,
        "n_missing_frames": n_missing,
        "missing_percent": round(missing_percent, 6),
        "n_gaps_total": len(gaps),
        "n_single_frame_gaps": n_single,
        "longest_gap_frames": longest_gap_frames,
        "longest_gap_seconds": round(longest_gap_seconds, 6),
        "mean_gap_frames": round(float(np.mean(gap_durations_frames)), 6) if gaps else None,
        "median_gap_frames": round(float(np.median(gap_durations_frames)), 6) if gaps else None,
        "n_gaps_ge_0p025s": _count_gaps_ge(gap_durations_seconds, thresholds["tiny_gap"], config),
        "n_gaps_ge_0p1s": _count_gaps_ge(gap_durations_seconds, thresholds["minor_gap"], config),
        "n_gaps_ge_0p2s": _count_gaps_ge(gap_durations_seconds, thresholds["moderate_gap"], config),
        "n_gaps_ge_0p5s": _count_gaps_ge(gap_durations_seconds, thresholds["large_gap"], config),
        "n_gaps_ge_1p0s": _count_gaps_ge(gap_durations_seconds, thresholds["severe_gap"], config),
        "first_missing_frame": first_missing,
        "last_missing_frame": last_missing,
        "quality_label": quality_label,
        "quality_reason": quality_reason,
      }
    )

  marker_quality_summary = pd.DataFrame(marker_rows)
  gap_events = pd.DataFrame(all_gaps)

  gap_summary_by_marker = marker_quality_summary[
    [
      "marker_name",
      "is_labeled",
      "is_unlabeled",
      "body_region_group",
      "n_gaps_total",
      "n_single_frame_gaps",
      "longest_gap_frames",
      "longest_gap_seconds",
      "n_gaps_ge_0p025s",
      "n_gaps_ge_0p1s",
      "n_gaps_ge_0p2s",
      "n_gaps_ge_0p5s",
      "n_gaps_ge_1p0s",
      "missing_percent",
      "quality_label",
    ]
  ].copy()

  gap_summary_by_group = _build_gap_summary_by_group(marker_quality_summary, gap_events)
  session_summary = _update_session_summary_layer2(
    session, marker_quality_summary, gap_events, config
  )

  figures = generate_layer2_plots(session, marker_quality_summary, gap_events, config)

  status = session.metadata.get("validation_status", "pass")
  return QCResult(
    layer_name="layer2",
    status=status,
    tables={
      "session_summary": session_summary,
      "marker_inventory": session.marker_inventory,
      "marker_quality_summary": marker_quality_summary,
      "gap_events": gap_events,
      "gap_summary_by_marker": gap_summary_by_marker,
      "gap_summary_by_group": gap_summary_by_group,
    },
    figures=figures,
    messages=messages,
    session=session,
  )


def _build_gap_summary_by_group(
  marker_quality: pd.DataFrame,
  gap_events: pd.DataFrame,
) -> pd.DataFrame:
  rows: list[dict[str, Any]] = []
  for group, group_df in marker_quality.groupby("body_region_group", sort=True):
    group_gaps = (
      gap_events[gap_events["body_region_group"] == group]
      if not gap_events.empty
      else pd.DataFrame()
    )
    worst_marker = None
    if not group_df.empty:
      worst_idx = group_df["missing_percent"].idxmax()
      worst_marker = group_df.loc[worst_idx, "marker_name"]
    rows.append(
      {
        "body_region_group": group,
        "n_markers": len(group_df),
        "n_labeled_markers": int(group_df["is_labeled"].sum()),
        "n_unlabeled_markers": int(group_df["is_unlabeled"].sum()),
        "total_missing_frames": int(group_df["n_missing_frames"].sum()),
        "mean_missing_percent": round(float(group_df["missing_percent"].mean()), 6)
        if len(group_df)
        else 0.0,
        "max_missing_percent": round(float(group_df["missing_percent"].max()), 6)
        if len(group_df)
        else 0.0,
        "n_gaps_total": int(group_df["n_gaps_total"].sum()),
        "n_gaps_ge_0p2s": int(group_df["n_gaps_ge_0p2s"].sum()),
        "n_gaps_ge_0p5s": int(group_df["n_gaps_ge_0p5s"].sum()),
        "n_gaps_ge_1p0s": int(group_df["n_gaps_ge_1p0s"].sum()),
        "longest_gap_frames": int(group_df["longest_gap_frames"].max()) if len(group_df) else 0,
        "longest_gap_seconds": round(float(group_df["longest_gap_seconds"].max()), 6)
        if len(group_df)
        else 0.0,
        "worst_marker": worst_marker,
      }
    )
  return pd.DataFrame(rows)


def _update_session_summary_layer2(
  session: MotiveSession,
  marker_quality: pd.DataFrame,
  gap_events: pd.DataFrame,
  config: dict[str, Any],
) -> pd.DataFrame:
  summary = _build_layer1_session_summary(session).iloc[0].to_dict()
  labeled = marker_quality[marker_quality["is_labeled"]]
  unlabeled = marker_quality[marker_quality["is_unlabeled"]]
  n_frames = int(session.metadata["total_frames_observed"])

  def _totals(df: pd.DataFrame) -> tuple[int, int, float]:
    total = int(df["n_total_frames"].sum()) if len(df) else 0
    missing = int(df["n_missing_frames"].sum()) if len(df) else 0
    pct = 100.0 * missing / total if total else 0.0
    return total, missing, round(pct, 6)

  total_all, missing_all, pct_all = _totals(marker_quality)
  total_lab, missing_lab, pct_lab = _totals(labeled)
  total_unl, missing_unl, pct_unl = _totals(unlabeled)

  labeled_gaps = gap_events[gap_events["is_labeled"]] if not gap_events.empty else pd.DataFrame()
  thresholds = config["gaps"]["thresholds_seconds"]
  n_ge_02 = _count_gaps_ge(
    labeled_gaps["duration_seconds"].tolist(), thresholds["moderate_gap"], config
  ) if not labeled_gaps.empty else 0
  n_ge_05 = _count_gaps_ge(
    labeled_gaps["duration_seconds"].tolist(), thresholds["large_gap"], config
  ) if not labeled_gaps.empty else 0
  n_ge_10 = _count_gaps_ge(
    labeled_gaps["duration_seconds"].tolist(), thresholds["severe_gap"], config
  ) if not labeled_gaps.empty else 0

  longest_marker = None
  longest_seconds = 0.0
  if not labeled_gaps.empty:
    idx = labeled_gaps["duration_seconds"].idxmax()
    longest_marker = labeled_gaps.loc[idx, "marker_name"]
    longest_seconds = float(labeled_gaps.loc[idx, "duration_seconds"])

  session_cfg = config["quality_labels"]["session"]
  acceptable = session_cfg["acceptable_for_preprocessing"]
  caution = session_cfg["caution_for_preprocessing"]
  if pct_lab <= acceptable["max_labeled_missing_percent"] and n_ge_05 <= acceptable["max_large_gaps_labeled"]:
    raw_status = "acceptable"
    raw_reason = "Labeled missingness and large-gap counts within acceptable preprocessing thresholds."
  elif pct_lab <= caution["max_labeled_missing_percent"] and n_ge_05 <= caution["max_large_gaps_labeled"]:
    raw_status = "caution"
    raw_reason = "Labeled missingness or large-gap counts require caution before preprocessing."
  else:
    raw_status = "poor"
    raw_reason = "Labeled missingness or large-gap counts exceed caution thresholds."

  summary.update(
    {
      "total_marker_frames_all": total_all,
      "missing_marker_frames_all": missing_all,
      "missing_percent_all": pct_all,
      "total_marker_frames_labeled": total_lab,
      "missing_marker_frames_labeled": missing_lab,
      "missing_percent_labeled": pct_lab,
      "total_marker_frames_unlabeled": total_unl,
      "missing_marker_frames_unlabeled": missing_unl,
      "missing_percent_unlabeled": pct_unl,
      "n_gaps_total_all": int(gap_events.shape[0]) if not gap_events.empty else 0,
      "n_gaps_total_labeled": int(labeled_gaps.shape[0]) if not labeled_gaps.empty else 0,
      "n_gaps_ge_0p2s_labeled": n_ge_02,
      "n_gaps_ge_0p5s_labeled": n_ge_05,
      "n_gaps_ge_1p0s_labeled": n_ge_10,
      "longest_gap_marker_labeled": longest_marker,
      "longest_gap_seconds_labeled": round(longest_seconds, 6),
      "raw_qc_preprocessing_status": raw_status,
      "raw_qc_status_reason": raw_reason,
    }
  )
  return pd.DataFrame([summary])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def generate_layer2_plots(
  session: MotiveSession,
  marker_quality: pd.DataFrame,
  gap_events: pd.DataFrame,
  config: dict[str, Any],
) -> dict[str, Path]:
  outputs_cfg = config["outputs"]
  if not outputs_cfg.get("plots", {}).get("enabled", True):
    return {}

  base_dir = Path(config.get("_base_dir", Path(config["_config_path"]).parent))
  plot_dir = _resolve_path(base_dir, config["paths"]["output_dir"]) / "plots"
  plot_dir.mkdir(parents=True, exist_ok=True)
  dpi = outputs_cfg.get("dpi", 300)
  fmt = outputs_cfg.get("plot_format", "png")
  figures: dict[str, Path] = {}

  if outputs_cfg["plots"].get("marker_completeness", True):
    path = plot_dir / f"marker_completeness.{fmt}"
    _plot_marker_completeness(marker_quality, path, dpi)
    figures["marker_completeness"] = path

  if outputs_cfg["plots"].get("gap_duration_histogram", True):
    path = plot_dir / f"gap_duration_histogram.{fmt}"
    _plot_gap_duration_histogram(gap_events, config, path, dpi)
    figures["gap_duration_histogram"] = path

  if outputs_cfg["plots"].get("missing_data_heatmap_labeled", True):
    path = plot_dir / f"missing_data_heatmap_labeled.{fmt}"
    _plot_missing_heatmap(session, labeled_only=True, config=config, output_path=path, dpi=dpi)
    figures["missing_data_heatmap_labeled"] = path

  if (
    outputs_cfg["plots"].get("missing_data_heatmap_unlabeled", True)
    and marker_quality["is_unlabeled"].any()
  ):
    path = plot_dir / f"missing_data_heatmap_unlabeled.{fmt}"
    _plot_missing_heatmap(session, labeled_only=False, config=config, output_path=path, dpi=dpi)
    figures["missing_data_heatmap_unlabeled"] = path

  return figures


def _plot_marker_completeness(marker_quality: pd.DataFrame, output_path: Path, dpi: int) -> None:
  df = marker_quality.sort_values(["is_labeled", "marker_name"], ascending=[False, True]).copy()
  df["completeness_percent"] = 100.0 - df["missing_percent"]
  colors = np.where(df["is_labeled"], "#2b6cb0", "#a0aec0")
  fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.18)))
  ax.barh(df["marker_name"], df["completeness_percent"], color=colors)
  ax.set_xlabel("Valid marker-frame percent")
  ax.set_ylabel("Marker")
  ax.set_xlim(0, 100)
  ax.set_title("Marker completeness (blue=labeled, gray=unlabeled)")
  fig.tight_layout()
  fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
  plt.close(fig)


def _plot_gap_duration_histogram(
  gap_events: pd.DataFrame,
  config: dict[str, Any],
  output_path: Path,
  dpi: int,
) -> None:
  fig, ax = plt.subplots(figsize=(10, 6))
  if gap_events.empty:
    ax.text(0.5, 0.5, "No gaps detected", ha="center", va="center")
  else:
    durations = gap_events["duration_seconds"]
    ax.hist(durations, bins=50, color="#4a5568", edgecolor="white")
    thresholds = config["gaps"]["primary_report_thresholds_seconds"]
    colors = ["#ecc94b", "#ed8936", "#e53e3e", "#9b2c2c"]
    for value, color in zip(thresholds, colors):
      ax.axvline(value, color=color, linestyle="--", linewidth=1.5, label=f"{value:.1f} s")
    ax.set_xlabel("Gap duration (seconds)")
    ax.set_ylabel("Count")
    ax.legend()
  ax.set_title("Gap duration distribution")
  fig.tight_layout()
  fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
  plt.close(fig)


def _plot_missing_heatmap(
  session: MotiveSession,
  labeled_only: bool,
  config: dict[str, Any],
  output_path: Path,
  dpi: int,
) -> None:
  inventory = session.marker_inventory
  if labeled_only:
    markers = inventory.loc[inventory["is_labeled"], "marker_name"].tolist()
    title = "Missing data heatmap (labeled markers)"
  else:
    markers = inventory.loc[inventory["is_unlabeled"], "marker_name"].tolist()
    title = "Missing data heatmap (unlabeled markers)"
  if not markers:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, "No markers in category", ha="center", va="center")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return

  max_markers = config["outputs"].get("max_markers_per_heatmap", 80)
  markers = markers[:max_markers]
  valid = session.valid_marker_frame.sel(marker=markers).values
  missing = (~valid).astype(float)
  frames = session.coordinates.coords["frame"].values
  max_frames = config["outputs"].get("heatmap_downsample_max_frames", 5000)
  downsample_note = ""
  if len(frames) > max_frames:
    step = int(np.ceil(len(frames) / max_frames))
    missing = missing[::step, :]
    frames = frames[::step]
    downsample_note = f" (frames downsampled every {step})"

  fig, ax = plt.subplots(figsize=(14, max(4, len(markers) * 0.2)))
  ax.imshow(missing.T, aspect="auto", interpolation="nearest", cmap="Reds", vmin=0, vmax=1)
  ax.set_xlabel(f"Frame index{downsample_note}")
  ax.set_ylabel("Marker")
  ax.set_yticks(range(len(markers)))
  ax.set_yticklabels(markers, fontsize=7)
  ax.set_title(title + downsample_note)
  fig.tight_layout()
  fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
  plt.close(fig)


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def _messages_to_dataframe(messages: list[QCMessage]) -> pd.DataFrame:
  return pd.DataFrame(
    [
      {
        "severity": m.severity,
        "code": m.code,
        "message": m.message,
        "context": str(m.context),
        "suggested_action": m.suggested_action,
      }
      for m in messages
    ]
  )


def _flatten_config(config: dict[str, Any]) -> pd.DataFrame:
  rows: list[dict[str, str]] = []

  def _walk(prefix: str, value: Any) -> None:
    if isinstance(value, dict):
      for key, child in value.items():
        if key.startswith("_"):
          continue
        _walk(f"{prefix}.{key}" if prefix else key, child)
    else:
      rows.append({"key": prefix, "value": str(value)})

  _walk("", config)
  return pd.DataFrame(rows)


def write_outputs(
  layer1_result: QCResult,
  layer2_result: QCResult,
  config: dict[str, Any],
  verbose: bool = False,
) -> list[Path]:
  base_dir = Path(config.get("_base_dir", Path(config["_config_path"]).parent))
  output_dir = _resolve_path(base_dir, config["paths"]["output_dir"])
  tables_dir = output_dir / "tables"
  tables_dir.mkdir(parents=True, exist_ok=True)
  written: list[Path] = []

  tables = layer2_result.tables
  messages = layer1_result.messages + layer2_result.messages

  if config["outputs"].get("write_csv_tables", True):
    for name, df in tables.items():
      path = tables_dir / f"{name}.csv"
      df.to_csv(path, index=False)
      written.append(path)
      if verbose:
        LOGGER.info("Wrote %s", path)

  if config["outputs"].get("write_config_used", True):
    config_path = output_dir / "config_used.yaml"
    clean_config = copy.deepcopy(config)
    clean_config.pop("_config_path", None)
    clean_config.pop("_base_dir", None)
    with config_path.open("w", encoding="utf-8") as handle:
      yaml.safe_dump(clean_config, handle, sort_keys=False)
    written.append(config_path)

  if config["outputs"].get("write_text_summary", True):
    summary_path = output_dir / "qc_report_summary.txt"
    _write_text_summary(summary_path, layer1_result, layer2_result, messages)
    written.append(summary_path)

  if config["outputs"].get("write_excel_workbook", True):
    excel_path = output_dir / "qc_report.xlsx"
    _write_excel_workbook(excel_path, tables, messages, config)
    written.append(excel_path)

  written.extend(layer2_result.figures.values())
  return written


def _write_text_summary(
  path: Path,
  layer1_result: QCResult,
  layer2_result: QCResult,
  messages: list[QCMessage],
) -> None:
  session = layer1_result.session
  assert session is not None
  md = session.metadata
  summary = layer2_result.tables["session_summary"].iloc[0]
  lines = [
    "Motive Raw Marker QC Summary (Layers 1-2)",
    "=" * 44,
    f"Input file: {md['input_file']}",
    f"Observed frames: {md['total_frames_observed']}",
    f"Effective frame rate (Hz): {md['effective_frame_rate_hz']}",
    f"Labeled markers: {md['n_labeled_markers']}",
    f"Unlabeled markers: {md['n_unlabeled_markers']}",
    f"Raw data status: {md['raw_data_status']}",
    "",
    "The exported file is consistent with raw reconstructed Motive marker XYZ data, "
    "with missing values preserved, according to the checks performed.",
    "No preprocessing (gap filling, smoothing, filtering, coordinate transforms) was applied by this script.",
    "",
    f"Labeled missing percent: {summary['missing_percent_labeled']}",
    f"Total gaps (all markers): {summary['n_gaps_total_all']}",
    f"Labeled gaps >= 0.2 s: {summary['n_gaps_ge_0p2s_labeled']}",
    f"Labeled gaps >= 0.5 s: {summary['n_gaps_ge_0p5s_labeled']}",
    f"Raw QC preprocessing status: {summary['raw_qc_preprocessing_status']}",
    f"Reason: {summary['raw_qc_status_reason']}",
    "",
    f"Validation status: {md['validation_status']}",
    f"Warnings: {md['n_warnings']}",
    f"Errors: {md['n_errors']}",
  ]
  if messages:
    lines.append("")
    lines.append("Messages:")
    for msg in messages:
      lines.append(f"  [{msg.severity}] {msg.code}: {msg.message}")
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_excel_workbook(
  path: Path,
  tables: dict[str, pd.DataFrame],
  messages: list[QCMessage],
  config: dict[str, Any],
) -> None:
  sheet_map = {
    "session_summary": "session_summary",
    "marker_inventory": "marker_inventory",
    "marker_quality_summary": "marker_quality",
    "gap_events": "gap_events",
    "gap_summary_by_marker": "gap_by_marker",
    "gap_summary_by_group": "gap_by_group",
  }
  with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
    for table_key, sheet_name in sheet_map.items():
      if table_key in tables:
        tables[table_key].to_excel(writer, sheet_name=sheet_name, index=False)
    _messages_to_dataframe(messages).to_excel(writer, sheet_name="validation_messages", index=False)
    _flatten_config(config).to_excel(writer, sheet_name="config_summary", index=False)


def write_validation_log(
  layer1_result: QCResult,
  layer2_result: QCResult,
  config: dict[str, Any],
  log_path: str | Path = "docs/VALIDATION_LOG.md",
  decision: str = "pending",
  validated_by: str = "",
  notes: str = "",
) -> Path:
  base_dir = Path(config.get("_base_dir", Path(config["_config_path"]).parent))
  path = _resolve_path(base_dir, log_path)
  path.parent.mkdir(parents=True, exist_ok=True)
  session = layer1_result.session
  assert session is not None
  md = session.metadata
  summary = layer2_result.tables["session_summary"].iloc[0]
  content = f"""# Validation Log

## v0.2 - Layers 1-2

Input file: {md['input_file']}
Date run: {datetime.now().isoformat(timespec='seconds')}
Motive version recorded: {md['motive_version']}
Expected frame count: {md['total_frames_metadata']}
Observed frame count: {md['total_frames_observed']}
Expected frame rate: {md['export_frame_rate_hz']}
Observed/effective frame rate: {md['effective_frame_rate_hz']}
Expected marker count:
Observed marker count: {md['n_marker_triplets_total']}
Expected major gaps:
Observed major gaps (labeled >= 0.5 s): {summary['n_gaps_ge_0p5s_labeled']}
Layer 1 decision: pending
Layer 2 decision: pending
Validated by: {validated_by}
Notes: {notes}
Decision: {decision}
"""
  path.write_text(content, encoding="utf-8")
  return path


# ---------------------------------------------------------------------------
# Notebook helpers
# ---------------------------------------------------------------------------


def display_layer1_outputs(result: QCResult) -> dict[str, pd.DataFrame]:
  return {
    "session_summary": result.tables["session_summary"],
    "marker_inventory": result.tables["marker_inventory"],
  }


def display_layer2_outputs(result: QCResult) -> dict[str, Any]:
  top_n = 20
  gap_events = result.tables.get("gap_events", pd.DataFrame())
  longest_gaps = (
    gap_events.sort_values("duration_seconds", ascending=False).head(top_n)
    if not gap_events.empty
    else gap_events
  )
  return {
    "session_summary": result.tables["session_summary"],
    "marker_quality_summary": result.tables["marker_quality_summary"],
    "gap_events_top": longest_gaps,
    "gap_summary_by_marker": result.tables["gap_summary_by_marker"],
    "gap_summary_by_group": result.tables["gap_summary_by_group"],
    "figures": result.figures,
  }


def run_layers_1_2(config: dict[str, Any], verbose: bool = False) -> tuple[QCResult, QCResult, list[Path]]:
  layer1 = run_layer1_parse(config, verbose=verbose)
  if layer1.status == "fail":
    raise QCValidationError("Layer 1 failed validation; Layer 2 was not run.")
  layer2 = run_layer2_gaps(layer1.session, config, verbose=verbose)
  files = write_outputs(layer1, layer2, config, verbose=verbose)
  return layer1, layer2, files


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _configure_logging(verbose: bool) -> None:
  level = logging.INFO if verbose else logging.WARNING
  logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="Motive raw marker QC (Layers 1-2)")
  parser.add_argument("--config", required=True, help="Path to config.yaml")
  parser.add_argument("--dry-run", action="store_true", help="Validate config and parse metadata only")
  parser.add_argument("--verbose", action="store_true", help="Print progress messages")
  args = parser.parse_args(argv)

  _configure_logging(args.verbose)
  config_path = Path(args.config).resolve()
  config = load_config(config_path)
  config["_base_dir"] = config_path.parent

  try:
    if args.dry_run:
      input_path = _resolve_path(config_path.parent, config["paths"]["input_csv"])
      if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
      metadata_row, header_rows, _ = _read_csv_header(input_path)
      header_map = _find_header_rows(header_rows)
      messages: list[QCMessage] = []
      marker_records, _ = _build_marker_columns(header_rows, header_map, config, messages)
      print(f"Dry run OK: {input_path.name}")
      print(f"  Markers found: {len(marker_records)}")
      print(f"  Capture rate: {metadata_row.get('Capture Frame Rate')}")
      print(f"  Export rate: {metadata_row.get('Export Frame Rate')}")
      return 0

    layer1, layer2, files = run_layers_1_2(config, verbose=args.verbose)
    if args.verbose:
      print(f"Wrote {len(files)} output files to {config['paths']['output_dir']}")
    return 0 if layer2.status != "fail" else 1
  except Exception as exc:
    LOGGER.error("%s: %s", type(exc).__name__, exc)
    if args.verbose:
      raise
    return 1


if __name__ == "__main__":
  sys.exit(main())
