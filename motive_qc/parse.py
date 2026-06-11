"""Layer 1: parse Motive raw CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from motive_qc.core import (
    LOGGER,
    MotiveCSVParseError,
    MotiveSession,
    QCMessage,
    QCResult,
    QCValidationError,
    SchemaValidationError,
    resolve_path,
)
from motive_qc.marker_meta import (
    build_marker_columns,
    filter_markers,
    find_header_rows,
    metadata_float,
    metadata_int,
    read_csv_header,
)

def run_layer1_parse(config: dict[str, Any], verbose: bool = False) -> QCResult:
  messages: list[QCMessage] = []
  base_dir = Path(config.get("_base_dir", Path(config["_config_path"]).parent))

  input_path = resolve_path(base_dir, config["paths"]["input_csv"])
  if not input_path.exists():
    raise FileNotFoundError(f"Input CSV not found: {input_path}")

  if verbose:
    LOGGER.info("Parsing %s", input_path)

  metadata_row, header_rows, data_start_idx = read_csv_header(input_path)
  header_map = find_header_rows(header_rows)
  marker_records, non_marker_types, excluded_non_marker_count = build_marker_columns(
    header_rows, header_map, config, messages
  )
  marker_records = filter_markers(marker_records, config)

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

  capture_rate = metadata_float(metadata_row, "Capture Frame Rate")
  export_rate = metadata_float(metadata_row, "Export Frame Rate")
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
    "total_frames_metadata": metadata_int(metadata_row, "Total Exported Frames")
    or metadata_int(metadata_row, "Total Frames in Take"),
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
    "excluded_non_marker_column_count": excluded_non_marker_count,
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

  session_summary = build_layer1_session_summary(session)
  status = validation_status
  return QCResult(
    layer_name="layer1",
    status=status,
    tables={"session_summary": session_summary, "marker_inventory": marker_inventory},
    messages=messages,
    session=session,
  )


def build_layer1_session_summary(session: MotiveSession) -> pd.DataFrame:
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
