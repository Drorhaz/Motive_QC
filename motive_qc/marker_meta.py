"""Marker naming, grouping, and header parsing helpers."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from motive_qc.core import MotiveCSVParseError, QCMessage, SchemaValidationError


def parse_metadata_row(row: list[str]) -> dict[str, str]:
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


def header_row_label(row: list[str]) -> str:
    if not row:
        return ""
    if row[0].strip():
        return row[0].strip()
    if len(row) > 1:
        return row[1].strip()
    return ""


def find_header_rows(rows: list[list[str]]) -> dict[str, int]:
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
        label = header_row_label(row)
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


def parse_subject_prefix(marker_name: str) -> tuple[str | None, str]:
    if ":" in marker_name:
        prefix, short = marker_name.split(":", 1)
        return prefix, short
    return None, marker_name


def parse_marker_identity(marker_name: str) -> tuple[str, str]:
    """Return ``(skeleton_prefix, canonical_short_name)`` for duplicate-skeleton logic.

    Handles Motive export naming variants in one file, e.g.::

        671:BackLeft        -> ('671', 'BackLeft')
        T3_671:LThighFront  -> ('T3_671', 'LThighFront')
        FKA-671_BackLeft    -> ('FKA-671', 'BackLeft')
    """
    name = marker_name.strip()
    if ":" in name:
        skel, short = name.split(":", 1)
        return skel.strip(), short.strip()
    if "_" in name:
        skel, short = name.split("_", 1)
        skel, short = skel.strip(), short.strip()
        if short and short[0].isupper() and not short.lower().startswith("unlabeled"):
            return skel, short
    return "", name


def is_unlabeled_marker(name: str, config: dict[str, Any] | None = None) -> bool:
    short_name = parse_subject_prefix(name.strip())[1].strip()
    if re.match(r"(?i)^unlabeled(?:[\s_]*\d+)?$", short_name):
        return True
    if config:
        patterns = config.get("markers", {}).get("unlabeled_name_patterns", [])
        for pattern in patterns:
            if re.match(pattern, short_name, re.IGNORECASE):
                return True
    return False


def assign_body_region(
    marker_name: str, marker_groups: dict[str, Any], config: dict[str, Any]
) -> str:
    short_name = parse_subject_prefix(marker_name)[1]
    if is_unlabeled_marker(marker_name, config):
        return "unlabeled"
    for group_name, group_cfg in marker_groups.items():
        if group_name == "unclassified":
            continue
        keywords = group_cfg.get("keywords", [])
        for keyword in keywords:
            if keyword and keyword.lower() in short_name.lower():
                return group_name
    return "unclassified"


def read_csv_header(path: Path) -> tuple[dict[str, str], list[list[str]], int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        metadata_row = next(reader)
        metadata = parse_metadata_row(metadata_row)
        header_rows: list[list[str]] = []
        data_start_idx = 1
        for row in reader:
            data_start_idx += 1
            if not any(cell.strip() for cell in row):
                continue
            label = header_row_label(row)
            if label == "Frame":
                header_rows.append(row)
                break
            header_rows.append(row)
        else:
            raise MotiveCSVParseError("Could not find axis header row starting with Frame.")

    return metadata, header_rows, data_start_idx


def build_marker_columns(
    header_rows: list[list[str]],
    header_map: dict[str, int],
    config: dict[str, Any],
    messages: list[QCMessage],
) -> tuple[list[dict[str, Any]], dict[str, set[str]], int]:
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
    skipped_marker_columns: list[dict[str, Any]] = []
    excluded_non_marker_column_count = 0
    duplicate_axis_markers: set[str] = set()

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
            excluded_non_marker_column_count += 1
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
                excluded_non_marker_column_count += 1
                non_marker_types["quaternion"].add(channel)
                continue

        if axis not in ("X", "Y", "Z"):
            skipped_marker_columns.append(
                {
                    "column_index": col,
                    "marker_name": marker_name,
                    "marker_type": marker_type,
                    "axis": axis,
                    "channel": channel,
                    "reason": "invalid_axis",
                }
            )
            continue

        if marker_name not in markers:
            prefix, short_name = parse_subject_prefix(marker_name)
            skeleton_prefix, canonical_short_name = parse_marker_identity(marker_name)
            is_unlabeled = is_unlabeled_marker(marker_name, config)
            markers[marker_name] = {
                "marker_name": marker_name,
                "marker_short_name": short_name,
                "skeleton_prefix": skeleton_prefix,
                "canonical_short_name": canonical_short_name,
                "subject_or_asset_prefix": prefix,
                "is_labeled": not is_unlabeled,
                "is_unlabeled": is_unlabeled,
                "marker_type_raw": marker_type or "Marker",
                "body_region_group": assign_body_region(
                    marker_name, config["marker_groups"], config
                ),
                "axes": {},
                "duplicate_axes": set(),
            }
        if axis in markers[marker_name]["axes"]:
            duplicate_axis_markers.add(marker_name)
            markers[marker_name]["duplicate_axes"].add(axis)
        else:
            markers[marker_name]["axes"][axis] = col

    marker_records: list[dict[str, Any]] = []
    duplicate_names: list[str] = []
    for marker_name, info in sorted(markers.items()):
        axes = info["axes"]
        has_x, has_y, has_z = "X" in axes, "Y" in axes, "Z" in axes
        if marker_name in duplicate_axis_markers or len(axes) > 3:
            duplicate_names.append(marker_name)
        parse_status = "ok"
        if marker_name in duplicate_axis_markers or info.get("duplicate_axes"):
            parse_status = "duplicate_axis"
        elif not (has_x and has_y and has_z):
            parse_status = "missing_axis"
        marker_records.append(
            {
                "marker_name": marker_name,
                "marker_short_name": info["marker_short_name"],
                "skeleton_prefix": info["skeleton_prefix"],
                "canonical_short_name": info["canonical_short_name"],
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
    if non_marker_types["other"]:
        messages.append(
            QCMessage(
                "WARNING",
                "OTHER_NON_MARKER_COLUMNS",
                "Other non-marker column types detected and excluded from raw marker QC.",
                {"types": sorted(non_marker_types["other"])},
            )
        )
    if skipped_marker_columns:
        messages.append(
            QCMessage(
                "WARNING",
                "SKIPPED_MARKER_COLUMNS",
                f"{len(skipped_marker_columns)} marker-named columns skipped due to invalid axis labels.",
                {"examples": skipped_marker_columns[:10]},
            )
        )
    if excluded_non_marker_column_count:
        messages.append(
            QCMessage(
                "INFO",
                "EXCLUDED_NON_MARKER_COLUMNS",
                f"{excluded_non_marker_column_count} solved/non-marker columns excluded from raw marker QC.",
                {"count": excluded_non_marker_column_count},
            )
        )

    return marker_records, non_marker_types, excluded_non_marker_column_count


def filter_markers(marker_records: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
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


def metadata_float(metadata: dict[str, str], key: str) -> float | None:
    value = metadata.get(key)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def metadata_int(metadata: dict[str, str], key: str) -> int | None:
    value = metadata.get(key)
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None
