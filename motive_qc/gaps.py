"""Layer 2: gaps, masks, unlabeled burden."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from motive_qc.core import LOGGER, MotiveSession, QCMessage, QCResult
from motive_qc.parse import build_layer1_session_summary
from motive_qc.plots import generate_layer2_plots

def seconds_to_key(seconds: float) -> str:
  return f"{seconds:.3f}".rstrip("0").rstrip(".").replace(".", "p") + "s"


def gap_threshold_labels(config: dict[str, Any]) -> list[tuple[str, float]]:
  thresholds = config["gaps"]["thresholds_seconds"]
  ordered = [
    ("tiny", thresholds.get("tiny_gap", 0.025)),
    ("minor", thresholds.get("minor_gap", 0.1)),
    ("moderate", thresholds.get("moderate_gap", 0.2)),
    ("large", thresholds.get("large_gap", 0.5)),
    ("severe", thresholds.get("severe_gap", 1.0)),
  ]
  return ordered


def crossed_thresholds(duration_seconds: float, config: dict[str, Any]) -> list[str]:
  use_ge = config["gaps"].get("use_greater_equal_thresholds", True)
  crossed = []
  for label, threshold in gap_threshold_labels(config):
    if use_ge:
      if duration_seconds >= threshold:
        crossed.append(label)
    else:
      if duration_seconds > threshold:
        crossed.append(label)
  return crossed


def severity_label(duration_frames: int, duration_seconds: float, config: dict[str, Any]) -> str:
  if duration_frames == 1:
    return "single_frame"
  crossed = crossed_thresholds(duration_seconds, config)
  if not crossed:
    return "tiny"
  return crossed[-1]


def recommended_status(severity: str, touches_edge: bool) -> str:
  if severity in ("severe", "large") or (touches_edge and severity in ("moderate", "large", "severe")):
    return "potential_exclusion"
  if severity in ("moderate", "large"):
    return "caution"
  return "document"


def count_gaps_ge(gap_durations_seconds: list[float], threshold: float, config: dict[str, Any]) -> int:
  use_ge = config["gaps"].get("use_greater_equal_thresholds", True)
  if use_ge:
    return sum(1 for value in gap_durations_seconds if value >= threshold)
  return sum(1 for value in gap_durations_seconds if value > threshold)


def marker_quality_label(
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


def detect_gaps_for_marker(
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
    crossed = crossed_thresholds(duration_seconds, config)
    severity = severity_label(duration_frames, duration_seconds, config)
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
        "recommended_status": recommended_status(severity, touches_edge),
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
    gaps, gap_id = detect_gaps_for_marker(
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
    n_large = count_gaps_ge(gap_durations_seconds, thresholds["large_gap"], config)

    missing_idx = np.where(~valid)[0]
    first_missing = int(frames[missing_idx[0]]) if len(missing_idx) else None
    last_missing = int(frames[missing_idx[-1]]) if len(missing_idx) else None

    quality_label, quality_reason = marker_quality_label(
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
        "n_gaps_ge_0p025s": count_gaps_ge(gap_durations_seconds, thresholds["tiny_gap"], config),
        "n_gaps_ge_0p1s": count_gaps_ge(gap_durations_seconds, thresholds["minor_gap"], config),
        "n_gaps_ge_0p2s": count_gaps_ge(gap_durations_seconds, thresholds["moderate_gap"], config),
        "n_gaps_ge_0p5s": count_gaps_ge(gap_durations_seconds, thresholds["large_gap"], config),
        "n_gaps_ge_1p0s": count_gaps_ge(gap_durations_seconds, thresholds["severe_gap"], config),
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

  gap_summary_by_group = build_gap_summary_by_group(marker_quality_summary, gap_events)
  session_summary = update_session_summary_layer2(
    session, marker_quality_summary, gap_events, config
  )
  unlabeled_marker_summary, unlabeled_frame_counts = build_unlabeled_summary(
    session, gap_events, config
  )
  frame_qc_mask = build_frame_qc_mask(session, gap_events, config)

  figures = generate_layer2_plots(
    session,
    marker_quality_summary,
    gap_events,
    config,
    unlabeled_frame_counts=unlabeled_frame_counts,
  )

  status = session.metadata.get("validation_status", "pass")
  tables: dict[str, pd.DataFrame] = {
    "session_summary": session_summary,
    "marker_inventory": session.marker_inventory,
    "marker_quality_summary": marker_quality_summary,
    "gap_events": gap_events,
    "gap_summary_by_marker": gap_summary_by_marker,
    "gap_summary_by_group": gap_summary_by_group,
    "unlabeled_marker_summary": unlabeled_marker_summary,
    "frame_qc_mask": frame_qc_mask,
  }
  if config["outputs"].get("write_unlabeled_frame_counts", True):
    tables["unlabeled_frame_counts"] = unlabeled_frame_counts

  return QCResult(
    layer_name="layer2",
    status=status,
    tables=tables,
    figures=figures,
    messages=messages,
    session=session,
  )


def build_gap_summary_by_group(
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


def session_gap_timeline_metrics(
  gap_events: pd.DataFrame,
  config: dict[str, Any],
  frame_rate: float,
) -> dict[str, float]:
  moderate_thr = config["gaps"]["thresholds_seconds"]["moderate_gap"]
  large_thr = config["gaps"]["thresholds_seconds"]["large_gap"]
  critical_groups = set(config.get("frame_quality", {}).get("critical_groups", []))

  empty = {
    "union_gap_seconds_ge_0p2": 0.0,
    "longest_merged_gap_run_seconds": 0.0,
    "max_critical_gap_seconds": 0.0,
    "n_gaps_ge_0p5": 0,
  }
  if gap_events.empty or frame_rate <= 0:
    return empty

  labeled_mod = gap_events[
    gap_events["is_labeled"] & (gap_events["duration_seconds"] >= moderate_thr)
  ]
  labeled_all = gap_events[gap_events["is_labeled"]]

  frames_union: set[int] = set()
  intervals: list[tuple[int, int]] = []
  for _, gap in labeled_mod.iterrows():
    start = int(gap["gap_start_frame"])
    end = int(gap["gap_end_frame"])
    frames_union.update(range(start, end + 1))
    intervals.append((start, end))

  union_seconds = len(frames_union) / frame_rate

  longest_merged_frames = 0
  if intervals:
    intervals.sort()
    merged_start, merged_end = intervals[0]
    for start, end in intervals[1:]:
      if start <= merged_end + 1:
        merged_end = max(merged_end, end)
      else:
        longest_merged_frames = max(longest_merged_frames, merged_end - merged_start + 1)
        merged_start, merged_end = start, end
    longest_merged_frames = max(longest_merged_frames, merged_end - merged_start + 1)

  critical_gaps = labeled_mod[labeled_mod["body_region_group"].isin(critical_groups)]
  max_critical = (
    float(critical_gaps["duration_seconds"].max()) if not critical_gaps.empty else 0.0
  )
  n_ge_05 = (
    count_gaps_ge(labeled_all["duration_seconds"].tolist(), large_thr, config)
    if not labeled_all.empty
    else 0
  )

  return {
    "union_gap_seconds_ge_0p2": round(union_seconds, 6),
    "longest_merged_gap_run_seconds": round(longest_merged_frames / frame_rate, 6),
    "max_critical_gap_seconds": round(max_critical, 6),
    "n_gaps_ge_0p5": n_ge_05,
  }


def evaluate_preprocessing_status(
  missing_percent_labeled: float,
  gap_metrics: dict[str, float],
  config: dict[str, Any],
) -> tuple[str, str]:
  session_cfg = config["quality_labels"]["session"]
  acceptable = session_cfg["acceptable_for_preprocessing"]
  caution = session_cfg["caution_for_preprocessing"]
  poor_if = session_cfg.get(
    "poor_if",
    {
      "missing_percent_above": 10.0,
      "union_gap_seconds_ge_0p2_above": 10.0,
      "critical_single_gap_seconds_above": 5.0,
    },
  )

  union_s = gap_metrics["union_gap_seconds_ge_0p2"]
  max_crit_s = gap_metrics["max_critical_gap_seconds"]
  n_large = int(gap_metrics["n_gaps_ge_0p5"])
  merged_s = gap_metrics["longest_merged_gap_run_seconds"]

  if missing_percent_labeled > poor_if.get("missing_percent_above", 10.0):
    return (
      "poor",
      f"Labeled missingness {missing_percent_labeled}% exceeds poor threshold "
      f"({poor_if.get('missing_percent_above', 10.0)}%).",
    )
  if union_s >= poor_if.get("union_gap_seconds_ge_0p2_above", 10.0):
    return (
      "poor",
      f"Union labeled gap time (>= moderate) is {union_s}s, at or above poor threshold "
      f"({poor_if.get('union_gap_seconds_ge_0p2_above', 10.0)}s).",
    )
  if max_crit_s >= poor_if.get("critical_single_gap_seconds_above", 5.0):
    return (
      "poor",
      f"Critical-region labeled gap up to {max_crit_s}s exceeds poor threshold "
      f"({poor_if.get('critical_single_gap_seconds_above', 5.0)}s).",
    )

  acc_union = acceptable.get("max_union_gap_seconds_ge_0p2", 5.0)
  acc_crit = acceptable.get("max_critical_single_gap_seconds", 1.0)
  if (
    missing_percent_labeled <= acceptable["max_labeled_missing_percent"]
    and union_s <= acc_union
    and max_crit_s < acc_crit
  ):
    return (
      "acceptable",
      "Labeled missingness and timeline gap burden within acceptable preprocessing thresholds.",
    )

  caut_union = caution.get("max_union_gap_seconds_ge_0p2", 10.0)
  caut_crit = caution.get("max_critical_single_gap_seconds", 5.0)
  if (
    missing_percent_labeled <= caution["max_labeled_missing_percent"]
    and union_s <= caut_union
    and max_crit_s < caut_crit
  ):
    parts = []
    if max_crit_s >= acc_crit:
      parts.append(f"critical-region gap up to {max_crit_s}s")
    if union_s > acc_union:
      parts.append(f"union moderate+ gap time {union_s}s")
    if n_large > 0:
      parts.append(f"{n_large} labeled gaps >= 0.5s")
    detail = "; ".join(parts) if parts else "borderline gap metrics"
    return (
      "caution",
      f"Review before preprocessing: {detail} (merged run up to {merged_s}s).",
    )

  return (
    "caution",
    f"Gap metrics require review: union time {union_s}s, max critical gap {max_crit_s}s, "
    f"{n_large} gaps >= 0.5s.",
  )


def update_session_summary_layer2(
  session: MotiveSession,
  marker_quality: pd.DataFrame,
  gap_events: pd.DataFrame,
  config: dict[str, Any],
) -> pd.DataFrame:
  summary = build_layer1_session_summary(session).iloc[0].to_dict()
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
  n_ge_02 = count_gaps_ge(
    labeled_gaps["duration_seconds"].tolist(), thresholds["moderate_gap"], config
  ) if not labeled_gaps.empty else 0
  n_ge_05 = count_gaps_ge(
    labeled_gaps["duration_seconds"].tolist(), thresholds["large_gap"], config
  ) if not labeled_gaps.empty else 0
  n_ge_10 = count_gaps_ge(
    labeled_gaps["duration_seconds"].tolist(), thresholds["severe_gap"], config
  ) if not labeled_gaps.empty else 0

  longest_marker = None
  longest_seconds = 0.0
  if not labeled_gaps.empty:
    idx = labeled_gaps["duration_seconds"].idxmax()
    longest_marker = labeled_gaps.loc[idx, "marker_name"]
    longest_seconds = float(labeled_gaps.loc[idx, "duration_seconds"])

  frame_rate = float(session.metadata["effective_frame_rate_hz"])
  gap_metrics = session_gap_timeline_metrics(gap_events, config, frame_rate)
  raw_status, raw_reason = evaluate_preprocessing_status(pct_lab, gap_metrics, config)

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
      "union_gap_seconds_ge_0p2_labeled": gap_metrics["union_gap_seconds_ge_0p2"],
      "longest_merged_gap_run_seconds": gap_metrics["longest_merged_gap_run_seconds"],
      "max_critical_gap_seconds_labeled": gap_metrics["max_critical_gap_seconds"],
      "raw_qc_preprocessing_status": raw_status,
      "raw_qc_status_reason": raw_reason,
    }
  )
  return pd.DataFrame([summary])


def frame_qc_mask_config(config: dict[str, Any]) -> dict[str, Any]:
  defaults = {
    "enabled": True,
    "caution_missing_labeled_percent": 10.0,
    "exclude_missing_labeled_percent": 20.0,
    "flag_moderate_gap": True,
    "flag_large_gap": True,
    "flag_unlabeled_present": False,
  }
  user = config.get("frame_qc_mask", {})
  return {**defaults, **user}


def build_unlabeled_summary(
  session: MotiveSession,
  gap_events: pd.DataFrame,
  config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
  inventory = session.marker_inventory
  unlabeled_markers = inventory.loc[inventory["is_unlabeled"], "marker_name"].tolist()
  frames = session.coordinates.coords["frame"].values
  n_frames = len(frames)
  frame_rate = float(session.metadata["effective_frame_rate_hz"])
  time_values = session.time_seconds.values

  empty_summary = {
    "total_unlabeled_tracks": 0,
    "frames_with_any_unlabeled": 0,
    "percent_frames_with_any_unlabeled": 0.0,
    "max_unlabeled_markers_in_frame": 0,
    "frame_of_max_unlabeled_count": None,
    "unlabeled_bursts_count": 0,
    "longest_unlabeled_burst_sec": 0.0,
    "overlap_with_labeled_gaps": False,
  }
  if not unlabeled_markers or n_frames == 0:
    return pd.DataFrame([empty_summary]), pd.DataFrame(
      columns=["frame", "time_seconds", "unlabeled_count", "labeled_missing_count", "overlap_flag"]
    )

  unlabeled_valid = session.valid_marker_frame.sel(marker=unlabeled_markers).values
  unlabeled_count_per_frame = unlabeled_valid.sum(axis=1).astype(int)
  any_unlabeled = unlabeled_count_per_frame > 0

  labeled_markers = inventory.loc[inventory["is_labeled"], "marker_name"].tolist()
  if labeled_markers:
    labeled_valid = session.valid_marker_frame.sel(marker=labeled_markers).values
    labeled_missing_per_frame = (~labeled_valid).sum(axis=1).astype(int)
  else:
    labeled_missing_per_frame = np.zeros(n_frames, dtype=int)

  bursts = 0
  longest_burst_frames = 0
  idx = 0
  while idx < n_frames:
    if not any_unlabeled[idx]:
      idx += 1
      continue
    start = idx
    while idx < n_frames and any_unlabeled[idx]:
      idx += 1
    bursts += 1
    longest_burst_frames = max(longest_burst_frames, idx - start)

  frames_with_unlabeled = int(any_unlabeled.sum())
  max_count = int(unlabeled_count_per_frame.max())
  max_frame_idx = int(np.argmax(unlabeled_count_per_frame))
  max_frame = int(frames[max_frame_idx]) if max_count > 0 else None
  overlap = bool(np.any(any_unlabeled & (labeled_missing_per_frame > 0)))

  summary = pd.DataFrame(
    [
      {
        "total_unlabeled_tracks": len(unlabeled_markers),
        "frames_with_any_unlabeled": frames_with_unlabeled,
        "percent_frames_with_any_unlabeled": round(
          100.0 * frames_with_unlabeled / n_frames, 6
        ),
        "max_unlabeled_markers_in_frame": max_count,
        "frame_of_max_unlabeled_count": max_frame,
        "unlabeled_bursts_count": bursts,
        "longest_unlabeled_burst_sec": round(longest_burst_frames / frame_rate, 6),
        "overlap_with_labeled_gaps": overlap,
      }
    ]
  )

  frame_counts = pd.DataFrame(
    {
      "frame": frames,
      "time_seconds": time_values,
      "unlabeled_count": unlabeled_count_per_frame,
      "labeled_missing_count": labeled_missing_per_frame,
      "overlap_flag": any_unlabeled & (labeled_missing_per_frame > 0),
    }
  )
  return summary, frame_counts


def build_frame_qc_mask(
  session: MotiveSession,
  gap_events: pd.DataFrame,
  config: dict[str, Any],
) -> pd.DataFrame:
  mask_cfg = frame_qc_mask_config(config)
  if not mask_cfg["enabled"]:
    return pd.DataFrame()

  frames = session.coordinates.coords["frame"].values
  n_frames = len(frames)
  time_values = session.time_seconds.values
  inventory = session.marker_inventory

  labeled_markers = inventory.loc[inventory["is_labeled"], "marker_name"].tolist()
  unlabeled_markers = inventory.loc[inventory["is_unlabeled"], "marker_name"].tolist()

  if labeled_markers:
    labeled_valid = session.valid_marker_frame.sel(marker=labeled_markers).values
    missing_labeled_count = (~labeled_valid).sum(axis=1).astype(int)
    missing_labeled_percent = 100.0 * missing_labeled_count / len(labeled_markers)
  else:
    missing_labeled_count = np.zeros(n_frames, dtype=int)
    missing_labeled_percent = np.zeros(n_frames, dtype=float)

  if unlabeled_markers:
    unlabeled_valid = session.valid_marker_frame.sel(marker=unlabeled_markers).values
    unlabeled_count = unlabeled_valid.sum(axis=1).astype(int)
    unlabeled_present = unlabeled_count > 0
  else:
    unlabeled_count = np.zeros(n_frames, dtype=int)
    unlabeled_present = np.zeros(n_frames, dtype=bool)

  thresholds = config["gaps"]["thresholds_seconds"]
  moderate_thr = thresholds["moderate_gap"]
  large_thr = thresholds["large_gap"]
  moderate_gap_present = np.zeros(n_frames, dtype=bool)
  large_gap_present = np.zeros(n_frames, dtype=bool)

  labeled_gaps = (
    gap_events[gap_events["is_labeled"]] if not gap_events.empty else pd.DataFrame()
  )
  for _, gap in labeled_gaps.iterrows():
    in_gap = (frames >= gap["gap_start_frame"]) & (frames <= gap["gap_end_frame"])
    if gap["duration_seconds"] >= moderate_thr:
      moderate_gap_present |= in_gap
    if gap["duration_seconds"] >= large_thr:
      large_gap_present |= in_gap

  qc_status_list: list[str] = []
  reason_codes_list: list[str] = []
  for i in range(n_frames):
    reasons: list[str] = []
    status = "use"
    if large_gap_present[i] and mask_cfg["flag_large_gap"]:
      reasons.append("LARGE_GAP")
      status = "exclude_or_review"
    elif moderate_gap_present[i] and mask_cfg["flag_moderate_gap"]:
      reasons.append("MODERATE_GAP")
      status = "caution"
    if missing_labeled_percent[i] >= mask_cfg["exclude_missing_labeled_percent"]:
      reasons.append("HIGH_MISSING_LABELED")
      status = "exclude_or_review"
    elif missing_labeled_percent[i] >= mask_cfg["caution_missing_labeled_percent"]:
      reasons.append("ELEVATED_MISSING_LABELED")
      if status == "use":
        status = "caution"
    if unlabeled_present[i] and mask_cfg["flag_unlabeled_present"]:
      reasons.append("UNLABELED_PRESENT")
      if status == "use":
        status = "caution"
    qc_status_list.append(status)
    reason_codes_list.append(";".join(reasons))

  return pd.DataFrame(
    {
      "frame": frames,
      "time_seconds": time_values,
      "missing_labeled_count": missing_labeled_count,
      "missing_labeled_percent": np.round(missing_labeled_percent, 6),
      "moderate_gap_present": moderate_gap_present,
      "large_gap_present": large_gap_present,
      "unlabeled_present": unlabeled_present,
      "unlabeled_count": unlabeled_count,
      "qc_status": qc_status_list,
      "reason_codes": reason_codes_list,
    }
  )
