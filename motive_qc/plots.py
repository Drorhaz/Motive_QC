"""Plotting helpers for Layers 2-5."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from motive_qc.core import MotiveSession, plot_dir_from_config

def generate_layer2_plots(
  session: MotiveSession,
  marker_quality: pd.DataFrame,
  gap_events: pd.DataFrame,
  config: dict[str, Any],
  unlabeled_frame_counts: pd.DataFrame | None = None,
) -> dict[str, Path]:
  outputs_cfg = config["outputs"]
  if not outputs_cfg.get("plots", {}).get("enabled", True):
    return {}

  plot_dir = plot_dir_from_config(config)
  dpi = outputs_cfg.get("dpi", 300)
  fmt = outputs_cfg.get("plot_format", "png")
  figures: dict[str, Path] = {}

  if outputs_cfg["plots"].get("marker_completeness", True):
    path = plot_dir / f"marker_completeness.{fmt}"
    plot_marker_completeness(marker_quality, path, dpi)
    figures["marker_completeness"] = path

  if outputs_cfg["plots"].get("gap_duration_histogram", True):
    path = plot_dir / f"gap_duration_histogram.{fmt}"
    plot_gap_duration_histogram(gap_events, config, path, dpi)
    figures["gap_duration_histogram"] = path

  if outputs_cfg["plots"].get("missing_data_heatmap_labeled", True):
    path = plot_dir / f"missing_data_heatmap_labeled.{fmt}"
    plot_missing_heatmap(
      session, labeled_only=True, config=config, output_path=path, dpi=dpi, gap_events=gap_events
    )
    figures["missing_data_heatmap_labeled"] = path

  if (
    outputs_cfg["plots"].get("missing_data_heatmap_unlabeled", False)
    and marker_quality["is_unlabeled"].any()
  ):
    path = plot_dir / f"missing_data_heatmap_unlabeled.{fmt}"
    plot_missing_heatmap(session, labeled_only=False, config=config, output_path=path, dpi=dpi)
    figures["missing_data_heatmap_unlabeled"] = path

  if outputs_cfg["plots"].get("gap_timeline", True):
    path = plot_dir / f"gap_timeline.{fmt}"
    plot_gap_timeline(gap_events, config, path, dpi)
    figures["gap_timeline"] = path

  if outputs_cfg["plots"].get("unlabeled_count_over_time", True):
    path = plot_dir / f"unlabeled_count_over_time.{fmt}"
    plot_unlabeled_count_over_time(
      unlabeled_frame_counts, gap_events, config, path, dpi
    )
    figures["unlabeled_count_over_time"] = path

  if outputs_cfg["plots"].get("labeled_missing_count_over_time", True):
    path = plot_dir / f"labeled_missing_count_over_time.{fmt}"
    plot_labeled_missing_count_over_time(
      unlabeled_frame_counts, gap_events, config, path, dpi
    )
    figures["labeled_missing_count_over_time"] = path

  return figures


def plot_marker_completeness(marker_quality: pd.DataFrame, output_path: Path, dpi: int) -> None:
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


def plot_gap_duration_histogram(
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


def plot_missing_heatmap(
  session: MotiveSession,
  labeled_only: bool,
  config: dict[str, Any],
  output_path: Path,
  dpi: int,
  gap_events: pd.DataFrame | None = None,
) -> None:
  inventory = session.marker_inventory
  if labeled_only:
    if "included_in_analysis" in inventory.columns:
      markers = inventory.loc[inventory["included_in_analysis"].astype(bool), "marker_name"].tolist()
      title = "Missing data heatmap (in-analysis labeled markers)"
    else:
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

  matrix = build_gap_severity_matrix(session, markers, config, gap_events)
  frames = session.coordinates.coords["frame"].values
  max_frames = config["outputs"].get("heatmap_downsample_max_frames", 5000)
  downsample_note = ""
  if len(frames) > max_frames:
    step = int(np.ceil(len(frames) / max_frames))
    downsample_note = f" (frames downsampled every {step})"
  matrix, _ = _downsample_matrix_max(matrix, max_frames)

  fig, ax = plt.subplots(figsize=(14, max(4, len(markers) * 0.2)))
  cmap = plt.cm.colors.ListedColormap(["#ffffff", "#ecc94b", "#e53e3e"])
  ax.imshow(matrix.T, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=2)
  ax.set_xlabel(f"Frame index{downsample_note}")
  ax.set_ylabel("Marker")
  ax.set_yticks(range(len(markers)))
  ax.set_yticklabels(markers, fontsize=7)
  ax.set_title(title + downsample_note)
  from matplotlib.patches import Patch
  ax.legend(
    handles=[
      Patch(facecolor="#ffffff", edgecolor="#ccc", label="present"),
      Patch(facecolor="#ecc94b", label="gap 0.2–0.5 s"),
      Patch(facecolor="#e53e3e", label="gap ≥0.5 s"),
    ],
    loc="upper right",
    fontsize=8,
  )
  fig.tight_layout()
  fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
  plt.close(fig)


def build_gap_severity_matrix(
  session: MotiveSession,
  markers: list[str],
  config: dict[str, Any],
  gap_events: pd.DataFrame | None = None,
) -> np.ndarray:
  """Per-frame per-marker severity: 0=present, 1=moderate gap, 2=large gap."""
  valid = session.valid_marker_frame.sel(marker=markers).values.astype(bool)
  n_frames, n_markers = valid.shape
  matrix = np.zeros((n_frames, n_markers), dtype=np.float32)
  matrix[~valid] = 1.0

  if gap_events is None:
    return matrix

  thresholds = config["gaps"]["thresholds_seconds"]
  moderate = float(thresholds["moderate_gap"])
  large = float(thresholds["large_gap"])
  frames = session.coordinates.coords["frame"].values.astype(int)
  marker_to_idx = {m: i for i, m in enumerate(markers)}

  labeled = gap_events[gap_events["is_labeled"]] if "is_labeled" in gap_events.columns else gap_events
  for _, g in labeled.iterrows():
    mk = str(g.get("marker_name", ""))
    if mk not in marker_to_idx:
      continue
    mi = marker_to_idx[mk]
    dur = float(g["duration_seconds"])
    if dur < moderate:
      continue
    sf, ef = int(g["gap_start_frame"]), int(g["gap_end_frame"])
    in_gap = (frames >= sf) & (frames <= ef)
    sev = 2.0 if dur >= large else 1.0
    matrix[in_gap, mi] = np.maximum(matrix[in_gap, mi], sev)
  return matrix


def _downsample_matrix_max(
    matrix: np.ndarray,
    max_frames: int,
    time_s: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Downsample (frames × markers) by max-pooling along time."""
    n_frames = matrix.shape[0]
    if n_frames <= max_frames:
        return matrix, (np.asarray(time_s, dtype=float) if time_s is not None else None)
    step = int(np.ceil(n_frames / max_frames))
    n_new = int(np.ceil(n_frames / step))
    pooled = np.zeros((n_new, matrix.shape[1]), dtype=matrix.dtype)
    pooled_time: np.ndarray | None = None
    if time_s is not None:
        pooled_time = np.zeros(n_new, dtype=float)
    for i in range(n_new):
        chunk = matrix[i * step : (i + 1) * step, :]
        if chunk.size:
            pooled[i] = chunk.max(axis=0)
        if pooled_time is not None:
            time_chunk = np.asarray(time_s[i * step : (i + 1) * step], dtype=float)
            if time_chunk.size:
                pooled_time[i] = float(time_chunk[-1])
    return pooled, pooled_time


def _in_analysis_marker_names(session: MotiveSession) -> list[str]:
    inventory = session.marker_inventory
    if "included_in_analysis" in inventory.columns:
        return inventory.loc[
            inventory["included_in_analysis"].astype(bool), "marker_name"
        ].tolist()
    return inventory.loc[inventory["is_labeled"], "marker_name"].tolist()


def build_artifact_candidate_matrix(
    session: MotiveSession,
    markers: list[str],
    artifact_candidates: pd.DataFrame,
) -> np.ndarray:
    """Per-frame per-marker artifact severity: 0=none, 1=minor, 2=moderate, 3=severe/swap."""
    n_frames = len(session.coordinates.coords["frame"])
    n_markers = len(markers)
    matrix = np.zeros((n_frames, n_markers), dtype=np.float32)
    if artifact_candidates.empty or not markers:
        return matrix

    frames = session.coordinates.coords["frame"].values.astype(int)
    frame_to_idx = {int(f): i for i, f in enumerate(frames)}
    marker_to_idx = {m: i for i, m in enumerate(markers)}
    severity_level = {"minor": 1.0, "moderate": 2.0, "severe": 3.0}

    for _, row in artifact_candidates.iterrows():
        mk = str(row.get("marker_name", ""))
        if mk not in marker_to_idx:
            continue
        fi = frame_to_idx.get(int(row["frame"]))
        if fi is None:
            continue
        mi = marker_to_idx[mk]
        method = str(row.get("method", ""))
        if method == "segment_length_violation":
            level = severity_level.get(str(row.get("severity", "moderate")), 2.0)
        else:
            level = severity_level.get(str(row.get("severity", "minor")), 1.0)
        matrix[fi, mi] = np.maximum(matrix[fi, mi], level)
    return matrix


def plot_artifact_candidate_heatmap_on_ax(
    ax: plt.Axes,
    session: MotiveSession,
    artifact_candidates: pd.DataFrame,
    config: dict[str, Any],
    title: str,
    *,
    sigma: float | None = None,
    show_marker_labels: bool = False,
) -> None:
    """Draw artifact-candidate severity heatmap on an existing axes."""
    markers = _in_analysis_marker_names(session)
    if not markers:
        ax.text(0.5, 0.5, "No in-analysis markers", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    max_markers = config["outputs"].get("max_markers_per_heatmap", 80)
    markers = markers[:max_markers]
    matrix = build_artifact_candidate_matrix(session, markers, artifact_candidates)
    max_frames = config["outputs"].get("heatmap_downsample_max_frames", 5000)
    time_s = session.time_seconds.values.astype(float)
    matrix, time_ds = _downsample_matrix_max(matrix, max_frames, time_s)

    cmap = plt.cm.colors.ListedColormap(["#ffffff", "#feebc8", "#ed8936", "#c53030"])
    n_cols = matrix.shape[0]
    if time_ds is not None and len(time_ds) == n_cols and n_cols > 0:
        t0, t1 = float(time_ds[0]), float(time_ds[-1])
        if n_cols > 1:
            half_col = (t1 - t0) / (n_cols - 1) / 2.0
            t0 -= half_col
            t1 += half_col
        extent = [t0, t1, -0.5, len(markers) - 0.5]
    else:
        extent = [0, n_cols, -0.5, len(markers) - 0.5]
    ax.imshow(
        matrix.T,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=3,
        extent=extent,
        origin="lower",
    )
    subtitle = f"MAD σ={sigma:g}" if sigma is not None else "artifact candidates"
    ax.set_title(f"{title}\n({subtitle})", fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Marker", fontsize=8)
    if show_marker_labels:
        ax.set_yticks(range(len(markers)))
        ax.set_yticklabels(markers, fontsize=7)
    else:
        ax.set_yticks([])


def plot_gap_severity_heatmap_on_ax(
  ax: plt.Axes,
  session: MotiveSession,
  gap_events: pd.DataFrame,
  config: dict[str, Any],
  title: str,
) -> None:
  """Draw a 3-level gap-severity heatmap on an existing axes (batch grid use)."""
  inventory = session.marker_inventory
  if "included_in_analysis" in inventory.columns:
    markers = inventory.loc[inventory["included_in_analysis"].astype(bool), "marker_name"].tolist()
  else:
    markers = inventory.loc[inventory["is_labeled"], "marker_name"].tolist()
  if not markers:
    ax.text(0.5, 0.5, "No in-analysis markers", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    return

  max_markers = config["outputs"].get("max_markers_per_heatmap", 80)
  markers = markers[:max_markers]
  matrix = build_gap_severity_matrix(session, markers, config, gap_events)
  max_frames = config["outputs"].get("heatmap_downsample_max_frames", 5000)
  matrix, _ = _downsample_matrix_max(matrix, max_frames)

  cmap = plt.cm.colors.ListedColormap(["#ffffff", "#ecc94b", "#e53e3e"])
  ax.imshow(matrix.T, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=2)
  ax.set_title(title, fontsize=10)
  ax.set_xlabel("Frame", fontsize=8)
  ax.set_ylabel("Marker", fontsize=8)
  ax.set_yticks([])


def plot_gap_timeline(
  gap_events: pd.DataFrame,
  config: dict[str, Any],
  output_path: Path,
  dpi: int,
) -> None:
  fig, ax = plt.subplots(figsize=(14, max(4, 6)))
  moderate_thr = config["gaps"]["thresholds_seconds"]["moderate_gap"]
  labeled = (
    gap_events[gap_events["is_labeled"] & (gap_events["duration_seconds"] >= moderate_thr)]
    if not gap_events.empty
    else pd.DataFrame()
  )
  if labeled.empty:
    ax.text(0.5, 0.5, "No labeled gaps >= moderate threshold", ha="center", va="center")
  else:
    labeled = labeled.sort_values("gap_start_frame")
    severity_colors = {
      "moderate": "#ed8936",
      "large": "#e53e3e",
      "severe": "#9b2c2c",
      "minor": "#ecc94b",
      "tiny": "#a0aec0",
      "single_frame": "#cbd5e0",
    }
    y_labels: list[str] = []
    for row_idx, (_, gap) in enumerate(labeled.iterrows()):
      color = severity_colors.get(gap["severity_label"], "#4a5568")
      ax.barh(
        row_idx,
        gap["duration_seconds"],
        left=gap["gap_start_time_seconds"],
        height=0.8,
        color=color,
        edgecolor="white",
      )
      y_labels.append(f"{gap['marker_name']} ({gap['severity_label']})")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Labeled gap events")
  ax.set_title(f"Labeled gap timeline (>= {moderate_thr:.1f} s)")
  fig.tight_layout()
  fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
  plt.close(fig)


def plot_labeled_missing_count_over_time(
  frame_counts: pd.DataFrame | None,
  gap_events: pd.DataFrame,
  config: dict[str, Any],
  output_path: Path,
  dpi: int,
) -> None:
  fig, ax = plt.subplots(figsize=(14, 5))
  if frame_counts is None or frame_counts.empty:
    ax.text(0.5, 0.5, "No frame-level missingness data", ha="center", va="center")
  else:
    ax.plot(
      frame_counts["time_seconds"],
      frame_counts["labeled_missing_count"],
      color="#2b6cb0",
      linewidth=0.8,
    )
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Missing labeled markers (count)")
    moderate_thr = config["gaps"]["thresholds_seconds"]["moderate_gap"]
    if not gap_events.empty:
      mod_gaps = gap_events[
        gap_events["is_labeled"] & (gap_events["duration_seconds"] >= moderate_thr)
      ]
      for _, gap in mod_gaps.iterrows():
        ax.axvspan(
          gap["gap_start_time_seconds"],
          gap["gap_end_time_seconds"],
          color="#fed7d7",
          alpha=0.35,
        )
  ax.set_title("Labeled marker missing count over time")
  fig.tight_layout()
  fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
  plt.close(fig)


def plot_unlabeled_count_over_time(
  frame_counts: pd.DataFrame | None,
  gap_events: pd.DataFrame,
  config: dict[str, Any],
  output_path: Path,
  dpi: int,
) -> None:
  fig, ax = plt.subplots(figsize=(14, 5))
  if frame_counts is None or frame_counts.empty:
    ax.text(0.5, 0.5, "No unlabeled marker data", ha="center", va="center")
  else:
    ax.plot(
      frame_counts["time_seconds"],
      frame_counts["unlabeled_count"],
      color="#4a5568",
      linewidth=0.8,
    )
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Unlabeled marker count")
    large_thr = config["gaps"]["thresholds_seconds"]["large_gap"]
    if not gap_events.empty:
      large_gaps = gap_events[
        gap_events["is_labeled"] & (gap_events["duration_seconds"] >= large_thr)
      ]
      for _, gap in large_gaps.iterrows():
        ax.axvspan(
          gap["gap_start_time_seconds"],
          gap["gap_end_time_seconds"],
          color="#fed7d7",
          alpha=0.4,
        )
  ax.set_title("Unlabeled marker count over time")
  fig.tight_layout()
  fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
  plt.close(fig)


def _plot_output_path(config: dict[str, Any], name: str) -> Path:
    plot_dir = plot_dir_from_config(config)
    fmt = config.get("outputs", {}).get("plot_format", "png")
    return plot_dir / f"{name}.{fmt}"


def _plot_dpi(config: dict[str, Any]) -> int:
    return int(config.get("outputs", {}).get("dpi", 300))


def plot_frame_missingness_timeline(
    frame_quality: pd.DataFrame, config: dict[str, Any]
) -> Path:
    path = _plot_output_path(config, "frame_missingness_timeline")
    dpi = _plot_dpi(config)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        frame_quality["time_seconds"],
        frame_quality["missing_labeled_percent"],
        color="#2b6cb0",
        linewidth=0.8,
    )
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Missing labeled markers (%)")
    ax.set_title("Labeled marker missing percent over time")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_window_quality_timeline(window_df: pd.DataFrame, config: dict[str, Any]) -> Path:
    path = _plot_output_path(config, "window_quality_timeline")
    dpi = _plot_dpi(config)
    colors = {
        "use": "#48bb78",
        "caution": "#ecc94b",
        "exclude_or_review": "#e53e3e",
    }
    fig, ax = plt.subplots(figsize=(14, 4))
    for _, row in window_df.iterrows():
        color = colors.get(row["window_quality_label"], "#a0aec0")
        ax.barh(
            0,
            row["end_time_s"] - row["start_time_s"],
            left=row["start_time_s"],
            height=0.5,
            color=color,
            edgecolor="white",
        )
    ax.set_yticks([])
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Window quality labels along session")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_gap_timeline_by_group(gap_events: pd.DataFrame, config: dict[str, Any]) -> Path:
    path = _plot_output_path(config, "gap_timeline_by_group")
    dpi = _plot_dpi(config)
    moderate_thr = config["gaps"]["thresholds_seconds"]["moderate_gap"]
    labeled = (
        gap_events[gap_events["is_labeled"] & (gap_events["duration_seconds"] >= moderate_thr)]
        if not gap_events.empty
        else pd.DataFrame()
    )
    fig, ax = plt.subplots(figsize=(14, max(4, 6)))
    if labeled.empty:
        ax.text(0.5, 0.5, "No labeled gaps >= moderate threshold", ha="center", va="center")
    else:
        groups = sorted(labeled["body_region_group"].unique())
        group_y = {g: i for i, g in enumerate(groups)}
        palette = plt.cm.tab20(np.linspace(0, 1, max(len(groups), 1)))
        for _, gap in labeled.iterrows():
            y = group_y[gap["body_region_group"]]
            ax.barh(
                y,
                gap["duration_seconds"],
                left=gap["gap_start_time_seconds"],
                height=0.7,
                color=palette[y % len(palette)],
                edgecolor="white",
            )
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels(groups, fontsize=8)
        ax.set_ylabel("Body region group")
    ax.set_xlabel("Time (seconds)")
    ax.set_title(f"Gap timeline by body region (>= {moderate_thr:.1f} s)")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_artifact_events_timeline(artifact_events: pd.DataFrame, config: dict[str, Any]) -> Path:
    path = _plot_output_path(config, "artifact_timeline")
    dpi = _plot_dpi(config)
    fig, ax = plt.subplots(figsize=(14, 6))
    if artifact_events.empty:
        ax.text(0.5, 0.5, "No artifact events detected", ha="center", va="center")
    else:
        class_colors = {
            "single_frame": "#ecc94b",
            "single_frame_spike": "#ecc94b",
            "short_burst": "#ed8936",
            "sustained": "#e53e3e",
        }
        plot_events = artifact_events.head(40)
        for row_idx, (_, ev) in enumerate(plot_events.iterrows()):
            color = class_colors.get(ev["event_class"], "#4a5568")
            start_t = float(ev.get("start_time_s", 0))
            end_t = float(ev.get("end_time_s", start_t))
            ax.barh(
                row_idx,
                max(end_t - start_t, 1e-6),
                left=start_t,
                height=0.7,
                color=color,
                edgecolor="white",
            )
        ax.set_yticks(range(len(plot_events)))
        labels = [
            f"{r['marker_name']} ({r['event_class']}, {r['body_region_group']})"
            for _, r in plot_events.iterrows()
        ]
        ax.set_yticklabels(labels, fontsize=6)
        title = "Artifact events by duration and body region"
        if len(artifact_events) > 40:
            title = f"Artifact events (first 40 of {len(artifact_events)})"
        ax.set_title(title)
    ax.set_xlabel("Time (seconds)")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _draw_velocity_artifact_histogram(
    ax: plt.Axes,
    distribution: dict[str, Any],
    flagged_speeds: np.ndarray,
    body_region_group: str | None = None,
) -> None:
    speeds = np.asarray(distribution.get("speeds_m_s", []), dtype=float)
    flagged = np.asarray(flagged_speeds, dtype=float) if flagged_speeds is not None else np.array([])
    group = body_region_group or distribution.get("body_region_group") or "all_labeled"
    group_label = "all labeled markers" if group == "all_labeled" else str(group).replace("_", " ")
    n_markers = int(distribution.get("n_markers", 0))

    if speeds.size == 0:
        ax.text(
            0.5,
            0.5,
            f"No gap-safe speed samples for {group_label}",
            ha="center",
            va="center",
        )
        ax.set_title(f"Velocity histogram — {group_label}")
        return

    vmax = float(np.percentile(speeds, 99.99))
    plot_speeds = speeds[speeds <= vmax] if vmax > 0 else speeds
    bins = min(80, max(20, int(np.sqrt(plot_speeds.size))))
    ax.hist(
        plot_speeds,
        bins=bins,
        range=(0, vmax),
        color="#cbd5e0",
        edgecolor="white",
        alpha=0.9,
        label=f"All speeds (n={speeds.size:,})",
    )
    if flagged.size:
        flagged_in = flagged[flagged <= vmax]
        if flagged_in.size:
            ax.hist(
                flagged_in,
                bins=bins,
                range=(0, vmax),
                color="#e53e3e",
                edgecolor="#c53030",
                alpha=0.75,
                label=f"Flagged velocity peaks (n={flagged.size:,})",
            )

    vel_mad_thr = float(distribution.get("vel_mad_threshold_m_s", 0))
    vel_pct_thr = float(distribution.get("vel_percentile_threshold_m_s", 0))
    vel_thr = float(distribution.get("vel_threshold_m_s", 0))
    vel_mult = distribution.get("vel_mad_multiplier", "")
    vel_pct = distribution.get("vel_percentile_config", "")

    if vel_mad_thr > 0:
        ax.axvline(
            vel_mad_thr,
            color="#2b6cb0",
            linestyle="--",
            linewidth=1.2,
            label=f"MAD threshold (σ={vel_mult})",
        )
    if vel_pct_thr > 0 and abs(vel_pct_thr - vel_mad_thr) > 1e-9:
        ax.axvline(vel_pct_thr, color="#805ad5", linestyle=":", linewidth=1.2, label="Percentile floor")
    if vel_thr > 0:
        ax.axvline(
            vel_thr,
            color="#e53e3e",
            linestyle="-",
            linewidth=1.5,
            label="Effective threshold (max of both)",
        )

    units = str(distribution.get("length_units") or "m")
    if units in ("", "unknown", "None"):
        units = "m"
    ax.set_xlabel(f"Marker speed ({units}/s)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Velocity artifact histogram — {group_label}\n"
        f"({n_markers} markers · MAD σ={vel_mult} · pct={vel_pct})"
    )
    ax.legend(loc="upper right", fontsize=8)
    n_above = int((speeds >= vel_thr).sum()) if vel_thr > 0 else 0
    ax.text(
        0.02,
        0.98,
        f"MAD σ={vel_mult}, pct={vel_pct}\n"
        f"Samples ≥ threshold: {n_above:,} ({100 * n_above / speeds.size:.3f}%)",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )


def figure_velocity_artifact_histogram(
    distribution: dict[str, Any],
    flagged_speeds: np.ndarray,
    body_region_group: str | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))
    _draw_velocity_artifact_histogram(ax, distribution, flagged_speeds, body_region_group)
    fig.tight_layout()
    return fig


def plot_velocity_artifact_histogram(
    distribution: dict[str, Any],
    flagged_speeds: np.ndarray,
    config: dict[str, Any],
    body_region_group: str | None = None,
) -> Path:
    """Histogram of gap-safe labeled-marker speeds with tuning threshold lines."""
    group = body_region_group or distribution.get("body_region_group") or "all_labeled"
    suffix = "" if group == "all_labeled" else f"_{group}"
    path = _plot_output_path(config, f"artifact_velocity_histogram{suffix}")
    dpi = _plot_dpi(config)
    fig = figure_velocity_artifact_histogram(distribution, flagged_speeds, group)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_artifact_timeline(artifact_candidates: pd.DataFrame, config: dict[str, Any]) -> Path:
    path = _plot_output_path(config, "artifact_timeline")
    dpi = _plot_dpi(config)
    fig, ax = plt.subplots(figsize=(14, 5))
    if artifact_candidates.empty:
        ax.text(0.5, 0.5, "No artifact candidates detected", ha="center", va="center")
    else:
        severity_colors = {
            "minor": "#ecc94b",
            "moderate": "#ed8936",
            "severe": "#e53e3e",
        }
        for severity in ("minor", "moderate", "severe"):
            subset = artifact_candidates[artifact_candidates["severity"] == severity]
            if subset.empty:
                continue
            ax.scatter(
                subset["time_seconds"],
                subset["method"],
                c=severity_colors[severity],
                s=30,
                alpha=0.7,
                label=severity,
            )
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Detection method")
    ax.set_title("Artifact candidate timeline")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
