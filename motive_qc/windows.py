"""Layer 3: PCA/jPCA window safety (runs after Layer 4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from motive_qc.core import LOGGER, MotiveSession, QCResult
from motive_qc.analysis_scope import analysis_labeled_marker_names
from motive_qc.plots import (
    plot_frame_missingness_timeline,
    plot_gap_timeline_by_group,
    plot_window_quality_timeline,
)


def _frame_quality_config(config: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "enabled": True,
        "missing_marker_percent_warn": 5.0,
        "missing_marker_percent_caution": 10.0,
        "missing_marker_percent_bad": 20.0,
        "critical_groups": [],
    }
    return {**defaults, **config.get("frame_quality", {})}


def build_frame_quality_summary(
    session: MotiveSession,
    gap_events: pd.DataFrame,
    frame_qc_mask: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    fq_cfg = _frame_quality_config(config)
    if frame_qc_mask.empty:
        return pd.DataFrame()

    frames = session.coordinates.coords["frame"].values
    inventory = session.marker_inventory
    labeled = analysis_labeled_marker_names(inventory, config)
    critical_groups = set(fq_cfg.get("critical_groups", []))

    affected_groups: list[str] = []
    frame_labels: list[str] = []

    if labeled:
        labeled_valid = session.valid_marker_frame.sel(marker=labeled).values
        inv_by_marker = inventory.set_index("marker_name")
    else:
        labeled_valid = np.zeros((len(frames), 0), dtype=bool)

    for i in range(len(frames)):
        missing_markers = []
        if labeled:
            for j, marker in enumerate(labeled):
                if not labeled_valid[i, j]:
                    missing_markers.append(marker)
        groups = sorted(
            {
                inv_by_marker.loc[m, "body_region_group"]
                for m in missing_markers
                if m in inv_by_marker.index
            }
        )
        affected_groups.append(";".join(groups))

        pct = float(frame_qc_mask.iloc[i]["missing_labeled_percent"])
        label = "use"
        if frame_qc_mask.iloc[i]["qc_status"] == "exclude_or_review":
            label = "exclude_or_review"
        elif frame_qc_mask.iloc[i]["qc_status"] == "caution":
            label = "caution"
        elif pct >= fq_cfg["missing_marker_percent_bad"]:
            label = "exclude_or_review"
        elif pct >= fq_cfg["missing_marker_percent_caution"]:
            label = "caution"
        elif any(g in critical_groups for g in groups) and pct >= fq_cfg["missing_marker_percent_warn"]:
            label = "caution"
        frame_labels.append(label)

    out = frame_qc_mask.copy()
    out["affected_body_groups"] = affected_groups
    out["frame_quality_label"] = frame_labels
    return out


def build_window_bins(
    session: MotiveSession,
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    win_cfg = config.get("windows", {})
    lengths = win_cfg.get("window_lengths_seconds", [0.5, 1.0])
    frame_rate = float(session.metadata["effective_frame_rate_hz"])
    frames = session.coordinates.coords["frame"].values
    time_values = session.time_seconds.values
    n_frames = len(frames)
    step_seconds = win_cfg.get("step_seconds")
    tables: dict[str, pd.DataFrame] = {}

    for length_s in lengths:
        win_frames = max(1, int(round(length_s * frame_rate)))
        step_frames = win_frames
        if step_seconds is not None:
            step_frames = max(1, int(round(float(step_seconds) * frame_rate)))

        rows: list[dict[str, Any]] = []
        window_id = 0
        start_idx = 0
        while start_idx < n_frames:
            end_idx = min(start_idx + win_frames - 1, n_frames - 1)
            window_id += 1
            rows.append(
                {
                    "window_id": f"W{window_id:05d}",
                    "start_frame": int(frames[start_idx]),
                    "end_frame": int(frames[end_idx]),
                    "start_time_s": float(time_values[start_idx]),
                    "end_time_s": float(time_values[end_idx]),
                    "duration_frames": end_idx - start_idx + 1,
                    "window_length_s": float(length_s),
                }
            )
            if end_idx >= n_frames - 1:
                break
            start_idx += step_frames

        key = f"window_quality_{str(length_s).replace('.', 'p')}s"
        tables[key] = pd.DataFrame(rows)
    return tables


def _events_in_window(
    events: pd.DataFrame,
    start_frame: int,
    end_frame: int,
) -> pd.DataFrame:
    if events.empty:
        return events
    return events[
        (events["end_frame"] >= start_frame) & (events["start_frame"] <= end_frame)
    ]


def _finalize_window_row(
    session: MotiveSession,
    bin_row: pd.Series,
    gap_events: pd.DataFrame,
    frame_quality: pd.DataFrame,
    artifact_events: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    win_cfg = config.get("windows", {})
    critical_groups = set(config.get("frame_quality", {}).get("critical_groups", []))
    frame_rate = float(session.metadata["effective_frame_rate_hz"])
    frames = session.coordinates.coords["frame"].values

    start_frame = int(bin_row["start_frame"])
    end_frame = int(bin_row["end_frame"])
    idx_start = int(np.searchsorted(frames, start_frame))
    idx_end = int(np.searchsorted(frames, end_frame))

    fq_slice = frame_quality.iloc[idx_start : idx_end + 1] if not frame_quality.empty else pd.DataFrame()
    mean_missing = float(fq_slice["missing_labeled_percent"].mean()) if not fq_slice.empty else 0.0
    max_missing = float(fq_slice["missing_labeled_percent"].max()) if not fq_slice.empty else 0.0

    overlapping = gap_events[
        gap_events["is_labeled"]
        & (gap_events["gap_start_frame"] <= end_frame)
        & (gap_events["gap_end_frame"] >= start_frame)
    ] if not gap_events.empty else pd.DataFrame()

    overlap_seconds = 0.0
    gap_groups: set[str] = set()
    worst_marker = None
    max_gap_dur = 0.0
    critical_affected = False

    if not overlapping.empty:
        for _, gap in overlapping.iterrows():
            g_start = max(int(gap["gap_start_frame"]), start_frame)
            g_end = min(int(gap["gap_end_frame"]), end_frame)
            if g_end >= g_start:
                overlap_seconds += (g_end - g_start + 1) / frame_rate
            gap_groups.add(str(gap["body_region_group"]))
            if float(gap["duration_seconds"]) > max_gap_dur:
                max_gap_dur = float(gap["duration_seconds"])
                worst_marker = gap["marker_name"]
            if gap["body_region_group"] in critical_groups:
                critical_affected = True

    win_events = _events_in_window(artifact_events, start_frame, end_frame)
    artifact_groups: set[str] = set()
    n_sustained = 0
    max_art_dur = 0.0
    if not win_events.empty:
        artifact_groups = set(win_events["body_region_group"].astype(str).tolist())
        n_sustained = int((win_events["event_class"] == "sustained").sum())
        max_art_dur = float(win_events["duration_seconds"].max())

    gap_thr = win_cfg.get("flag_if_gap_at_least_seconds", 0.2)
    large_gap_thr = win_cfg.get("flag_if_large_gap_at_least_seconds", 0.5)
    missing_thr = win_cfg.get("flag_if_missing_marker_percent_above", 10.0)

    reasons: list[str] = []
    label = "use"

    if overlap_seconds >= large_gap_thr:
        reasons.append("LARGE_GAP_OVERLAP")
        label = "exclude_or_review"
    elif overlap_seconds >= gap_thr:
        reasons.append("GAP_OVERLAP")
        if label == "use":
            label = "caution"

    if max_missing >= missing_thr * 2:
        reasons.append("HIGH_MISSING")
        label = "exclude_or_review"
    elif max_missing >= missing_thr:
        reasons.append("ELEVATED_MISSING")
        if label == "use":
            label = "caution"

    if critical_affected and not overlapping.empty:
        reasons.append("CRITICAL_GROUP_GAP")
        if label == "use":
            label = "caution"

    if not win_events.empty:
        severe = win_events[win_events["severity"] == "severe"]
        sustained = win_events[win_events["event_class"] == "sustained"]
        if not severe.empty or not sustained.empty:
            reasons.append("SUSTAINED_ARTIFACT_IN_WINDOW")
            label = "exclude_or_review"
        elif label == "use":
            reasons.append("ARTIFACT_EVENT_IN_WINDOW")
            label = "caution"

    all_groups = sorted(
        g for g in (gap_groups | artifact_groups) if g and g != "unlabeled"
    )

    row = bin_row.to_dict()
    row.update(
        {
            "mean_missing_labeled_percent": round(mean_missing, 6),
            "max_missing_labeled_percent": round(max_missing, 6),
            "n_gaps_overlapping": len(overlapping),
            "overlap_gap_seconds": round(overlap_seconds, 6),
            "max_gap_duration_s": round(max_gap_dur, 6),
            "worst_gap_marker": worst_marker,
            "critical_group_affected": critical_affected,
            "n_artifact_events": len(win_events),
            "n_sustained_artifact_events": n_sustained,
            "max_artifact_event_duration_s": round(max_art_dur, 6),
            "affected_body_groups": ";".join(all_groups),
            "artifact_body_groups": ";".join(sorted(artifact_groups)),
            "window_quality_label": label,
            "reason_codes": ";".join(reasons),
        }
    )
    return row


def finalize_window_quality(
    session: MotiveSession,
    layer2_result: QCResult,
    layer4_result: QCResult,
    window_bins: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    gap_events = layer2_result.tables.get("gap_events", pd.DataFrame())
    frame_qc_mask = layer2_result.tables.get("frame_qc_mask", pd.DataFrame())
    frame_quality = build_frame_quality_summary(session, gap_events, frame_qc_mask, config)
    artifact_events = layer4_result.tables.get("artifact_events", pd.DataFrame())

    finalized: dict[str, pd.DataFrame] = {}
    for key, bins in window_bins.items():
        rows = [
            _finalize_window_row(session, row, gap_events, frame_quality, artifact_events, config)
            for _, row in bins.iterrows()
        ]
        finalized[key] = pd.DataFrame(rows)
    return finalized, frame_quality


def build_window_quality_summary(
    window_tables: dict[str, pd.DataFrame],
    session: MotiveSession,
) -> pd.DataFrame:
    duration_s = float(session.metadata.get("duration_seconds", 0))
    rows: list[dict[str, Any]] = []
    for key, wdf in window_tables.items():
        if wdf.empty:
            continue
        length_s = float(wdf.iloc[0].get("window_length_s", 0.5))
        flagged = wdf[wdf["window_quality_label"] != "use"]
        flagged_time = float(
            flagged["end_time_s"].sum() - flagged["start_time_s"].sum()
        ) if not flagged.empty else 0.0
        rows.append(
            {
                "window_length_s": length_s,
                "table_name": key,
                "n_windows": len(wdf),
                "n_with_gap_overlap": int((wdf["n_gaps_overlapping"] > 0).sum()),
                "n_with_artifact_events": int((wdf["n_artifact_events"] > 0).sum()),
                "n_flagged_caution": int((wdf["window_quality_label"] == "caution").sum()),
                "n_flagged_exclude": int(
                    (wdf["window_quality_label"] == "exclude_or_review").sum()
                ),
                "pct_session_in_flagged_windows": round(
                    100.0 * flagged_time / duration_s, 4
                )
                if duration_s > 0
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def enrich_group_quality_summary(
    gap_summary_by_group: pd.DataFrame,
    window_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if gap_summary_by_group.empty:
        return gap_summary_by_group
    out = gap_summary_by_group.copy()
    for length_key, wdf in window_tables.items():
        suffix = length_key.replace("window_quality_", "")
        if wdf.empty:
            out[f"n_windows_flagged_{suffix}"] = 0
            continue
        flagged = wdf[wdf["window_quality_label"] != "use"]
        out[f"n_windows_flagged_{suffix}"] = len(flagged)
    return out


def run_layer3_windows(
    session: MotiveSession,
    layer2_result: QCResult,
    layer4_result: QCResult | None,
    config: dict[str, Any],
    verbose: bool = False,
) -> QCResult:
    if not config.get("windows", {}).get("enabled", False):
        return QCResult(layer_name="layer3", status="skipped", session=session)

    if layer4_result is None:
        return QCResult(layer_name="layer3", status="skipped", session=session)

    if verbose:
        LOGGER.info("Running Layer 3 window safety (L2 gaps + L4 artifacts)")

    window_bins = build_window_bins(session, config)
    window_tables, frame_quality = finalize_window_quality(
        session, layer2_result, layer4_result, window_bins, config
    )
    window_summary = build_window_quality_summary(window_tables, session)
    group_summary = enrich_group_quality_summary(
        layer2_result.tables.get("gap_summary_by_group", pd.DataFrame()),
        window_tables,
    )

    gap_events = layer2_result.tables.get("gap_events", pd.DataFrame())
    figures: dict[str, Path] = {}
    plots_cfg = config.get("outputs", {}).get("plots", {})
    if plots_cfg.get("enabled", True):
        if plots_cfg.get("frame_missingness_timeline", True) and not frame_quality.empty:
            figures["frame_missingness_timeline"] = plot_frame_missingness_timeline(
                frame_quality, config
            )
        if plots_cfg.get("window_quality_timeline", True):
            primary = window_tables.get("window_quality_0p5s", pd.DataFrame())
            if not primary.empty:
                figures["window_quality_timeline"] = plot_window_quality_timeline(
                    primary, config
                )
        if plots_cfg.get("gap_timeline_by_group", True):
            figures["gap_timeline_by_group"] = plot_gap_timeline_by_group(gap_events, config)

    tables: dict[str, pd.DataFrame] = {
        "frame_quality_summary": frame_quality,
        "group_quality_summary": group_summary,
        "window_quality_summary": window_summary,
    }
    tables.update(window_tables)

    return QCResult(
        layer_name="layer3",
        status="pass",
        tables=tables,
        figures=figures,
        messages=list(session.validation_messages),
        session=session,
    )
