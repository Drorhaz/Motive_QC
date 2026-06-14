"""Focused, remediation-oriented per-session deliverables.

These are the lean outputs the pipeline is centered on:

* ``gaps_over_0p5s`` / ``gaps_over_0p2s`` -- per labeled marker, "how many and
  which marker" gaps exceed the threshold.
* ``artifacts_by_segment``               -- artifact burden rolled up per body region.
* ``qc_mask`` + ``qc_mask_intervals``    -- criterion-tagged, advisory frame mask
  (gap_0p2 / gap_0p5 / artifact_sigma / segment_swap / edge_effect) that downstream
  Layer 2/3 consume, choosing which criteria to honor.

Nothing here repairs data; it only labels and recommends.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from motive_qc.analysis_scope import (
    analysis_labeled_marker_names,
    filter_artifact_events_for_analysis,
    filter_gap_events_for_analysis,
)

UNLABELED_BODY_GROUP = "unlabeled"

MASK_CRITERIA = (
    "gaps_over_0p2",
    "gaps_over_0p5",
    "artifact_sigma",
    "segment_swap",
    "edge_effect",
)


def _in_analysis_labeled_gaps(
    gap_events: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    if gap_events.empty or "is_labeled" not in gap_events.columns:
        return gap_events
    gaps = filter_gap_events_for_analysis(gap_events, config)
    return gaps[gaps["is_labeled"]].copy()


def build_gaps_over_threshold(
    gap_events: pd.DataFrame,
    config: dict[str, Any],
    threshold_seconds: float,
    upper_seconds: float | None = None,
) -> pd.DataFrame:
    """Per labeled marker: count/length of gaps >= threshold (optionally < upper)."""
    cols = [
        "marker_name",
        "body_region_group",
        "n_gaps",
        "total_gap_seconds",
        "longest_gap_seconds",
        "gap_intervals_s",
    ]
    labeled = _in_analysis_labeled_gaps(gap_events, config)
    if labeled.empty:
        return pd.DataFrame(columns=cols)
    sel = labeled[labeled["duration_seconds"] >= threshold_seconds]
    if upper_seconds is not None:
        sel = sel[sel["duration_seconds"] < upper_seconds]
    if sel.empty:
        return pd.DataFrame(columns=cols)

    rows: list[dict[str, Any]] = []
    for marker, grp in sel.groupby("marker_name"):
        grp = grp.sort_values("gap_start_time_seconds")
        intervals = "; ".join(
            f"{r['gap_start_time_seconds']:.2f}-{r['gap_end_time_seconds']:.2f}"
            for _, r in grp.iterrows()
        )
        rows.append(
            {
                "marker_name": marker,
                "body_region_group": grp.iloc[0]["body_region_group"],
                "n_gaps": int(len(grp)),
                "total_gap_seconds": round(float(grp["duration_seconds"].sum()), 4),
                "longest_gap_seconds": round(float(grp["duration_seconds"].max()), 4),
                "gap_intervals_s": intervals,
            }
        )
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(
        ["n_gaps", "longest_gap_seconds"], ascending=False
    ).reset_index(drop=True)


def build_artifacts_by_segment(
    artifact_events: pd.DataFrame,
    config: dict[str, Any],
    duration_seconds: float,
) -> pd.DataFrame:
    """Artifact burden per body region (labeled, in-analysis only)."""
    cols = [
        "body_region_group",
        "n_events",
        "n_single_frame",
        "n_short_burst",
        "n_sustained",
        "n_segment_swap",
        "events_per_minute",
    ]
    if artifact_events.empty:
        return pd.DataFrame(columns=cols)
    events = filter_artifact_events_for_analysis(artifact_events, config)
    if "body_region_group" in events.columns:
        events = events[events["body_region_group"].astype(str) != UNLABELED_BODY_GROUP]
    if events.empty:
        return pd.DataFrame(columns=cols)

    minutes = max(duration_seconds / 60.0, 1e-9)
    rows: list[dict[str, Any]] = []
    for region, grp in events.groupby("body_region_group"):
        ec = grp["event_class"]
        method = grp["method"] if "method" in grp.columns else pd.Series([], dtype=str)
        rows.append(
            {
                "body_region_group": region,
                "n_events": int(len(grp)),
                "n_single_frame": int(
                    ec.isin(["single_frame", "single_frame_spike"]).sum()
                ),
                "n_short_burst": int((ec == "short_burst").sum()),
                "n_sustained": int((ec == "sustained").sum()),
                "n_segment_swap": int((method == "segment_length_violation").sum())
                if len(method)
                else 0,
                "events_per_minute": round(len(grp) / minutes, 4),
            }
        )
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("n_events", ascending=False).reset_index(drop=True)


def _frames_in_events(
    events: pd.DataFrame, frames: np.ndarray, method_filter: set[str] | None = None
) -> np.ndarray:
    flag = np.zeros(len(frames), dtype=bool)
    if events.empty:
        return flag
    ev = events
    if method_filter is not None and "method" in ev.columns:
        ev = ev[ev["method"].isin(method_filter)]
    for _, e in ev.iterrows():
        flag |= (frames >= int(e["start_frame"])) & (frames <= int(e["end_frame"]))
    return flag


def build_qc_mask(
    session,
    gap_events: pd.DataFrame,
    artifact_events: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Criterion-tagged advisory frame mask + interval list (with edge buffers)."""
    frames = session.coordinates.coords["frame"].values.astype(int)
    times = np.asarray(session.time_seconds.values, dtype=float)
    n = len(frames)
    thresholds = config["gaps"]["thresholds_seconds"]
    moderate = float(thresholds["moderate_gap"])
    large = float(thresholds["large_gap"])
    edge_buffer = int(config.get("artifacts", {}).get("edge_buffer_frames", 8))

    flag_gap_0p2 = np.zeros(n, dtype=bool)
    flag_gap_0p5 = np.zeros(n, dtype=bool)
    flag_edge = np.zeros(n, dtype=bool)

    labeled_gaps = _in_analysis_labeled_gaps(gap_events, config)
    for _, g in labeled_gaps.iterrows():
        in_gap = (frames >= int(g["gap_start_frame"])) & (
            frames <= int(g["gap_end_frame"])
        )
        dur = float(g["duration_seconds"])
        if dur >= large:
            flag_gap_0p5 |= in_gap
            # Edge-effect buffers on both sides of a large gap.
            s = int(np.argmax(in_gap)) if in_gap.any() else None
            if s is not None:
                idx = np.where(in_gap)[0]
                lo, hi = idx[0], idx[-1]
                pre_lo = max(0, lo - edge_buffer)
                post_hi = min(n - 1, hi + edge_buffer)
                buf = np.zeros(n, dtype=bool)
                buf[pre_lo:lo] = True
                buf[hi + 1 : post_hi + 1] = True
                flag_edge |= buf
        elif dur >= moderate:
            flag_gap_0p2 |= in_gap

    labeled_art = filter_artifact_events_for_analysis(artifact_events, config)
    if "body_region_group" in labeled_art.columns:
        labeled_art = labeled_art[
            labeled_art["body_region_group"].astype(str) != UNLABELED_BODY_GROUP
        ]
    flag_segment_swap = _frames_in_events(
        labeled_art, frames, {"segment_length_violation"}
    )
    sigma_methods = {"velocity_mad", "acceleration_mad", "single_frame_spike"}
    flag_artifact_sigma = _frames_in_events(labeled_art, frames, sigma_methods)

    # Edge buffers never override a real gap frame.
    flag_edge &= ~(flag_gap_0p5 | flag_gap_0p2)

    status = np.full(n, "use", dtype=object)
    reasons: list[str] = []
    for i in range(n):
        codes: list[str] = []
        st = "use"
        if flag_gap_0p5[i]:
            codes.append("GAP_GE_0P5")
            st = "exclude"
        if flag_gap_0p2[i]:
            codes.append("GAP_GE_0P2")
            if st == "use":
                st = "caution"
        if flag_artifact_sigma[i]:
            codes.append("ARTIFACT_SIGMA")
            if st == "use":
                st = "caution"
        if flag_segment_swap[i]:
            codes.append("SEGMENT_SWAP")
            if st == "use":
                st = "caution"
        if flag_edge[i]:
            codes.append("EDGE_EFFECT")
            if st == "use":
                st = "caution"
        status[i] = st
        reasons.append(";".join(codes))

    mask = pd.DataFrame(
        {
            "frame": frames,
            "time_s": np.round(times, 6),
            "status": status,
            "flag_gap_0p2": flag_gap_0p2,
            "flag_gap_0p5": flag_gap_0p5,
            "flag_artifact_sigma": flag_artifact_sigma,
            "flag_segment_swap": flag_segment_swap,
            "flag_edge_effect": flag_edge,
            "reason": reasons,
        }
    )
    intervals = _mask_intervals(mask, labeled_gaps, labeled_art)
    intervals = enrich_mask_intervals(intervals)
    return mask, intervals


def enrich_mask_intervals(intervals: pd.DataFrame) -> pd.DataFrame:
    """Add boolean flag columns parsed from interval ``reason`` strings."""
    if intervals.empty:
        return intervals
    out = intervals.copy()
    reasons = out["reason"].astype(str)
    out["has_gap_ge_0p5"] = reasons.str.contains("GAP_GE_0P5", regex=False)
    out["has_gap_0p2"] = reasons.str.contains("GAP_GE_0P2", regex=False)
    out["has_artifact_sigma"] = reasons.str.contains("ARTIFACT_SIGMA", regex=False)
    out["has_segment_swap"] = reasons.str.contains("SEGMENT_SWAP", regex=False)
    out["has_edge_effect"] = reasons.str.contains("EDGE_EFFECT", regex=False)
    return out


def _merge_adjacent_artifact_intervals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge consecutive caution intervals that are artifact-only for readability."""
    if not rows:
        return rows
    merged: list[dict[str, Any]] = []
    i = 0
    while i < len(rows):
        row = rows[i]
        reason = str(row.get("reason", ""))
        if (
            row.get("status") == "caution"
            and "ARTIFACT_SIGMA" in reason
            and "GAP_GE_0P5" not in reason
            and "GAP_GE_0P2" not in reason
            and "SEGMENT_SWAP" not in reason
        ):
            j = i + 1
            affected = set(str(row.get("affected_markers", "")).split(";")) - {""}
            while j < len(rows):
                nxt = rows[j]
                nr = str(nxt.get("reason", ""))
                if (
                    nxt.get("status") == "caution"
                    and "ARTIFACT_SIGMA" in nr
                    and "GAP_GE_0P5" not in nr
                    and nxt["start_frame"] == rows[j - 1]["end_frame"] + 1
                ):
                    affected.update(str(nxt.get("affected_markers", "")).split(";"))
                    j += 1
                else:
                    break
            if j > i + 1:
                end_row = rows[j - 1]
                row = {
                    **row,
                    "end_frame": end_row["end_frame"],
                    "end_s": end_row["end_s"],
                    "duration_s": round(float(end_row["end_s"]) - float(row["start_s"]), 4),
                    "affected_markers": ";".join(sorted(affected)[:20]),
                }
            i = j
        else:
            i += 1
        merged.append(row)
    return merged


def _primary_criterion(reason: str) -> str:
    order = [
        ("GAP_GE_0P5", "gaps_over_0p5"),
        ("SEGMENT_SWAP", "segment_swap"),
        ("ARTIFACT_SIGMA", "artifact_sigma"),
        ("GAP_GE_0P2", "gaps_over_0p2"),
        ("EDGE_EFFECT", "edge_effect"),
    ]
    for code, crit in order:
        if code in reason:
            return crit
    return ""


def _mask_intervals(
    mask: pd.DataFrame,
    labeled_gaps: pd.DataFrame,
    labeled_art: pd.DataFrame,
) -> pd.DataFrame:
    cols = [
        "start_frame",
        "end_frame",
        "start_s",
        "end_s",
        "duration_s",
        "status",
        "reason",
        "criterion",
        "affected_markers",
    ]
    if mask.empty:
        return pd.DataFrame(columns=cols)
    frames = mask["frame"].values
    times = mask["time_s"].values
    statuses = mask["status"].values
    reasons = mask["reason"].values
    rows: list[dict[str, Any]] = []
    i = 0
    n = len(frames)
    while i < n:
        if statuses[i] == "use":
            i += 1
            continue
        start_i = i
        while (
            i < n
            and statuses[i] == statuses[start_i]
            and reasons[i] == reasons[start_i]
        ):
            i += 1
        end_i = i - 1
        sf, ef = int(frames[start_i]), int(frames[end_i])
        affected: set[str] = set()
        if not labeled_gaps.empty:
            ov = labeled_gaps[
                (labeled_gaps["gap_end_frame"] >= sf)
                & (labeled_gaps["gap_start_frame"] <= ef)
            ]
            affected.update(ov["marker_name"].astype(str).tolist())
        if not labeled_art.empty:
            ov = labeled_art[
                (labeled_art["end_frame"] >= sf) & (labeled_art["start_frame"] <= ef)
            ]
            affected.update(ov["marker_name"].astype(str).tolist())
        rows.append(
            {
                "start_frame": sf,
                "end_frame": ef,
                "start_s": round(float(times[start_i]), 4),
                "end_s": round(float(times[end_i]), 4),
                "duration_s": round(float(times[end_i] - times[start_i]), 4),
                "status": statuses[start_i],
                "reason": reasons[start_i],
                "criterion": _primary_criterion(str(reasons[start_i])),
                "affected_markers": ";".join(sorted(affected)[:12]),
            }
        )
    rows = _merge_adjacent_artifact_intervals(rows)
    return pd.DataFrame(rows, columns=cols)


def load_qc_mask(
    run_dir: str | Path,
    skip_criteria: list[str] | None = None,
) -> pd.DataFrame:
    """Load a session ``qc_mask.csv`` and add a boolean ``skip`` column.

    ``skip`` is the OR of only the selected criteria flag columns, letting a caller
    choose strictness (e.g. honor 0.5s gaps + chosen sigma, ignore 0.2s gaps).
    """
    run_dir = Path(run_dir)
    mask_path = run_dir / "tables" / "qc_mask.csv"
    if not mask_path.exists():
        mask_path = run_dir / "qc_mask.csv"
    mask = pd.read_csv(mask_path)
    if skip_criteria is None:
        skip_criteria = ["gaps_over_0p5", "artifact_sigma", "segment_swap", "edge_effect"]
    flag_for = {
        "gaps_over_0p2": "flag_gap_0p2",
        "gaps_over_0p5": "flag_gap_0p5",
        "artifact_sigma": "flag_artifact_sigma",
        "segment_swap": "flag_segment_swap",
        "edge_effect": "flag_edge_effect",
    }
    skip = np.zeros(len(mask), dtype=bool)
    for crit in skip_criteria:
        col = flag_for.get(crit)
        if col and col in mask.columns:
            skip |= mask[col].astype(bool).values
    mask["skip"] = skip
    return mask
