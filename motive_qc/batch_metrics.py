"""Extract per-session EDA metrics from in-memory Layer 1-5 results."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from motive_qc.analysis_scope import excluded_body_groups
from motive_qc.artifacts import (
    collect_session_velocity_distribution,
    list_velocity_histogram_groups,
    velocity_mad_sigma,
    velocity_percentile_threshold,
)
from motive_qc.core import MotiveSession, QCResult


def compute_window_yield(window_df: pd.DataFrame) -> float:
    if window_df.empty or "window_quality_label" not in window_df.columns:
        return 0.0
    n_use = int((window_df["window_quality_label"] == "use").sum())
    return round(100.0 * n_use / len(window_df), 4)


def _mask_pct(analysis_mask: pd.DataFrame, status: str) -> float:
    if analysis_mask.empty or "analysis_status" not in analysis_mask.columns:
        return 0.0
    n = len(analysis_mask)
    return round(100.0 * (analysis_mask["analysis_status"] == status).sum() / n, 4)


def _gap_counts(gap_events: pd.DataFrame) -> dict[str, int]:
    out = {
        "n_gaps_total_labeled": 0,
        "n_gaps_le_0p1s": 0,
        "n_gaps_le_0p2s": 0,
        "n_gaps_0p2_to_0p5s": 0,
        "n_gaps_0p5_to_1s": 0,
        "n_gaps_gt_1s": 0,
        "n_gaps_ge_0p2s": 0,
        "n_gaps_ge_0p5s": 0,
        "n_gaps_ge_1p0s": 0,
        "n_critical_region_gaps_ge_0p5s": 0,
    }
    if gap_events.empty:
        return out
    labeled = gap_events[gap_events["is_labeled"]] if "is_labeled" in gap_events.columns else gap_events
    if labeled.empty:
        return out
    durs = labeled["duration_seconds"].astype(float)
    out["n_gaps_total_labeled"] = len(labeled)
    out["n_gaps_le_0p1s"] = int((durs <= 0.1).sum())
    out["n_gaps_le_0p2s"] = int((durs < 0.2).sum())
    out["n_gaps_0p2_to_0p5s"] = int(((durs >= 0.2) & (durs < 0.5)).sum())
    out["n_gaps_0p5_to_1s"] = int(((durs >= 0.5) & (durs < 1.0)).sum())
    out["n_gaps_gt_1s"] = int((durs >= 1.0).sum())
    out["n_gaps_ge_0p2s"] = int((durs >= 0.2).sum())
    out["n_gaps_ge_0p5s"] = int((durs >= 0.5).sum())
    out["n_gaps_ge_1p0s"] = int((durs >= 1.0).sum())
    critical = {"torso_chest_back", "pelvis_waist", "head_neck", "thigh_knee", "shank_ankle", "foot"}
    if "body_region_group" in labeled.columns:
        crit = labeled[
            labeled["body_region_group"].isin(critical) & (durs >= 0.5)
        ]
        out["n_critical_region_gaps_ge_0p5s"] = len(crit)
    return out


def _qc_mask_status_pct(qc_mask: pd.DataFrame, status: str) -> float:
    if qc_mask.empty or "status" not in qc_mask.columns:
        return 0.0
    return round(100.0 * (qc_mask["status"] == status).sum() / len(qc_mask), 4)


def usable_after_remediation_pct(qc_mask: pd.DataFrame) -> float:
    """Estimated usable data after remediation: frames not lost to a >=0.5s gap or
    a marker-swap (both unrecoverable without manual fix). Short gaps and spikes are
    treated as recoverable (interpolation / Butterworth)."""
    if qc_mask.empty:
        return 100.0
    cols = qc_mask.columns
    unrec = pd.Series(False, index=qc_mask.index)
    if "flag_gap_0p5" in cols:
        unrec = unrec | qc_mask["flag_gap_0p5"].astype(bool)
    if "flag_segment_swap" in cols:
        unrec = unrec | qc_mask["flag_segment_swap"].astype(bool)
    return round(100.0 * (~unrec).sum() / len(qc_mask), 4)


def pct_frames_above_coverage(pct_time_below_coverage: float) -> float:
    """Frames where >= min_marker_coverage_pct of in-analysis markers are present."""
    return round(100.0 - float(pct_time_below_coverage or 0.0), 4)


def _gap_time_pct(gap_events: pd.DataFrame, duration_s: float, lo: float, hi: float) -> float:
    """Percent of session duration spent in labeled gaps with duration in [lo, hi)."""
    if gap_events.empty or duration_s <= 0:
        return 0.0
    labeled = gap_events[gap_events["is_labeled"]] if "is_labeled" in gap_events.columns else gap_events
    if labeled.empty:
        return 0.0
    durs = labeled["duration_seconds"].astype(float)
    sel = labeled[(durs >= lo) & (durs < hi)]
    if sel.empty:
        return 0.0
    return round(100.0 * float(sel["duration_seconds"].sum()) / duration_s, 4)


def _artifact_burden_summary(
    layer4: QCResult | None,
    session: MotiveSession,
    total_frames: int,
    config: dict[str, Any],
) -> str:
    """Narrative artifact burden: % candidate frames + top segment/marker share."""
    sigma = velocity_mad_sigma(config)
    pct_floor = velocity_percentile_threshold(config)
    sigma_tag = f"vel MAD σ={sigma:g}, pct={pct_floor:g}"
    if layer4 is None or total_frames <= 0:
        return f"0% candidate frames ({sigma_tag})"
    candidates = layer4.tables.get("artifact_candidates", pd.DataFrame())
    if candidates.empty or "frame" not in candidates.columns:
        return f"0% candidate frames ({sigma_tag})"

    pct_all = round(100.0 * candidates["frame"].astype(int).nunique() / total_frames, 2)
    inv = session.marker_inventory.set_index("marker_name")
    excl = excluded_body_groups(config)

    def _marker_segment(name: str) -> str:
        if name not in inv.index:
            return "unknown"
        return str(inv.loc[name].get("body_region_group", "unknown"))

    cand = candidates.copy()
    cand["frame"] = cand["frame"].astype(int)
    cand["body_region_group"] = cand["marker_name"].map(_marker_segment)
    if excl:
        cand = cand[~cand["body_region_group"].astype(str).isin(excl)]

    parts = [f"{pct_all}% candidate frames ({sigma_tag})"]
    if cand.empty:
        return parts[0]

    frames_by_seg: dict[str, set[int]] = {}
    frames_by_marker: dict[str, set[int]] = {}
    for _, row in cand.iterrows():
        f = int(row["frame"])
        seg = str(row["body_region_group"])
        mk = str(row["marker_name"])
        frames_by_seg.setdefault(seg, set()).add(f)
        frames_by_marker.setdefault(mk, set()).add(f)

    if frames_by_seg:
        top_seg = max(frames_by_seg, key=lambda s: len(frames_by_seg[s]))
        pct_seg = round(100.0 * len(frames_by_seg[top_seg]) / total_frames, 2)
        parts.append(f"top segment {top_seg} ({pct_seg}%)")
    if frames_by_marker:
        top_mk = max(frames_by_marker, key=lambda m: len(frames_by_marker[m]))
        pct_mk = round(100.0 * len(frames_by_marker[top_mk]) / total_frames, 2)
        parts.append(f"top marker {top_mk} ({pct_mk}%)")
    return "; ".join(parts)


def collect_markers_to_remove(
    layer2: QCResult,
    config: dict[str, Any],
) -> list[str]:
    """Markers to drop for the entire session (sustained dropout + never-solved non-phantom)."""
    sustained_sec = float(config.get("readiness", {}).get("sustained_dropout_seconds", 2.0))
    out: list[str] = []
    quar = layer2.tables.get("quarantined_markers", pd.DataFrame())
    if not quar.empty:
        for _, r in quar.iterrows():
            if str(r.get("quarantine_reason", "")) != "phantom_skeleton":
                out.append(str(r["marker_name"]))
    mq = layer2.tables.get("marker_quality_summary", pd.DataFrame())
    if not mq.empty and "included_in_analysis" in mq.columns:
        sustained = mq[
            mq["included_in_analysis"]
            & (mq.get("longest_gap_seconds", 0).astype(float) >= sustained_sec)
        ]
        out.extend(sustained["marker_name"].astype(str).tolist())
    return sorted(set(out))


def recommend_remediation(
    status: str,
    n_quarantined: int,
    n_sustained_dropout: int,
    n_segment_swap_events: int,
    n_gaps_0p2_to_0p5s: int,
    n_single_frame_events: int,
    *,
    analysis_skeleton: str = "",
    ignored_skeletons: list[str] | None = None,
    markers_to_remove: list[str] | None = None,
) -> str:
    """Plain-language remediation decision for the PI report."""
    actions: list[str] = []
    ignored = [s for s in (ignored_skeletons or []) if s]
    if analysis_skeleton and ignored:
        actions.append(
            f"use skeleton '{analysis_skeleton}' (ignore phantom skeleton(s): {', '.join(ignored)})"
        )
    elif n_quarantined > 0:
        actions.append(f"exclude {n_quarantined} phantom/never-solved marker(s)")
    remove = [m for m in (markers_to_remove or []) if m]
    if remove:
        if len(remove) == 1:
            actions.append(f"remove marker {remove[0]} from session")
        else:
            actions.append(f"remove markers {', '.join(remove)} from session")
    if n_segment_swap_events > 0:
        actions.append("manual review (marker swaps)")
    if n_gaps_0p2_to_0p5s > 0:
        actions.append("interpolate short gaps (0.2-0.5s)")
    if n_single_frame_events > 0:
        actions.append("Butterworth-filter velocity spikes")
    if not actions:
        return "ready (no remediation needed)"
    return "; ".join(actions)


def _candidate_frame_stats(layer4: QCResult | None, total_frames: int) -> dict[str, Any]:
    out = {"n_candidate_frames": 0, "pct_candidate_frames": 0.0}
    if layer4 is None:
        return out
    candidates = layer4.tables.get("artifact_candidates", pd.DataFrame())
    if candidates.empty or "frame" not in candidates.columns:
        return out
    n_unique = int(candidates["frame"].astype(int).nunique())
    out["n_candidate_frames"] = n_unique
    if total_frames > 0:
        out["pct_candidate_frames"] = round(100.0 * n_unique / total_frames, 4)
    return out


def _tuning_params(config: dict[str, Any]) -> dict[str, Any]:
    art = config.get("artifacts", {})
    methods = art.get("methods", {})
    spike = art.get("single_frame_spike", {})
    return {
        "velocity_mad_multiplier": art.get("velocity_mad_multiplier"),
        "velocity_percentile_threshold": art.get("velocity_percentile_threshold"),
        "spike_min_jump_distance_m": spike.get("min_jump_distance_m"),
        "acceleration_mad_enabled": bool(methods.get("acceleration_mad", False)),
        "constant_hold_enabled": bool(methods.get("constant_position_hold", False)),
    }


def extract_session_metrics_row(
    session_row: dict[str, Any],
    layer1: QCResult,
    layer2: QCResult,
    layer3: QCResult | None,
    layer4: QCResult | None,
    layer5: QCResult | None,
    config: dict[str, Any],
    *,
    batch_status: str = "ok",
    run_output_dir: str = "",
    error_message: str = "",
) -> dict[str, Any]:
    """Build one executive-summary row for dataset_eda_report.csv."""
    session = layer1.session
    assert session is not None
    md = session.metadata
    summ = layer2.tables.get("session_summary", pd.DataFrame())
    s = summ.iloc[0].to_dict() if not summ.empty else {}

    gap_events = layer2.tables.get("gap_events", pd.DataFrame())
    gap_ct = _gap_counts(gap_events)

    art_summary = (
        layer4.tables.get("artifact_session_summary", pd.DataFrame()).iloc[0].to_dict()
        if layer4 and not layer4.tables.get("artifact_session_summary", pd.DataFrame()).empty
        else {}
    )
    events = layer4.tables.get("artifact_events", pd.DataFrame()) if layer4 else pd.DataFrame()
    if not events.empty and "body_region_group" in events.columns:
        excl = excluded_body_groups(config)
        if excl:
            events = events[~events["body_region_group"].astype(str).isin(excl)]
    pct_near_gap = 0.0
    if not events.empty and "near_gap" in events.columns:
        pct_near_gap = round(100.0 * events["near_gap"].astype(bool).sum() / len(events), 4)

    w05 = layer3.tables.get("window_quality_0p5s", pd.DataFrame()) if layer3 else pd.DataFrame()
    w10 = layer3.tables.get("window_quality_1p0s", pd.DataFrame()) if layer3 else pd.DataFrame()

    mask = layer5.tables.get("analysis_frame_mask", pd.DataFrame()) if layer5 else pd.DataFrame()
    intervals = layer5.tables.get("qc_intervals", pd.DataFrame()) if layer5 else pd.DataFrame()
    qc_mask = layer5.tables.get("qc_mask", pd.DataFrame()) if layer5 else pd.DataFrame()
    gaps_over_05 = layer5.tables.get("gaps_over_0p5s", pd.DataFrame()) if layer5 else pd.DataFrame()
    n_segment_swap_events = 0
    if not events.empty and "method" in events.columns:
        n_segment_swap_events = int((events["method"] == "segment_length_violation").sum())

    unl = layer2.tables.get("unlabeled_marker_summary", pd.DataFrame())
    pct_unlabeled_frames = float(unl.iloc[0].get("percent_frames_with_any_unlabeled", 0.0)) if not unl.empty else 0.0

    worst_segment = ""
    if not events.empty and "body_region_group" in events.columns:
        counts = events["body_region_group"].value_counts()
        if not counts.empty:
            worst_segment = str(counts.index[0])

    dur_s = float(md.get("duration_seconds", 0) or 0)
    total_frames = int(md.get("total_frames_observed", 0))
    union_gap = float(s.get("union_gap_seconds_ge_0p2_labeled") or 0.0)
    pct_gap_time = round(100.0 * union_gap / dur_s, 4) if dur_s > 0 else 0.0
    cand_stats = _candidate_frame_stats(layer4, total_frames)
    skeleton_events = md.get("skeleton_selection_events") or []
    ignored_skeletons = sorted({str(e.get("ignored_skeleton", "")) for e in skeleton_events if e.get("ignored_skeleton")})
    analysis_skeleton = str(md.get("analysis_skeleton_prefix", "") or "")
    markers_to_remove = collect_markers_to_remove(layer2, config) if batch_status == "ok" else []
    pct_below = float(s.get("pct_time_below_coverage", 0) or 0)
    pct_usable_frames = pct_frames_above_coverage(pct_below)
    pct_gap_02_05 = _gap_time_pct(gap_events, dur_s, 0.2, 0.5)
    artifact_summary = _artifact_burden_summary(layer4, session, total_frames, config)

    row: dict[str, Any] = {
        "subject_id": session_row.get("subject_id", md.get("subject_id")),
        "session_id": session_row.get("session_id", md.get("session_id")),
        "file_name": session_row.get("file_name", md.get("file_name")),
        "batch_status": batch_status,
        "run_output_dir": run_output_dir,
        "error_message": error_message,
        "duration_seconds": round(dur_s, 3),
        "duration_minutes": round(dur_s / 60.0, 3),
        "total_frames_observed": int(md.get("total_frames_observed", 0)),
        "effective_frame_rate_hz": md.get("effective_frame_rate_hz"),
        "n_labeled_markers": int(md.get("n_labeled_markers", 0)),
        "n_labeled_markers_in_analysis": int(md.get("n_labeled_markers_in_analysis", 0)),
        "n_quarantined_markers": int(md.get("n_quarantined_markers", 0)),
        "analysis_skeleton_prefix": analysis_skeleton,
        "ignored_skeleton_prefixes": ";".join(ignored_skeletons),
        "frame_rate_status": md.get("frame_rate_status"),
        "validation_status": md.get("validation_status"),
        "missing_percent_labeled": s.get("missing_percent_labeled"),
        "labeled_marker_coverage_mean_pct": s.get("labeled_marker_coverage_mean_pct"),
        "pct_time_below_coverage": s.get("pct_time_below_coverage"),
        "n_markers_sustained_dropout": s.get("n_markers_sustained_dropout"),
        **gap_ct,
        "n_markers_gaps_ge_0p5s": int(len(gaps_over_05)),
        "raw_qc_preprocessing_status": s.get("raw_qc_preprocessing_status"),
        "raw_qc_status_reason": s.get("raw_qc_status_reason"),
        "union_gap_seconds_ge_0p2_labeled": s.get("union_gap_seconds_ge_0p2_labeled"),
        "pct_session_gap_time_ge_0p2": pct_gap_time,
        "longest_gap_seconds_labeled": s.get("longest_gap_seconds_labeled"),
        "n_artifact_events": art_summary.get("n_events", 0),
        **cand_stats,
        "n_single_frame_events": art_summary.get("n_single_frame_events", 0),
        "n_short_burst_events": art_summary.get("n_short_burst_events", 0),
        "n_sustained_events": art_summary.get("n_sustained_events", 0),
        "n_segment_swap_events": n_segment_swap_events,
        "pct_artifact_events_near_gap": pct_near_gap,
        "worst_artifact_body_segment": worst_segment,
        "pct_frames_use": _qc_mask_status_pct(qc_mask, "use"),
        "pct_frames_caution": _qc_mask_status_pct(qc_mask, "caution"),
        "pct_frames_exclude": _qc_mask_status_pct(qc_mask, "exclude"),
        "pct_frames_above_coverage": pct_usable_frames,
        "usable_after_remediation_pct": pct_usable_frames,
        "pct_gap_time_0p2_to_0p5": pct_gap_02_05,
        "artifact_burden_summary": artifact_summary,
        "markers_to_remove": ";".join(markers_to_remove),
        "recommended_remediation": recommend_remediation(
            str(s.get("raw_qc_preprocessing_status", "")),
            int(md.get("n_quarantined_markers", 0)),
            int(s.get("n_markers_sustained_dropout", 0) or 0),
            n_segment_swap_events,
            int(gap_ct.get("n_gaps_0p2_to_0p5s", 0)),
            int(art_summary.get("n_single_frame_events", 0)),
            analysis_skeleton=analysis_skeleton,
            ignored_skeletons=ignored_skeletons,
            markers_to_remove=markers_to_remove,
        ),
        "window_yield_0p5s_pct": compute_window_yield(w05),
        "window_yield_1p0s_pct": compute_window_yield(w10),
        "n_qc_intervals": len(intervals),
        "pct_unlabeled_frames": pct_unlabeled_frames,
        **_tuning_params(config),
    }
    return row


def extract_top_markers(
    session_row: dict[str, Any],
    layer2: QCResult,
    layer4: QCResult | None,
    config: dict[str, Any],
    top_n: int = 3,
) -> pd.DataFrame:
    mq = layer2.tables.get("marker_quality_summary", pd.DataFrame())
    if mq.empty:
        return pd.DataFrame()

    labeled = mq[mq["is_labeled"]].copy() if "is_labeled" in mq.columns else mq.copy()
    if "included_in_analysis" in labeled.columns:
        labeled = labeled[labeled["included_in_analysis"]]
    else:
        excl = excluded_body_groups(config)
        if excl and "body_region_group" in labeled.columns:
            labeled = labeled[~labeled["body_region_group"].astype(str).isin(excl)]
    events = layer4.tables.get("artifact_events", pd.DataFrame()) if layer4 else pd.DataFrame()
    art_by_marker: dict[str, int] = {}
    if not events.empty and "marker_name" in events.columns:
        art_by_marker = events.groupby("marker_name").size().to_dict()

    labeled["n_artifact_events"] = labeled["marker_name"].map(art_by_marker).fillna(0).astype(int)
    labeled["rank_score"] = labeled["missing_percent"].astype(float) + labeled["n_artifact_events"] * 0.5
    labeled = labeled.sort_values("rank_score", ascending=False).head(top_n)

    rows = []
    for rank, (_, r) in enumerate(labeled.iterrows(), start=1):
        rows.append(
            {
                "subject_id": session_row.get("subject_id"),
                "session_id": session_row.get("session_id"),
                "rank": rank,
                "marker_name": r["marker_name"],
                "body_region_group": r.get("body_region_group", ""),
                "missing_percent": r.get("missing_percent"),
                "n_gaps": r.get("n_gaps_total", 0),
                "n_artifact_events": int(r["n_artifact_events"]),
                "quality_label": r.get("quality_label", ""),
            }
        )
    return pd.DataFrame(rows)


def extract_artifact_type_distribution(
    session_row: dict[str, Any],
    layer4: QCResult | None,
) -> pd.DataFrame:
    events = layer4.tables.get("artifact_events", pd.DataFrame()) if layer4 else pd.DataFrame()
    if events.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    n_total = len(events)
    for dim in ("method", "event_class"):
        if dim not in events.columns:
            continue
        counts = events[dim].value_counts()
        for value, count in counts.items():
            rows.append(
                {
                    "subject_id": session_row.get("subject_id"),
                    "session_id": session_row.get("session_id"),
                    "dimension": dim,
                    "category": str(value),
                    "count": int(count),
                    "proportion": round(100.0 * count / n_total, 4),
                }
            )
    return pd.DataFrame(rows)


def extract_velocity_by_body_segment(
    session_row: dict[str, Any],
    session: MotiveSession,
    config: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group in list_velocity_histogram_groups(session, config):
        if group == "all_labeled":
            continue
        dist = collect_session_velocity_distribution(session, config, group)
        speeds = np.asarray(dist.get("speeds_m_s", []), dtype=float)
        if speeds.size == 0:
            continue
        rows.append(
            {
                "subject_id": session_row.get("subject_id"),
                "session_id": session_row.get("session_id"),
                "body_region_group": group,
                "mean_speed_m_s": round(float(np.mean(speeds)), 6),
                "max_speed_m_s": round(float(np.max(speeds)), 6),
                "p99_speed_m_s": round(float(np.percentile(speeds, 99)), 6),
                "n_speed_samples": int(speeds.size),
                "n_markers": int(dist.get("n_markers", 0)),
            }
        )
    return pd.DataFrame(rows)


def extract_gap_windows(
    layer3: QCResult | None,
    *,
    min_gap_s: float = 0.2,
) -> pd.DataFrame:
    """0.5 s windows where max overlapping labeled gap duration >= min_gap_s."""
    if layer3 is None:
        return pd.DataFrame()
    wdf = layer3.tables.get("window_quality_0p5s", pd.DataFrame())
    if wdf.empty or "max_gap_duration_s" not in wdf.columns:
        return pd.DataFrame()
    flagged = wdf[wdf["max_gap_duration_s"].astype(float) >= min_gap_s].copy()
    if flagged.empty:
        return pd.DataFrame()
    flagged = flagged.sort_values("start_time_s")
    out = pd.DataFrame(
        {
            "start_second": flagged["start_time_s"],
            "end_second": flagged["end_time_s"],
            "max_gap_duration_s": flagged["max_gap_duration_s"],
            "window_label": flagged["window_quality_label"],
        }
    )
    if "worst_gap_marker" in flagged.columns:
        out["worst_gap_marker"] = flagged["worst_gap_marker"]
    if "affected_body_groups" in flagged.columns:
        out["affected_body_groups"] = flagged["affected_body_groups"]
    return out


def extract_artifact_intervals(layer4: QCResult | None) -> pd.DataFrame:
    """Artifact event intervals with start/end in seconds."""
    if layer4 is None:
        return pd.DataFrame()
    events = layer4.tables.get("artifact_events", pd.DataFrame())
    if events.empty:
        return pd.DataFrame()
    events = events.sort_values("start_time_s")
    out = pd.DataFrame(
        {
            "start_second": events["start_time_s"],
            "end_second": events["end_time_s"],
            "duration_seconds": events["duration_seconds"],
            "marker_name": events["marker_name"],
            "event_class": events["event_class"],
            "method": events["method"],
        }
    )
    if "body_region_group" in events.columns:
        out["body_region_group"] = events["body_region_group"]
    return out


def extract_segment_burden(
    session_row: dict[str, Any],
    layer2: QCResult,
    layer4: QCResult | None,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Per body segment: gap counts and artifact event counts."""
    gap_events = layer2.tables.get("gap_events", pd.DataFrame())
    events = layer4.tables.get("artifact_events", pd.DataFrame()) if layer4 else pd.DataFrame()
    excl = excluded_body_groups(config)
    if excl:
        if not gap_events.empty and "body_region_group" in gap_events.columns:
            gap_events = gap_events[~gap_events["body_region_group"].astype(str).isin(excl)]
        if not events.empty and "body_region_group" in events.columns:
            events = events[~events["body_region_group"].astype(str).isin(excl)]

    segments: set[str] = set()
    gap_by_seg: dict[str, pd.DataFrame] = {}
    if not gap_events.empty and "body_region_group" in gap_events.columns:
        labeled = gap_events[gap_events["is_labeled"]] if "is_labeled" in gap_events.columns else gap_events
        for seg, grp in labeled.groupby("body_region_group"):
            seg_s = str(seg)
            segments.add(seg_s)
            gap_by_seg[seg_s] = grp

    art_by_seg: dict[str, int] = {}
    if not events.empty and "body_region_group" in events.columns:
        art_by_seg = {str(k): int(v) for k, v in events.groupby("body_region_group").size().items()}
        segments.update(art_by_seg.keys())

    rows: list[dict[str, Any]] = []
    for seg in sorted(segments):
        gaps = gap_by_seg.get(seg, pd.DataFrame())
        n_gaps = len(gaps)
        durs = gaps["duration_seconds"].astype(float) if not gaps.empty else pd.Series(dtype=float)
        n_ge_02 = int((durs >= 0.2).sum()) if not durs.empty else 0
        n_ge_05 = int((durs >= 0.5).sum()) if not durs.empty else 0
        gap_sec = round(float(durs.sum()), 6) if not durs.empty else 0.0
        n_art = int(art_by_seg.get(seg, 0))
        if n_gaps > 0 and n_art > 0:
            flag = "both"
        elif n_gaps > 0:
            flag = "gap"
        elif n_art > 0:
            flag = "artifact"
        else:
            flag = ""
        rows.append(
            {
                "session": f"{session_row.get('subject_id')}_{session_row.get('session_id')}",
                "body_region_group": seg,
                "n_gaps": n_gaps,
                "n_gaps_ge_0p2s": n_ge_02,
                "n_gaps_ge_0p5s": n_ge_05,
                "gap_seconds_total": gap_sec,
                "n_artifact_events": n_art,
                "flag": flag,
            }
        )
    return pd.DataFrame(rows)


def extract_session_details(
    session_row: dict[str, Any],
    layer1: QCResult,
    layer2: QCResult,
    layer3: QCResult | None,
    layer4: QCResult | None,
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    session = layer1.session
    assert session is not None
    return {
        "top_markers": extract_top_markers(session_row, layer2, layer4, config),
        "artifact_types": extract_artifact_type_distribution(session_row, layer4),
        "velocity_by_segment": extract_velocity_by_body_segment(session_row, session, config),
    }
