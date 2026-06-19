"""Layer 5: publication-ready report package."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from motive_qc.analysis_scope import (
    filter_artifact_events_for_analysis,
    filter_gap_events_for_analysis,
)
from motive_qc.core import QCMessage, QCResult
from motive_qc.deliverables import (
    build_artifacts_by_segment,
    build_gaps_over_threshold,
    build_qc_mask,
)
from motive_qc.marker_meta import is_unlabeled_marker
from motive_qc.reason_codes import (
    build_reason_codes_markdown,
    primary_reason_code,
    reason_codes_to_human,
)

UNLABELED_BODY_GROUP = "unlabeled"


def critical_region_large_gaps(gap_events: pd.DataFrame, config: dict[str, Any]) -> bool:
    if gap_events.empty:
        return False
    critical_groups = config.get("frame_quality", {}).get("critical_groups", [])
    large_thr = config["gaps"]["thresholds_seconds"]["large_gap"]
    labeled_large = filter_gap_events_for_analysis(gap_events, config)
    labeled_large = labeled_large[
        labeled_large["is_labeled"] & (labeled_large["duration_seconds"] >= large_thr)
    ]
    if labeled_large.empty or not critical_groups:
        return False
    return bool(labeled_large["body_region_group"].isin(critical_groups).any())


def _primary_window_table(layer3_result: QCResult | None) -> pd.DataFrame:
    if layer3_result is None:
        return pd.DataFrame()
    return layer3_result.tables.get("window_quality_0p5s", pd.DataFrame())


def build_analysis_frame_mask(
    layer2_result: QCResult,
    layer3_result: QCResult | None,
    config: dict[str, Any],
) -> pd.DataFrame:
    frame_mask = layer2_result.tables.get("frame_qc_mask", pd.DataFrame()).copy()
    if frame_mask.empty:
        return frame_mask

    windows = _primary_window_table(layer3_result)
    frames = frame_mask["frame"].values.astype(int)
    analysis_status = frame_mask["qc_status"].astype(str).tolist()
    reason_codes = frame_mask.get("reason_codes", pd.Series([""] * len(frame_mask))).astype(str).tolist()

    if not windows.empty:
        for _, win in windows.iterrows():
            in_win = (frames >= int(win["start_frame"])) & (frames <= int(win["end_frame"]))
            wlabel = win["window_quality_label"]
            wreason = str(win.get("reason_codes", ""))
            for i in np.where(in_win)[0]:
                if wlabel == "exclude_or_review":
                    analysis_status[i] = "exclude_or_review"
                    reason_codes[i] = _merge_reasons(reason_codes[i], wreason)
                elif wlabel == "caution" and analysis_status[i] == "use":
                    analysis_status[i] = "caution"
                    reason_codes[i] = _merge_reasons(reason_codes[i], wreason)

    out = frame_mask.copy()
    out["analysis_status"] = analysis_status
    out["analysis_reason_codes"] = reason_codes
    return out


def _merge_reasons(existing: str, new: str) -> str:
    parts = [p for p in f"{existing};{new}".split(";") if p.strip()]
    return ";".join(dict.fromkeys(parts))


def _labeled_gap_events(gap_events: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if gap_events.empty or "is_labeled" not in gap_events.columns:
        return gap_events
    gaps = filter_gap_events_for_analysis(gap_events, config)
    return gaps[gaps["is_labeled"]]


def _labeled_artifact_events(
    artifact_events: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    if artifact_events.empty:
        return artifact_events
    events = filter_artifact_events_for_analysis(artifact_events, config)
    if "body_region_group" in events.columns:
        return events[
            events["body_region_group"].astype(str) != UNLABELED_BODY_GROUP
        ]
    return events


def _clean_body_groups(groups: str) -> str:
    parts = [
        g
        for g in str(groups).split(";")
        if g and g != "nan" and g != UNLABELED_BODY_GROUP
    ]
    return ";".join(sorted(dict.fromkeys(parts)))


def _body_groups_for_interval(
    start_frame: int,
    end_frame: int,
    gap_events: pd.DataFrame,
    artifact_events: pd.DataFrame,
    config: dict[str, Any],
) -> str:
    groups: set[str] = set()
    labeled_gaps = _labeled_gap_events(gap_events, config)
    if not labeled_gaps.empty:
        gaps = labeled_gaps[
            (labeled_gaps["gap_end_frame"] >= start_frame)
            & (labeled_gaps["gap_start_frame"] <= end_frame)
        ]
        groups.update(gaps["body_region_group"].astype(str).tolist())
    labeled_events = _labeled_artifact_events(artifact_events, config)
    if not labeled_events.empty:
        ev = labeled_events[
            (labeled_events["end_frame"] >= start_frame)
            & (labeled_events["start_frame"] <= end_frame)
        ]
        groups.update(ev["body_region_group"].astype(str).tolist())
    return _clean_body_groups(";".join(groups))


def _interval_has_labeled_evidence(
    start_frame: int,
    end_frame: int,
    reason: str,
    n_artifact_events: int,
    gap_events: pd.DataFrame,
    config: dict[str, Any],
) -> bool:
    if n_artifact_events > 0:
        return True
    labeled_gaps = _labeled_gap_events(gap_events, config)
    if not labeled_gaps.empty:
        overlap = labeled_gaps[
            (labeled_gaps["gap_end_frame"] >= start_frame)
            & (labeled_gaps["gap_start_frame"] <= end_frame)
        ]
        if not overlap.empty:
            return True
    reason_upper = str(reason).upper()
    return any(
        token in reason_upper
        for token in (
            "ARTIFACT",
            "MISSING",
            "WINDOW",
            "SUSTAINED",
            "HIGH_",
            "ELEVATED_",
        )
    )


def _finalize_qc_intervals(
    intervals: pd.DataFrame,
    gap_events: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    if intervals.empty:
        return intervals
    rows: list[dict[str, Any]] = []
    labeled_gaps = _labeled_gap_events(gap_events, config)
    for _, row in intervals.iterrows():
        dominant = row.get("dominant_gap_marker")
        if dominant is not None and pd.notna(dominant):
            if is_unlabeled_marker(str(dominant), config):
                dominant = None
        cleaned_groups = _clean_body_groups(row.get("affected_body_groups", ""))
        if not _interval_has_labeled_evidence(
            int(row["start_frame"]),
            int(row["end_frame"]),
            str(row.get("reason", "")),
            int(row.get("n_artifact_events", 0)),
            gap_events,
            config,
        ):
            continue
        out = row.to_dict()
        out["affected_body_groups"] = cleaned_groups
        out["dominant_gap_marker"] = dominant
        rows.append(out)
    return pd.DataFrame(rows)


def build_qc_intervals(
    analysis_mask: pd.DataFrame,
    gap_events: pd.DataFrame,
    artifact_events: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    if analysis_mask.empty or "analysis_status" not in analysis_mask.columns:
        return pd.DataFrame(
            columns=[
                "start_frame",
                "end_frame",
                "start_time_s",
                "end_time_s",
                "duration_s",
                "status",
                "reason",
                "primary_reason_code",
                "reason_human",
                "affected_body_groups",
                "n_artifact_events",
                "dominant_gap_marker",
                "recommended_bvh_action",
            ]
        )

    rows: list[dict[str, Any]] = []
    frames = analysis_mask["frame"].values
    times = analysis_mask["time_seconds"].values
    statuses = analysis_mask["analysis_status"].values
    reasons = analysis_mask["analysis_reason_codes"].values

    i = 0
    n = len(frames)
    min_duration_frames = int(config.get("reporting", {}).get("min_interval_frames", 1))

    while i < n:
        if statuses[i] == "use":
            i += 1
            continue
        start_i = i
        while i < n and statuses[i] == statuses[start_i]:
            i += 1
        end_i = i - 1
        if end_i - start_i + 1 < min_duration_frames:
            continue

        start_frame = int(frames[start_i])
        end_frame = int(frames[end_i])
        reason = str(reasons[start_i])
        status = statuses[start_i]
        action = "review" if status == "caution" else "exclude_from_bvh_analysis"

        labeled_gaps = _labeled_gap_events(gap_events, config)
        gaps_in = labeled_gaps[
            (labeled_gaps["gap_end_frame"] >= start_frame)
            & (labeled_gaps["gap_start_frame"] <= end_frame)
        ] if not labeled_gaps.empty else pd.DataFrame()
        labeled_events = _labeled_artifact_events(artifact_events, config)
        ev_in = labeled_events[
            (labeled_events["end_frame"] >= start_frame)
            & (labeled_events["start_frame"] <= end_frame)
        ] if not labeled_events.empty else pd.DataFrame()

        dominant_gap = None
        if not gaps_in.empty:
            dominant_gap = gaps_in.loc[gaps_in["duration_seconds"].idxmax(), "marker_name"]

        rows.append(
            {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time_s": float(times[start_i]),
                "end_time_s": float(times[end_i]),
                "duration_s": round(float(times[end_i] - times[start_i]), 6),
                "status": status,
                "reason": reason,
                "primary_reason_code": primary_reason_code(reason),
                "reason_human": reason_codes_to_human(reason),
                "affected_body_groups": _body_groups_for_interval(
                    start_frame, end_frame, gap_events, artifact_events, config
                ),
                "n_artifact_events": len(ev_in),
                "dominant_gap_marker": dominant_gap,
                "recommended_bvh_action": action,
            }
        )
    return pd.DataFrame(rows)


def build_analysis_mask_summary(analysis_mask: pd.DataFrame) -> pd.DataFrame:
    if analysis_mask.empty or "analysis_status" not in analysis_mask.columns:
        return pd.DataFrame(columns=["status", "n_frames", "source"])
    counts = analysis_mask["analysis_status"].value_counts()
    return pd.DataFrame(
        [{"status": k, "n_frames": int(v), "source": "analysis_frame_mask"} for k, v in counts.items()]
    )


def build_qc_report_markdown(
    layer1: QCResult,
    layer2: QCResult,
    layer3: QCResult | None,
    layer4: QCResult | None,
    layer5_tables: dict[str, pd.DataFrame],
    messages: list[QCMessage],
    config: dict[str, Any],
) -> str:
    session = layer1.session
    assert session is not None
    md = session.metadata
    summary = layer2.tables["session_summary"].iloc[0]
    tables = layer2.tables
    unlabeled = tables.get("unlabeled_marker_summary", pd.DataFrame())
    unl_row = unlabeled.iloc[0] if not unlabeled.empty else {}
    gap_events = tables.get("gap_events", pd.DataFrame())
    marker_quality = tables.get("marker_quality_summary", pd.DataFrame())
    labeled_with_missing = 0
    if not marker_quality.empty:
        labeled_with_missing = int(
            marker_quality[marker_quality["is_labeled"] & (marker_quality["n_missing_frames"] > 0)].shape[0]
        )
    critical_large = critical_region_large_gaps(gap_events, config)
    export_type = md["raw_data_status"]
    if md.get("contains_rigid_body_columns") or md.get("contains_skeleton_columns"):
        export_type = "mixed"
    overall_qc = summary.get("raw_qc_preprocessing_status", md["validation_status"])

    mask_summary = layer5_tables.get("analysis_mask_summary", pd.DataFrame())
    intervals = layer5_tables.get("qc_intervals", pd.DataFrame())
    win_summary = (
        layer3.tables.get("window_quality_summary", pd.DataFrame()) if layer3 else pd.DataFrame()
    )
    art_summary = (
        layer4.tables.get("artifact_session_summary", pd.DataFrame()).iloc[0].to_dict()
        if layer4 and not layer4.tables.get("artifact_session_summary", pd.DataFrame()).empty
        else {}
    )
    events = layer4.tables.get("artifact_events", pd.DataFrame()) if layer4 else pd.DataFrame()

    lines = [
        "# Raw Motive Marker QC Report",
        "",
        "## Layer guide",
        "",
        "- **Layer 2:** per-marker gaps and missingness (evidence).",
        "- **Layer 4:** kinematic artifact **events** on gap-safe segments (candidates only).",
        "- **Layer 3:** fixed 0.5 s windows — final **safe for PCA/jPCA?** verdict using L2 + L4.",
        "",
        "---",
        "",
        "## 1. Session and export identity",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Session ID | `{md['session_id']}` |",
        f"| Input file | `{md['file_name']}` |",
        f"| Motive version | `{md['motive_version']}` |",
        f"| Export type detected | `{export_type}` |",
        f"| Frame rate | `{md['effective_frame_rate_hz']}` Hz |",
        f"| Frame range | `{md['start_frame']}-{md['end_frame']}` |",
        f"| Duration | `{md['duration_seconds']:.3f}` seconds |",
        f"| Units | `{md.get('length_units') or 'unknown'}` |",
        f"| Labeled markers | `{md['n_labeled_markers']}` |",
        f"| Unlabeled marker tracks | `{md['n_unlabeled_markers']}` |",
        f"| Overall QC status | `{overall_qc}` |",
        "",
        "---",
        "",
        "## 2. Marker completeness and gap structure",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Labeled marker missingness | `{summary['missing_percent_labeled']}%` |",
        f"| Markers with any missing frames | `{labeled_with_missing}` |",
        f"| Total continuous labeled-marker gaps | `{summary['n_gaps_total_labeled']}` |",
        f"| Gaps >=0.2 s | `{summary['n_gaps_ge_0p2s_labeled']}` |",
        f"| Gaps >=0.5 s | `{summary['n_gaps_ge_0p5s_labeled']}` |",
        f"| Longest labeled-marker gap | `{summary['longest_gap_seconds_labeled']}` s |",
        f"| Critical body-region large gaps | `{'yes' if critical_large else 'no'}` |",
        "",
        f"**Key finding:** {summary['raw_qc_status_reason']}",
        "",
        "---",
        "",
        "## 3. Unlabeled-marker burden",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Frames with any unlabeled marker | `{unl_row.get('frames_with_any_unlabeled', 0)}` |",
        f"| Percent frames with unlabeled markers | `{unl_row.get('percent_frames_with_any_unlabeled', 0.0)}%` |",
        f"| Max unlabeled markers in one frame | `{unl_row.get('max_unlabeled_markers_in_frame', 0)}` |",
        f"| Longest unlabeled burst | `{unl_row.get('longest_unlabeled_burst_sec', 0.0)}` s |",
        "",
        "---",
        "",
        "## 4. Candidate artifact screening (labeled markers only)",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Artifact events | `{art_summary.get('n_events', 0)}` |",
        f"| Single-frame events | `{art_summary.get('n_single_frame_events', 0)}` |",
        f"| Short bursts (2-5 frames) | `{art_summary.get('n_short_burst_events', 0)}` |",
        f"| Sustained events (>5 frames) | `{art_summary.get('n_sustained_events', 0)}` |",
        f"| Frames with velocity candidate | `{art_summary.get('n_frames_velocity_candidate', 0)}` |",
        f"| Frames with acceleration candidate | `{art_summary.get('n_frames_acceleration_candidate', 0)}` |",
        f"| Frames with **both** vel and accel | `{art_summary.get('n_frames_both_velocity_and_acceleration', 0)}` |",
        "",
        f"**Interpretation:** {art_summary.get('recommendation', 'No artifact screening run.')}",
        "",
    ]

    if not win_summary.empty:
        lines.extend(["---", "", "## 5. Analysis window safety (0.5 s)", ""])
        for _, row in win_summary.iterrows():
            lines.append(
                f"- **{row['window_length_s']} s windows:** {row['n_windows']} total; "
                f"{row['n_with_gap_overlap']} with gap overlap; "
                f"{row['n_with_artifact_events']} with artifact events; "
                f"{row['n_flagged_caution']} caution, {row['n_flagged_exclude']} exclude."
            )
        lines.append("")

    lines.extend(["---", "", "## 6. BVH analysis mask", ""])
    if not mask_summary.empty:
        lines.append("| Status | Number of frames | Meaning |")
        lines.append("|---|---:|---|")
        for _, row in mask_summary.iterrows():
            lines.append(f"| {row['status']} | {row['n_frames']} | Merged L2+L3 frame mask |")
    else:
        lines.append("Frame mask not computed.")

    lines.extend(
        [
            "",
            "### Exclusion/caution intervals (labeled markers only; unlabeled excluded)",
            "",
        ]
    )
    if not intervals.empty:
        lines.append(
            "| Start | End | Duration | Status | Body groups | Reason | Action |"
        )
        lines.append("|---:|---:|---:|---|---|---|---|")
        for _, row in intervals.head(30).iterrows():
            body_groups = _clean_body_groups(row.get("affected_body_groups", ""))
            lines.append(
                f"| {row['start_frame']} | {row['end_frame']} | {row['duration_s']} | "
                f"{row['status']} | {body_groups} | "
                f"{row.get('reason_human', row['reason'])} | {row['recommended_bvh_action']} |"
            )
    else:
        lines.append("No caution or exclude intervals identified.")

    lines.extend(["", "---", "", "## 7. Reason code glossary", ""])
    lines.append("See `qc_reason_codes.md` in the run folder for full code definitions.")

    conclusion = (
        f"Raw marker-level QC status is `{overall_qc}`. "
        f"Artifact screening found {len(events)} events; "
        f"{art_summary.get('n_frames_both_velocity_and_acceleration', 0)} frames had both velocity and acceleration candidates. "
        "Use window quality tables and qc_intervals for PCA/jPCA planning."
    )
    lines.extend(["", "---", "", "## Final QC conclusion", "", conclusion, ""])

    if messages:
        lines.extend(["", "## Validation messages", ""])
        for msg in messages:
            lines.append(f"- **[{msg.severity}] {msg.code}:** {msg.message}")

    return "\n".join(lines) + "\n"


def write_markdown_summary_l12(
    path: Path,
    layer1_result: QCResult,
    layer2_result: QCResult,
    messages: list[QCMessage],
    config: dict[str, Any],
) -> None:
    text = build_qc_report_markdown(
        layer1_result,
        layer2_result,
        None,
        None,
        layer2_result.tables,
        messages,
        config,
    )
    path.write_text(text, encoding="utf-8")


def run_layer5_report(
    layer1: QCResult,
    layer2: QCResult,
    layer3: QCResult | None,
    layer4: QCResult | None,
    config: dict[str, Any],
    verbose: bool = False,
) -> QCResult:
    gap_events = layer2.tables.get("gap_events", pd.DataFrame())
    artifact_events = (
        layer4.tables.get("artifact_events", pd.DataFrame()) if layer4 else pd.DataFrame()
    )

    analysis_mask = build_analysis_frame_mask(layer2, layer3, config)
    intervals = build_qc_intervals(analysis_mask, gap_events, artifact_events, config)
    intervals = _finalize_qc_intervals(intervals, gap_events, config)
    mask_summary = build_analysis_mask_summary(analysis_mask)

    thresholds = config["gaps"]["thresholds_seconds"]
    moderate = float(thresholds["moderate_gap"])
    large = float(thresholds["large_gap"])
    duration_s = float(layer1.session.metadata.get("duration_seconds", 0.0))

    gaps_over_0p5s = build_gaps_over_threshold(gap_events, config, large)
    gaps_over_0p2s = build_gaps_over_threshold(gap_events, config, moderate, upper_seconds=large)
    artifacts_by_segment = build_artifacts_by_segment(artifact_events, config, duration_s)
    qc_mask, qc_mask_intervals = build_qc_mask(
        layer1.session, gap_events, artifact_events, config
    )
    segment_length_qc = (
        layer4.tables.get("segment_length_qc", pd.DataFrame()) if layer4 else pd.DataFrame()
    )

    tables = {
        "analysis_frame_mask": analysis_mask,
        "qc_intervals": intervals,
        "analysis_mask_summary": mask_summary,
        "gaps_over_0p5s": gaps_over_0p5s,
        "gaps_over_0p2s": gaps_over_0p2s,
        "artifacts_by_segment": artifacts_by_segment,
        "segment_length_qc": segment_length_qc,
        "qc_mask": qc_mask,
        "qc_mask_intervals": qc_mask_intervals,
    }

    all_messages = list(layer1.messages)
    for result in (layer2, layer3, layer4):
        if result:
            all_messages.extend(result.messages)

    return QCResult(
        layer_name="layer5",
        status="pass",
        tables=tables,
        messages=all_messages,
        session=layer1.session,
        files_written=[],
    )


def write_reason_codes_file(path: Path) -> None:
    path.write_text(build_reason_codes_markdown(), encoding="utf-8")
