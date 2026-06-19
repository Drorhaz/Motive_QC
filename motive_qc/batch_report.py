"""Build PI-facing Layer 6 executive reports and comparison plots."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from motive_qc.core import __version__
from motive_qc.batch_plots import write_batch_session_plots
from motive_qc.batch_workbook import write_pi_workbook, build_artifact_sigma_sensitivity
from motive_qc.artifacts import (
    artifact_candidate_severity_note,
    velocity_mad_sigma,
    velocity_percentile_threshold,
)
from motive_qc.deliverables import enrich_mask_intervals


def _session_label(row: pd.Series) -> str:
    return f"{row.get('subject_id', '')}_{row.get('session_id', '')}"


def build_dataset_eda_markdown(
    eda_df: pd.DataFrame,
    top_markers: pd.DataFrame,
    artifact_types: pd.DataFrame,
    velocity_df: pd.DataFrame,
    failures_df: pd.DataFrame,
    batch_dir: Path,
    config: dict[str, Any],
) -> str:
    n = len(eda_df)
    ok_df = eda_df[eda_df["batch_status"] == "ok"] if not eda_df.empty else eda_df
    subjects = sorted(eda_df["subject_id"].unique().tolist()) if not eda_df.empty else []

    status_counts = (
        ok_df["raw_qc_preprocessing_status"].value_counts().to_dict() if not ok_df.empty else {}
    )
    median_usable = (
        float(ok_df["pct_frames_above_coverage"].median())
        if not ok_df.empty and "pct_frames_above_coverage" in ok_df.columns
        else float(ok_df["usable_after_remediation_pct"].median())
        if not ok_df.empty and "usable_after_remediation_pct" in ok_df.columns
        else 0.0
    )
    total_quarantined = (
        int(ok_df["n_quarantined_markers"].sum())
        if not ok_df.empty and "n_quarantined_markers" in ok_df.columns
        else 0
    )

    dom_artifact = "none"
    if not artifact_types.empty:
        by_class = artifact_types[artifact_types["dimension"] == "event_class"]
        if not by_class.empty:
            dom_artifact = str(by_class.sort_values("count", ascending=False).iloc[0]["category"])

    art_cfg = config.get("artifacts", {})
    lines = [
        "# Dataset EDA & QC Executive Report",
        "",
        f"**Generated:** {datetime.now().isoformat(timespec='seconds')}  ",
        f"**Motive_QC version:** {__version__}  ",
        f"**Batch folder:** `{batch_dir}`  ",
        f"**Prepared for:** PI review (cross-session raw marker QC)",
        "",
        "---",
        "",
        "## 1. Executive summary",
        "",
        f"- **Sessions processed:** {n} across subject(s) {', '.join(subjects) or 'n/a'}",
        f"- **Successful runs:** {len(ok_df)}; **failures:** {len(failures_df)}",
        f"- **Preprocessing status distribution:** {status_counts or 'n/a'}",
        f"- **Median usable frame % (coverage-based):** {median_usable:.1f}%",
        f"- **Phantom/never-solved markers quarantined (dataset-wide):** {total_quarantined}",
        f"- **Dominant artifact event class (dataset-wide):** {dom_artifact}",
        f"- **Artifact tuning:** velocity MAD σ={art_cfg.get('velocity_mad_multiplier')}, "
        f"velocity pct={art_cfg.get('velocity_percentile_threshold')}, "
        f"spike jump={art_cfg.get('single_frame_spike', {}).get('min_jump_distance_m')} m",
        "",
        "> Raw marker QC only — no gap filling, smoothing, or automatic exclusions. "
        "Artifact detections are **candidates** requiring visual review.",
        "",
        "---",
        "",
        "## 2. Dataset inventory",
        "",
    ]

    if not ok_df.empty:
        lines.append(
            "| Subject | Session | Duration (min) | Frames | Labeled markers (in-analysis / quarantined) | Preprocessing |"
        )
        lines.append("|---|---|---:|---:|---|---|")
        for _, row in ok_df.iterrows():
            lines.append(
                f"| {row['subject_id']} | {row['session_id']} | {row['duration_minutes']} | "
                f"{row['total_frames_observed']} | "
                f"{row.get('n_labeled_markers_in_analysis', '?')} / {row.get('n_quarantined_markers', 0)} | "
                f"{row['raw_qc_preprocessing_status']} |"
            )
    else:
        lines.append("*No successful sessions.*")

    lines.extend(["", "---", "", "## 3. Missingness and gaps", ""])
    if not ok_df.empty:
        lines.append("| Session | Missing % (labeled) | Gaps ≤0.2s | Gaps ≥0.2s | Gaps ≥0.5s | Longest gap (s) |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for _, row in ok_df.iterrows():
            lines.append(
                f"| {_session_label(row)} | {row['missing_percent_labeled']} | "
                f"{row['n_gaps_le_0p2s']} | {row['n_gaps_ge_0p2s']} | {row['n_gaps_ge_0p5s']} | "
                f"{row['longest_gap_seconds_labeled']} |"
            )

    lines.extend(["", "---", "", "## 4. Artifact profiling", ""])
    if not ok_df.empty:
        lines.append("| Session | Events | Single-frame | Short burst | Sustained | Near-gap % |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for _, row in ok_df.iterrows():
            lines.append(
                f"| {_session_label(row)} | {row['n_artifact_events']} | "
                f"{row['n_single_frame_events']} | {row['n_short_burst_events']} | "
                f"{row['n_sustained_events']} | {row['pct_artifact_events_near_gap']} |"
            )

    lines.extend(["", "---", "", "## 5. Analysis readiness (marker coverage & remediation)", ""])
    if not ok_df.empty:
        lines.append(
            "| Session | Coverage mean (%) | Time below coverage (%) | Sustained dropouts | Usable after remediation (%) | Recommended remediation |"
        )
        lines.append("|---|---:|---:|---:|---:|---|")
        for _, row in ok_df.iterrows():
            usable = float(row.get("usable_after_remediation_pct", 0.0) or 0.0)
            flag = " ⚠" if usable < 70 else ""
            lines.append(
                f"| {_session_label(row)}{flag} | {row.get('labeled_marker_coverage_mean_pct', '?')} | "
                f"{row.get('pct_time_below_coverage', '?')} | {row.get('n_markers_sustained_dropout', '?')} | "
                f"{row.get('usable_after_remediation_pct', '?')} | {row.get('recommended_remediation', '')} |"
            )
        lines.append("")
        lines.append(
            "*Coverage = mean fraction of in-analysis labeled markers present per frame. "
            "Usable-after-remediation excludes only >=0.5s gaps and marker-swap frames. "
            "Sessions below 70% usable are marked ⚠.*"
        )

    lines.extend(["", "---", "", "## 6. Kinematic profile (raw, unsmoothed)", ""])
    if not velocity_df.empty:
        pivot = velocity_df.pivot_table(
            index="body_region_group", columns=["subject_id", "session_id"], values="mean_speed_m_s"
        )
        lines.append("Mean gap-safe marker speed (m/s) by body segment — see `details/velocity_by_body_segment.csv`.")
        lines.append("")

    lines.extend(["", "---", "", "## 7. Recurring problem markers", ""])
    if not top_markers.empty:
        recur = top_markers.groupby("marker_name").size().sort_values(ascending=False).head(10)
        for marker, count in recur.items():
            lines.append(f"- **{marker}** appears in top-3 worst markers in {count} session(s)")
    else:
        lines.append("*No marker ranking available.*")

    lines.extend(["", "---", "", "## 8. Per-session reports", ""])
    for _, row in ok_df.iterrows():
        run_dir = row.get("run_output_dir", "")
        lines.append(f"- **{_session_label(row)}:** `{run_dir}/qc_report.md`")

    lines.extend(["", "---", "", "## 9. PI comparison plots", ""])
    for name in (
        "batch_preprocessing_status.png",
        "batch_usable_after_remediation.png",
        "batch_artifact_events.png",
        "batch_missingness.png",
    ):
        lines.append(f"![{name}](plots/{name})")
        lines.append("")

    if not failures_df.empty:
        lines.extend(["---", "", "## 10. Failures", ""])
        for _, row in failures_df.iterrows():
            lines.append(
                f"- **{row.get('subject_id')}_{row.get('session_id')}** ({row.get('file_name')}): "
                f"{row.get('error_message')}"
            )

    lines.extend(
        [
            "",
            "---",
            "",
            "## Methods note",
            "",
            "Velocity is frame-to-frame 3D Euclidean speed on gap-safe segments (no Savitzky–Golay). "
            "Artifact events cluster consecutive candidate frames; single-frame events are common "
            "during expressive movement. Never-solved/phantom markers are quarantined (excluded from "
            "missingness). The preprocessing verdict is coverage-based; window yield is retained only "
            "as a legacy debug metric.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _sustained_dropout_seconds(config: dict[str, Any]) -> float:
    return float(config.get("readiness", {}).get("sustained_dropout_seconds", 2.0))


def _session_key(sr: Any) -> str:
    return f"{sr.session_row.get('subject_id', '')}_{sr.session_row.get('session_id', '')}"


def _collect_skeleton_selection(session_results: list[Any] | None) -> pd.DataFrame:
    """Sessions where competing labeled skeletons were resolved to one analysis set."""
    rows: list[dict[str, Any]] = []
    if not session_results:
        return pd.DataFrame(
            columns=[
                "session",
                "analysis_skeleton",
                "ignored_skeleton",
                "n_markers_quarantined",
                "winner_coverage_pct",
                "loser_coverage_pct",
            ]
        )
    for sr in session_results:
        if getattr(sr, "batch_status", "") != "ok" or sr.layer1 is None:
            continue
        md = sr.layer1.session.metadata if sr.layer1.session else {}
        events = md.get("skeleton_selection_events") or []
        key = _session_key(sr)
        if events:
            for event in events:
                rows.append(
                    {
                        "session": key,
                        "analysis_skeleton": event.get("chosen_skeleton"),
                        "ignored_skeleton": event.get("ignored_skeleton"),
                        "n_markers_quarantined": event.get("n_markers_quarantined"),
                        "winner_coverage_pct": event.get("winner_mean_coverage_pct"),
                        "loser_coverage_pct": event.get("loser_mean_coverage_pct"),
                    }
                )
        elif md.get("analysis_skeleton_prefix"):
            rows.append(
                {
                    "session": key,
                    "analysis_skeleton": md.get("analysis_skeleton_prefix"),
                    "ignored_skeleton": "",
                    "n_markers_quarantined": 0,
                    "winner_coverage_pct": "",
                    "loser_coverage_pct": "",
                }
            )
    return pd.DataFrame(rows)


def _collect_mask_interval_candidates(
    session_results: list[Any] | None,
    *,
    max_per_session: int = 40,
) -> pd.DataFrame:
    """Per-frame mask intervals (time windows to skip), primary remediation view."""
    rows: list[dict[str, Any]] = []
    if not session_results:
        return pd.DataFrame(
            columns=[
                "session",
                "start_s",
                "end_s",
                "duration_s",
                "status",
                "criterion",
                "affected_markers",
            ]
        )
    for sr in session_results:
        if getattr(sr, "batch_status", "") != "ok" or sr.layer5 is None:
            continue
        intervals = sr.layer5.tables.get("qc_mask_intervals", pd.DataFrame())
        if intervals.empty:
            intervals = sr.layer5.tables.get("qc_intervals", pd.DataFrame())
        if intervals.empty:
            continue
        key = _session_key(sr)
        work = intervals.copy()
        if "status" in work.columns:
            work["_prio"] = work["status"].map({"exclude": 0, "caution": 1}).fillna(2)
            work = work.sort_values(["_prio", "duration_s"], ascending=[True, False])
        elif "duration_s" in work.columns:
            work = work.sort_values("duration_s", ascending=False)
        for _, r in work.head(max_per_session).iterrows():
            rows.append(
                {
                    "session": key,
                    "start_s": round(float(r.get("start_s", 0) or 0), 3),
                    "end_s": round(float(r.get("end_s", 0) or 0), 3),
                    "duration_s": round(float(r.get("duration_s", 0) or 0), 3),
                    "status": r.get("status", ""),
                    "criterion": r.get("criterion", r.get("reason", "")),
                    "affected_markers": r.get("affected_markers", ""),
                }
            )
    return pd.DataFrame(rows)


def _collect_whole_marker_exclusions(
    session_results: list[Any] | None,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Markers to drop for the entire session (not per-frame masking)."""
    rows: list[dict[str, Any]] = []
    if not session_results:
        return pd.DataFrame(columns=["session", "marker_name", "reason", "detail"])
    sustained_sec = _sustained_dropout_seconds(config)
    for sr in session_results:
        if getattr(sr, "batch_status", "") != "ok" or sr.layer2 is None:
            continue
        key = _session_key(sr)
        quar = sr.layer2.tables.get("quarantined_markers", pd.DataFrame())
        if not quar.empty:
            for _, r in quar.iterrows():
                reason = str(r.get("quarantine_reason", ""))
                if reason == "phantom_skeleton":
                    continue
                rows.append(
                    {
                        "session": key,
                        "marker_name": r.get("marker_name"),
                        "reason": reason,
                        "detail": f"{r.get('missing_percent')}% missing",
                    }
                )
        mq = sr.layer2.tables.get("marker_quality_summary", pd.DataFrame())
        if not mq.empty and "included_in_analysis" in mq.columns:
            sustained = mq[
                mq["included_in_analysis"]
                & (mq.get("longest_gap_seconds", 0) >= sustained_sec)
            ]
            for _, r in sustained.iterrows():
                rows.append(
                    {
                        "session": key,
                        "marker_name": r.get("marker_name"),
                        "reason": "sustained_dropout",
                        "detail": f"longest gap {r.get('longest_gap_seconds')}s",
                    }
                )
    return pd.DataFrame(rows)


def _html_table(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df.empty:
        return "<p><em>None.</em></p>"
    df = df.head(max_rows)
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{'' if pd.isna(v) else v}</td>" for v in r) + "</tr>"
        for r in df.itertuples(index=False)
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def export_session_interval_csvs(
    session_results: list[Any] | None,
    details_dir: Path,
) -> dict[str, Path]:
    """Write per-session qc_mask_intervals CSV to batch details/."""
    paths: dict[str, Path] = {}
    if not session_results:
        return paths
    details_dir.mkdir(parents=True, exist_ok=True)
    for sr in session_results:
        if getattr(sr, "batch_status", "") != "ok" or sr.layer5 is None:
            continue
        key = _session_key(sr)
        intervals = sr.layer5.tables.get("qc_mask_intervals", pd.DataFrame())
        if intervals.empty:
            intervals = sr.layer5.tables.get("qc_intervals", pd.DataFrame())
        if intervals.empty:
            continue
        intervals = enrich_mask_intervals(intervals)
        out = details_dir / f"{key}_qc_mask_intervals.csv"
        intervals.to_csv(out, index=False)
        paths[key] = out
    return paths


def _interval_summary_stats(intervals: pd.DataFrame, qc_mask: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    art_sigma = velocity_mad_sigma(config)
    vel_pct = velocity_percentile_threshold(config)
    n_art_frames = 0
    if not qc_mask.empty and "flag_artifact_sigma" in qc_mask.columns:
        n_art_frames = int(qc_mask["flag_artifact_sigma"].astype(bool).sum())
    if intervals.empty:
        return {
            "n_exclude": 0,
            "n_caution": 0,
            "n_artifact_sigma_intervals": 0,
            "n_artifact_sigma_frames": n_art_frames,
            "art_sigma": art_sigma,
            "vel_percentile": vel_pct,
        }
    return {
        "n_exclude": int((intervals["status"] == "exclude").sum()),
        "n_caution": int((intervals["status"] == "caution").sum()),
        "n_artifact_sigma_intervals": int(intervals.get("has_artifact_sigma", pd.Series(dtype=bool)).sum())
        if "has_artifact_sigma" in intervals.columns
        else int(intervals["reason"].astype(str).str.contains("ARTIFACT_SIGMA", regex=False).sum()),
        "n_artifact_sigma_frames": n_art_frames,
        "art_sigma": art_sigma,
        "vel_percentile": vel_pct,
    }


def _build_session_tabs_html(
    session_results: list[Any] | None,
    batch_dir: Path,
    config: dict[str, Any],
) -> str:
    if not session_results:
        return "<p><em>No session data.</em></p>"

    tabs: list[str] = []
    panels: list[str] = []
    idx = 0
    for sr in session_results:
        if getattr(sr, "batch_status", "") != "ok":
            continue
        key = _session_key(sr)
        tab_id = f"tab-{idx}"
        active = " active" if idx == 0 else ""
        tabs.append(
            f'<button class="tab-btn{active}" onclick="openTab(event, \'{tab_id}\')">{key}</button>'
        )

        intervals = pd.DataFrame()
        qc_mask = pd.DataFrame()
        if sr.layer5:
            intervals = sr.layer5.tables.get("qc_mask_intervals", pd.DataFrame())
            if intervals.empty:
                intervals = sr.layer5.tables.get("qc_intervals", pd.DataFrame())
            qc_mask = sr.layer5.tables.get("qc_mask", pd.DataFrame())
        stats = _interval_summary_stats(intervals, qc_mask, config)

        csv_rel = f"details/{key}_qc_mask_intervals.csv"
        csv_exists = (batch_dir / csv_rel).exists()

        skel_note = ""
        if sr.layer1 and sr.layer1.session:
            md = sr.layer1.session.metadata
            events = md.get("skeleton_selection_events") or []
            if events:
                ev = events[0]
                skel_note = (
                    f"<p class=\"sub\">Skeleton: using <code>{ev.get('chosen_skeleton')}</code>, "
                    f"ignored phantom <code>{ev.get('ignored_skeleton')}</code> "
                    f"({ev.get('n_markers_quarantined')} markers quarantined).</p>"
                )

        preview = ""
        if not intervals.empty:
            excl = intervals[intervals["status"] == "exclude"].sort_values(
                "duration_s", ascending=False
            ).head(10)
            if not excl.empty:
                preview = "<h4>Top exclude intervals (preview)</h4>" + _html_table(
                    excl[["start_s", "end_s", "duration_s", "criterion", "affected_markers"]]
                )

        hist_path = f"plots/sessions/{key}_artifact_histograms.png"
        hist_img = ""
        if (batch_dir / hist_path).exists():
            hist_img = f'<figure><img src="{hist_path}" alt="artifact histograms"/><figcaption>Top-3 segment artifact velocity histograms</figcaption></figure>'

        art_heat_path = f"plots/sessions/{key}_artifact_heatmap.png"
        art_heat_img = ""
        if (batch_dir / art_heat_path).exists():
            art_caption = artifact_candidate_severity_note(config)
            art_heat_img = (
                f'<figure><img src="{art_heat_path}" alt="artifact heatmap"/>'
                f"<figcaption><strong>Artifact candidate heatmap</strong> — x-axis is time (seconds); "
                f"y-axis is in-analysis markers. {art_caption}</figcaption></figure>"
            )

        csv_link = (
            f'<p><a href="{csv_rel}"><strong>Download full interval table</strong> ({csv_rel})</a></p>'
            if csv_exists
            else "<p><em>Interval CSV not available.</em></p>"
        )

        panel = f"""<div id="{tab_id}" class="tab-panel{active}">
  <h3>{key}</h3>
  {skel_note}
  <ul>
    <li><strong>{stats['n_exclude']}</strong> exclude intervals · <strong>{stats['n_caution']}</strong> caution intervals</li>
    <li><strong>{stats['n_artifact_sigma_frames']:,}</strong> frames flagged by velocity MAD
     (σ={stats['art_sigma']}, percentile={stats['vel_percentile']})</li>
    <li><strong>{stats['n_artifact_sigma_intervals']}</strong> artifact-sigma interval rows in CSV</li>
  </ul>
  {csv_link}
  {preview}
  {art_heat_img}
  {hist_img}
</div>"""
        panels.append(panel)
        idx += 1

    if not tabs:
        return "<p><em>No successful sessions.</em></p>"

    grid_img = ""
    if (batch_dir / "plots/batch_missing_heatmaps_6sessions.png").exists():
        grid_img = (
            '<figure style="margin-bottom:20px"><img src="plots/batch_missing_heatmaps_6sessions.png" '
            'alt="batch gap heatmaps"/><figcaption>All sessions — gap severity heatmap (white=present, '
            'yellow=gap 0.2–0.5s, red=gap ≥0.5s)</figcaption></figure>'
        )

    return f"""{grid_img}
<div class="tab-bar">{"".join(tabs)}</div>
<div class="tab-content">{"".join(panels)}</div>
<script>
function openTab(evt, tabId) {{
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(tabId).classList.add('active');
  evt.currentTarget.classList.add('active');
}}
</script>"""


def build_quality_report_html(
    eda_df: pd.DataFrame,
    failures_df: pd.DataFrame,
    session_results: list[Any] | None,
    batch_dir: Path,
    config: dict[str, Any],
) -> str:
    """Single shareable, publication-grade data-quality report for a PI/professor."""
    ok = eda_df[eda_df["batch_status"] == "ok"].copy() if not eda_df.empty else eda_df
    subjects = ", ".join(sorted(eda_df["subject_id"].astype(str).unique())) if not eda_df.empty else "n/a"

    fps_vals = ok["effective_frame_rate_hz"].dropna().unique() if not ok.empty else []
    if len(fps_vals) == 1:
        fps_note = f"{float(fps_vals[0]):.0f} fps"
    elif len(fps_vals) > 1:
        fps_note = ", ".join(f"{float(v):.0f}" for v in sorted(fps_vals)) + " fps"
    else:
        fps_note = "n/a"

    verdict_rows = ""
    if not ok.empty:
        for _, r in ok.iterrows():
            gaps_02_05 = r.get("n_gaps_0p2_to_0p5s", "")
            pct_gap_02_05 = r.get("pct_gap_time_0p2_to_0p5", "")
            gaps_cell = f"{gaps_02_05} ({pct_gap_02_05}% time)" if pct_gap_02_05 != "" else gaps_02_05
            usable = r.get("pct_frames_above_coverage", r.get("usable_after_remediation_pct", ""))
            verdict_rows += (
                "<tr>"
                f"<td>{r.get('subject_id')}_{r.get('session_id')}</td>"
                f"<td>{r.get('n_labeled_markers_in_analysis', '')}</td>"
                f"<td>{r.get('total_frames_observed', '')}</td>"
                f"<td>{r.get('duration_minutes', '')}</td>"
                f"<td>{r.get('missing_percent_labeled')}</td>"
                f"<td>{r.get('labeled_marker_coverage_mean_pct')}</td>"
                f"<td>{gaps_cell}</td>"
                f"<td>{r.get('n_markers_gaps_ge_0p5s', '')}</td>"
                f"<td>{r.get('n_markers_sustained_dropout', '')}</td>"
                f"<td>{r.get('artifact_burden_summary', r.get('pct_candidate_frames', ''))}</td>"
                f"<td>{usable}</td>"
                f"<td>{r.get('recommended_remediation', '')}</td>"
                "</tr>"
            )

    whole_markers = _collect_whole_marker_exclusions(session_results, config)
    session_tabs = _build_session_tabs_html(session_results, batch_dir, config)

    art_sigma = velocity_mad_sigma(config)
    vel_pct = velocity_percentile_threshold(config)
    art_severity_note = artifact_candidate_severity_note(config)

    fail_html = ""
    if not failures_df.empty:
        fail_html = "<h2>Failures</h2>" + _html_table(failures_df)

    min_cov = config.get("readiness", {}).get("min_marker_coverage_pct", 90)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Motive QC — Dataset Quality Report</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
   margin: 0 auto; max-width: 1100px; padding: 32px; color: #1a202c; line-height: 1.5; }}
 h1 {{ font-size: 24px; margin-bottom: 4px; }}
 h2 {{ margin-top: 36px; border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; }}
 h3 {{ font-size: 16px; margin-top: 0; }}
 .sub {{ color: #718096; font-size: 13px; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 12px; margin-top: 10px; }}
 th, td {{ border: 1px solid #e2e8f0; padding: 6px 8px; text-align: left; vertical-align: top; }}
 th {{ background: #f7fafc; }}
 .badge {{ color: #fff; padding: 2px 8px; border-radius: 10px; font-size: 12px; }}
 figure {{ margin: 16px 0; }} img {{ max-width: 100%; border: 1px solid #e2e8f0; border-radius: 6px; }}
 figcaption {{ color: #718096; font-size: 12px; }}
 .note {{ background: #f0fff4; border-left: 4px solid #48bb78; padding: 10px 14px; font-size: 13px; }}
 .tab-bar {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 16px 0 8px; }}
 .tab-btn {{ background: #edf2f7; border: 1px solid #cbd5e0; padding: 8px 14px; cursor: pointer;
   border-radius: 6px; font-size: 13px; }}
 .tab-btn.active {{ background: #2b6cb0; color: #fff; border-color: #2b6cb0; }}
 .tab-panel {{ display: none; padding: 12px 0; }}
 .tab-panel.active {{ display: block; }}
</style></head><body>
<h1>Motive QC — Dataset Quality Report</h1>
<p class="sub">Generated {datetime.now().isoformat(timespec='seconds')} · Motive_QC v{__version__} ·
 Subject(s) {subjects} · {len(ok)} session(s) · {fps_note}</p>
<div class="note"><strong>Reading this report.</strong> QC masking is <em>per frame</em>. Skip time windows
 listed in session tabs (CSV links) — do not delete whole markers unless section 3 says so.
 <strong>Usable frame %</strong> = share of frames where ≥{min_cov}% of in-analysis markers are present
 (one bad marker does not collapse the whole session). Phantom skeletons are auto-quarantined.</div>

<h2>1. At-a-glance verdict</h2>
<table><thead><tr>
 <th>Session</th><th>Markers</th><th>Frames</th><th>Duration (min)</th>
 <th>Missing %</th><th>Coverage mean %</th><th>Gaps 0.2–0.5s</th><th>Markers w/ gap≥0.5s</th>
 <th>Sustained dropouts</th><th>Artifact burden (MAD σ)</th><th>Usable frame %</th><th>Recommended remediation</th>
</tr></thead><tbody>{verdict_rows}</tbody></table>

<h2>2. Session details</h2>
<p class="sub">Per-session mask interval summaries, CSV downloads, and plots. Full interval tables
 (including velocity MAD artifact flags) are in <code>details/&lt;session&gt;_qc_mask_intervals.csv</code>.</p>
{session_tabs}

<h2>3. Whole-marker exclusions</h2>
<p class="sub">Markers to drop for the <em>entire</em> session — never-solved (non-phantom) or sustained
 dropout (single gap ≥ {_sustained_dropout_seconds(config)}&nbsp;s). Example:
 <code>T3_671:LThighFront</code> in 671_T3_P1_R2.</p>
{_html_table(whole_markers)}

{fail_html}

<h2>Methods note</h2>
<p class="sub">Speed = frame-to-frame 3D Euclidean displacement / dt on gap-safe segments (no smoothing).
 Artifact velocity flags use MAD σ={art_sigma:g} and velocity percentile {vel_pct:g} from
 <code>config.yaml</code> (<code>artifacts.velocity_mad_multiplier</code>). Histogram red bars =
 flagged speeds; dashed line = MAD threshold at σ={art_sigma:g}. Gap heatmap: yellow = gap 0.2–0.5&nbsp;s,
 red = gap ≥0.5&nbsp;s on in-analysis labeled markers only.</p>
<p class="sub"><strong>Artifact heatmap severity.</strong> {art_severity_note}</p>
</body></html>
"""


def write_dataset_eda_excel(
    path: Path,
    eda_df: pd.DataFrame,
    top_markers: pd.DataFrame,
    artifact_types: pd.DataFrame,
    velocity_df: pd.DataFrame,
    failures_df: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    config_rows = [{"key": k, "value": str(v)} for k, v in _flatten_config(config).items()]
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        eda_df.to_excel(writer, sheet_name="Executive", index=False)
        top_markers.to_excel(writer, sheet_name="TopMarkers", index=False)
        artifact_types.to_excel(writer, sheet_name="ArtifactTypes", index=False)
        velocity_df.to_excel(writer, sheet_name="VelocityBySegment", index=False)
        failures_df.to_excel(writer, sheet_name="Failures", index=False)
        pd.DataFrame(config_rows).to_excel(writer, sheet_name="Config", index=False)


def _flatten_config(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in config.items():
        if key.startswith("_"):
            continue
        full = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flatten_config(value, full))
        else:
            out[full] = value
    return out


def write_pi_plots(eda_df: pd.DataFrame, plots_dir: Path, dpi: int = 150) -> list[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    ok = eda_df[eda_df["batch_status"] == "ok"].copy() if not eda_df.empty else eda_df
    if ok.empty:
        return []
    ok["label"] = ok.apply(_session_label, axis=1)
    written: list[Path] = []

    # Preprocessing status
    fig, ax = plt.subplots(figsize=(max(8, len(ok) * 0.5), 5))
    colors = {"acceptable": "#48bb78", "caution": "#ecc94b", "poor": "#e53e3e"}
    statuses = ok["raw_qc_preprocessing_status"].fillna("unknown")
    bar_colors = [colors.get(s, "#a0aec0") for s in statuses]
    ax.bar(ok["label"], [1] * len(ok), color=bar_colors)
    ax.set_yticks([])
    ax.set_title("Preprocessing QC status by session (green=acceptable, yellow=caution, red=poor)")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    p = plots_dir / "batch_preprocessing_status.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # Usable data before vs after remediation
    if "usable_after_remediation_pct" in ok.columns:
        fig, ax = plt.subplots(figsize=(max(8, len(ok) * 0.5), 5))
        x = np.arange(len(ok))
        w = 0.38
        ax.bar(x - w / 2, ok["pct_frames_use"], w, label="usable now", color="#2b6cb0")
        ax.bar(
            x + w / 2,
            ok["usable_after_remediation_pct"],
            w,
            label="usable after remediation",
            color="#48bb78",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(ok["label"], rotation=45, ha="right")
        ax.set_ylabel("% of frames")
        ax.set_title("Usable data: now vs. after interpolation/filtering")
        ax.axhline(70, color="#e53e3e", linestyle="--", linewidth=0.8)
        ax.legend()
        fig.tight_layout()
        p = plots_dir / "batch_usable_after_remediation.png"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    # Artifact events stacked
    fig, ax = plt.subplots(figsize=(max(8, len(ok) * 0.5), 5))
    bottom = np.zeros(len(ok))
    for col, color in [
        ("n_single_frame_events", "#ecc94b"),
        ("n_short_burst_events", "#ed8936"),
        ("n_sustained_events", "#e53e3e"),
    ]:
        vals = ok[col].fillna(0).astype(float).values
        ax.bar(ok["label"], vals, bottom=bottom, label=col.replace("n_", ""), color=color)
        bottom += vals
    ax.set_title("Artifact events by class")
    ax.legend(fontsize=8)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    p = plots_dir / "batch_artifact_events.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # Missingness
    fig, ax = plt.subplots(figsize=(max(8, len(ok) * 0.5), 5))
    ax.bar(ok["label"], ok["missing_percent_labeled"], color="#4a5568")
    ax.set_ylabel("Missing % (labeled)")
    ax.set_title("Labeled marker missingness by session")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    p = plots_dir / "batch_missingness.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    return written


def write_batch_reports(
    batch_dir: Path,
    eda_df: pd.DataFrame,
    top_markers: pd.DataFrame,
    artifact_types: pd.DataFrame,
    velocity_df: pd.DataFrame,
    failures_df: pd.DataFrame,
    config: dict[str, Any],
    session_pointers: list[dict[str, Any]],
    session_results: list[Any] | None = None,
) -> dict[str, Path]:
    """Write all Layer 6 deliverables; return paths dict."""
    batch_dir.mkdir(parents=True, exist_ok=True)
    details_dir = batch_dir / "details"
    details_dir.mkdir(exist_ok=True)
    sessions_dir = batch_dir / "sessions"
    sessions_dir.mkdir(exist_ok=True)
    plots_dir = batch_dir / "plots"

    paths: dict[str, Path] = {}

    csv_path = batch_dir / "dataset_eda_report.csv"
    eda_df.to_csv(csv_path, index=False)
    paths["csv"] = csv_path

    md_path = batch_dir / "dataset_eda_report.md"
    md_path.write_text(
        build_dataset_eda_markdown(
            eda_df, top_markers, artifact_types, velocity_df, failures_df, batch_dir, config
        ),
        encoding="utf-8",
    )
    paths["md"] = md_path

    # Interval CSVs and plots before HTML (report embeds them).
    interval_paths = export_session_interval_csvs(session_results, details_dir)
    plot_paths = write_batch_session_plots(session_results or [], plots_dir, config)

    batch_out_cfg = config.get("outputs", {}).get("batch", {})

    # Canonical human deliverable: a single shareable HTML quality report.
    html_path = batch_dir / "dataset_quality_report.html"
    html_path.write_text(
        build_quality_report_html(eda_df, failures_df, session_results, batch_dir, config),
        encoding="utf-8",
    )
    paths["html"] = html_path

    # Opt-in heavy/redundant outputs (off by default to avoid over-reporting).
    if batch_out_cfg.get("write_eda_excel", False):
        xlsx_path = batch_dir / "dataset_eda_report.xlsx"
        write_dataset_eda_excel(
            xlsx_path, eda_df, top_markers, artifact_types, velocity_df, failures_df, config
        )
        paths["xlsx"] = xlsx_path

    sigma_df = build_artifact_sigma_sensitivity(session_results, config) if session_results else pd.DataFrame()
    if not sigma_df.empty:
        sigma_df.to_csv(details_dir / "artifact_sigma_sensitivity.csv", index=False)
    if batch_out_cfg.get("write_workbook", False) and session_results:
        workbook_path = batch_dir / "dataset_eda_workbook.xlsx"
        write_pi_workbook(workbook_path, session_results, config, sigma_df=sigma_df)
        paths["workbook"] = workbook_path

    top_markers.to_csv(details_dir / "top_markers_by_session.csv", index=False)
    artifact_types.to_csv(details_dir / "artifact_type_distribution.csv", index=False)
    velocity_df.to_csv(details_dir / "velocity_by_body_segment.csv", index=False)

    if not failures_df.empty:
        fail_path = batch_dir / "failures.csv"
        failures_df.to_csv(fail_path, index=False)
        paths["failures"] = fail_path

    config_snap = batch_dir / "config_snapshot.yaml"
    clean = {k: v for k, v in config.items() if not str(k).startswith("_")}
    with config_snap.open("w", encoding="utf-8") as f:
        yaml.safe_dump(clean, f, sort_keys=False)
    paths["config"] = config_snap

    for ptr in session_pointers:
        name = f"{ptr['subject_id']}_{ptr['session_id']}.json"
        (sessions_dir / name).write_text(json.dumps(ptr, indent=2), encoding="utf-8")

    paths["plots"] = plots_dir

    manifest_files = [
        p.name for k, p in paths.items() if k in {"csv", "md", "html", "xlsx", "workbook"}
    ]
    manifest = {
        "motive_qc_version": __version__,
        "batch_time": datetime.now().isoformat(timespec="seconds"),
        "batch_dir": str(batch_dir.resolve()),
        "n_sessions": len(eda_df),
        "n_failures": len(failures_df),
        "files": manifest_files,
        "plot_files": [p.name for p in plot_paths],
        "interval_csv_files": [p.name for p in interval_paths.values()],
    }
    manifest_path = batch_dir / "BATCH_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    paths["manifest"] = manifest_path

    return paths
