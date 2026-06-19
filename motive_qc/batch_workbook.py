"""PI-facing structured Excel workbook for Layer 6 batch runs."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pandas as pd

from motive_qc.artifacts import run_layer4_artifacts
from motive_qc.batch_metrics import (
    _candidate_frame_stats,
    _gap_counts,
    extract_artifact_intervals,
    extract_gap_windows,
    extract_segment_burden,
)

DEFAULT_SIGMA_SENSITIVITY = (4.0, 8.0, 12.0)
SIGMA_LABELS = {4.0: "stricter", 8.0: "Gaga default", 12.0: "looser"}


def _sigma_sensitivity_config(config: dict[str, Any]) -> dict[str, Any]:
    defaults = {"enabled": True, "sigmas": list(DEFAULT_SIGMA_SENSITIVITY), "chosen_sigma": 8.0}
    return {**defaults, **config.get("artifacts", {}).get("sigma_sensitivity", {})}


def _session_key(row: dict[str, Any]) -> str:
    return f"{row.get('subject_id', '')}_{row.get('session_id', '')}"


def _artifact_methods_text(config: dict[str, Any]) -> list[str]:
    art = config.get("artifacts", {})
    methods = art.get("methods", {})
    spike = art.get("single_frame_spike", {})
    return [
        "Artifact screening methods (raw QC, no smoothing):",
        "  Speed = ||p[t+1] - p[t]|| / dt on gap-safe segments only.",
        "  Threshold per marker = max(median + sigma*MAD, percentile); flag local speed peaks.",
        f"  velocity_mad_multiplier (sigma) = {art.get('velocity_mad_multiplier')}",
        f"  velocity_percentile_threshold = {art.get('velocity_percentile_threshold')}",
        f"  spike min_jump_distance_m = {spike.get('min_jump_distance_m')}",
        f"  acceleration_mad enabled = {bool(methods.get('acceleration_mad', False))}",
        f"  constant_position_hold enabled = {bool(methods.get('constant_position_hold', False))}",
        "Detections are candidates requiring visual review.",
    ]


def build_executive_summary_sections(
    session_results: list[Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build DataFrames and methods text for the ExecutiveSummary sheet."""
    inventory_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    segment_frames: list[pd.DataFrame] = []

    for sr in session_results:
        row = sr.session_row
        session = _session_key(row)
        if sr.batch_status != "ok" or sr.layer1 is None or sr.layer2 is None:
            inventory_rows.append(
                {
                    "session": session,
                    "file_name": row.get("file_name", ""),
                    "batch_status": sr.batch_status,
                    "error_message": sr.error_message,
                }
            )
            continue

        md = sr.layer1.session.metadata if sr.layer1.session else {}
        summ = sr.layer2.tables.get("session_summary", pd.DataFrame())
        s = summ.iloc[0].to_dict() if not summ.empty else {}
        gap_events = sr.layer2.tables.get("gap_events", pd.DataFrame())
        gap_ct = _gap_counts(gap_events)
        dur_s = float(md.get("duration_seconds", 0) or 0)
        total_frames = int(md.get("total_frames_observed", 0))
        union_gap = float(s.get("union_gap_seconds_ge_0p2_labeled") or 0.0)
        pct_gap_time = round(100.0 * union_gap / dur_s, 4) if dur_s > 0 else 0.0
        cand = _candidate_frame_stats(sr.layer4, total_frames)

        art_summary = (
            sr.layer4.tables.get("artifact_session_summary", pd.DataFrame()).iloc[0].to_dict()
            if sr.layer4
            and not sr.layer4.tables.get("artifact_session_summary", pd.DataFrame()).empty
            else {}
        )

        inventory_rows.append(
            {
                "session": session,
                "file_name": row.get("file_name", md.get("file_name", "")),
                "total_frames": total_frames,
                "frame_rate_hz": md.get("effective_frame_rate_hz"),
                "duration_seconds": round(dur_s, 3),
                "preprocessing_status": s.get("raw_qc_preprocessing_status"),
                "status_reason": s.get("raw_qc_status_reason"),
            }
        )
        gap_rows.append(
            {
                "session": session,
                "missing_percent_labeled": s.get("missing_percent_labeled"),
                "n_gaps_total": gap_ct["n_gaps_total_labeled"],
                "n_gaps_le_0p1s": gap_ct["n_gaps_le_0p1s"],
                "n_gaps_0p2_to_0p5s": gap_ct["n_gaps_0p2_to_0p5s"],
                "n_gaps_0p5_to_1s": gap_ct["n_gaps_0p5_to_1s"],
                "n_gaps_gt_1s": gap_ct["n_gaps_gt_1s"],
                "n_gaps_ge_0p2s": gap_ct["n_gaps_ge_0p2s"],
                "n_gaps_ge_0p5s": gap_ct["n_gaps_ge_0p5s"],
                "n_gaps_ge_1p0s": gap_ct["n_gaps_ge_1p0s"],
                "union_gap_seconds_ge_0p2": union_gap,
                "pct_session_gap_time_ge_0p2": pct_gap_time,
                "longest_gap_seconds": s.get("longest_gap_seconds_labeled"),
            }
        )
        artifact_rows.append(
            {
                "session": session,
                "n_candidate_frames": cand["n_candidate_frames"],
                "pct_candidate_frames": cand["pct_candidate_frames"],
                "n_artifact_events": art_summary.get("n_events", 0),
                "n_single_frame_events": art_summary.get("n_single_frame_events", 0),
                "n_short_burst_events": art_summary.get("n_short_burst_events", 0),
                "n_sustained_events": art_summary.get("n_sustained_events", 0),
            }
        )
        seg = extract_segment_burden(row, sr.layer2, sr.layer4, config)
        if not seg.empty:
            segment_frames.append(seg)

    segments_df = pd.concat(segment_frames, ignore_index=True) if segment_frames else pd.DataFrame()
    return {
        "methods_text": _artifact_methods_text(config),
        "inventory": pd.DataFrame(inventory_rows),
        "gaps": pd.DataFrame(gap_rows),
        "artifacts": pd.DataFrame(artifact_rows),
        "segments": segments_df,
    }


def build_artifact_sigma_sensitivity(
    session_results: list[Any],
    config: dict[str, Any],
    sigmas: tuple[float, ...] | None = None,
) -> pd.DataFrame:
    """Re-run L4 at multiple velocity MAD multipliers (L2 evidence held fixed).

    Gated by ``artifacts.sigma_sensitivity.enabled``; sigmas default to 4/8/12.
    Heavy (re-runs L4 per session per sigma) so this is a batch-workbook-only,
    opt-in tuning aid, not a routine per-session output.
    """
    ss_cfg = _sigma_sensitivity_config(config)
    if not ss_cfg.get("enabled", True):
        return pd.DataFrame()
    if sigmas is None:
        sigmas = tuple(float(s) for s in ss_cfg.get("sigmas", DEFAULT_SIGMA_SENSITIVITY))
    art_cfg = config.get("artifacts", {})
    vel_pct = art_cfg.get("velocity_percentile_threshold")
    rows: list[dict[str, Any]] = []

    for sr in session_results:
        if sr.batch_status != "ok" or sr.layer1 is None or sr.layer2 is None:
            continue
        session = sr.layer1.session
        if session is None:
            continue
        session_key = _session_key(sr.session_row)
        total_frames = int(session.metadata.get("total_frames_observed", 0))

        for sigma in sigmas:
            cfg = copy.deepcopy(config)
            cfg.setdefault("artifacts", {})["velocity_mad_multiplier"] = float(sigma)
            l4 = run_layer4_artifacts(session, sr.layer2, cfg, verbose=False)
            summ_df = l4.tables.get("artifact_session_summary", pd.DataFrame())
            if summ_df.empty:
                s: dict[str, Any] = {}
            else:
                s = summ_df.iloc[0].to_dict()
            cand = _candidate_frame_stats(l4, total_frames)
            n_cand = cand["n_candidate_frames"]
            if n_cand == 0 and not summ_df.empty:
                n_cand = int(summ_df.iloc[0].get("n_frame_candidates", 0))
                pct = round(100.0 * n_cand / total_frames, 4) if total_frames else 0.0
            else:
                pct = cand["pct_candidate_frames"]
            rows.append(
                {
                    "session": session_key,
                    "velocity_mad_sigma": sigma,
                    "sigma_label": SIGMA_LABELS.get(sigma, ""),
                    "velocity_percentile_threshold": vel_pct,
                    "n_artifact_events": int(s.get("n_events", 0)),
                    "n_candidate_frames": n_cand,
                    "pct_candidate_frames": pct,
                    "n_single_frame_events": int(s.get("n_single_frame_events", 0)),
                    "n_short_burst_events": int(s.get("n_short_burst_events", 0)),
                    "n_sustained_events": int(s.get("n_sustained_events", 0)),
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["session", "velocity_mad_sigma"]).reset_index(drop=True)
    return df


def _write_section_header(worksheet, row: int, title: str, fmt) -> int:
    worksheet.write(row, 0, title, fmt)
    return row + 2


def _write_dataframe(worksheet, row: int, df: pd.DataFrame, header_fmt, cell_fmt) -> int:
    if df.empty:
        worksheet.write(row, 0, "(no data)", cell_fmt)
        return row + 2
    for col_idx, col_name in enumerate(df.columns):
        worksheet.write(row, col_idx, col_name, header_fmt)
    row += 1
    for _, record in df.iterrows():
        for col_idx, col_name in enumerate(df.columns):
            val = record[col_name]
            if pd.isna(val):
                val = ""
            worksheet.write(row, col_idx, val, cell_fmt)
        row += 1
    return row + 2


def _excel_sheet_name(session_key: str) -> str:
    name = session_key.replace("/", "_")[:31]
    return name or "session"


def write_pi_workbook(
    path: Path,
    session_results: list[Any],
    config: dict[str, Any],
    sigma_df: pd.DataFrame | None = None,
) -> Path:
    """Write dataset_eda_workbook.xlsx with ExecutiveSummary + per-session tabs."""
    path = Path(path)
    sections = build_executive_summary_sections(session_results, config)
    if sigma_df is None:
        sigma_df = build_artifact_sigma_sensitivity(session_results, config)

    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        workbook = writer.book
        title_fmt = workbook.add_format({"bold": True, "font_size": 12})
        section_fmt = workbook.add_format({"bold": True, "font_size": 11, "bg_color": "#E2E8F0"})
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#EDF2F7"})
        cell_fmt = workbook.add_format({"text_wrap": False})
        wrap_fmt = workbook.add_format({"text_wrap": True})

        ws = workbook.add_worksheet("ExecutiveSummary")
        writer.sheets["ExecutiveSummary"] = ws
        row = 0
        ws.write(row, 0, "Layer 6 PI Executive Summary", title_fmt)
        row += 2

        row = _write_section_header(ws, row, "A. Session inventory", section_fmt)
        row = _write_dataframe(ws, row, sections["inventory"], header_fmt, cell_fmt)

        row = _write_section_header(ws, row, "B. Gaps and missingness", section_fmt)
        row = _write_dataframe(ws, row, sections["gaps"], header_fmt, cell_fmt)

        row = _write_section_header(ws, row, "C. Artifact screening", section_fmt)
        for line in sections["methods_text"]:
            ws.write(row, 0, line, wrap_fmt)
            row += 1
        row += 1
        row = _write_dataframe(ws, row, sections["artifacts"], header_fmt, cell_fmt)

        row = _write_section_header(ws, row, "D. Major segments (gaps and/or artifacts)", section_fmt)
        _write_dataframe(ws, row, sections["segments"], header_fmt, cell_fmt)

        sws_sigma = workbook.add_worksheet("ArtifactSigmaSensitivity")
        writer.sheets["ArtifactSigmaSensitivity"] = sws_sigma
        r = 0
        sws_sigma.write(
            r,
            0,
            "Velocity MAD sigma sensitivity (L4 re-run per session; L2 gaps fixed; fingers excluded)",
            title_fmt,
        )
        r += 2
        sws_sigma.write(
            r,
            0,
            "Higher sigma = looser gate (fewer artifact events). Percentile threshold unchanged per row.",
            wrap_fmt,
        )
        r += 2
        _write_dataframe(sws_sigma, r, sigma_df, header_fmt, cell_fmt)

        for sr in session_results:
            row_dict = sr.session_row
            sheet_name = _excel_sheet_name(_session_key(row_dict))
            sws = workbook.add_worksheet(sheet_name)
            writer.sheets[sheet_name] = sws

            if sr.batch_status != "ok":
                sws.write(0, 0, f"Session failed: {sr.error_message}", wrap_fmt)
                continue

            r = 0
            sws.write(r, 0, f"Session: {_session_key(row_dict)}", title_fmt)
            r += 2

            r = _write_section_header(sws, r, "Gap windows (0.5 s, gap >= 0.2 s)", section_fmt)
            gap_w = extract_gap_windows(sr.layer3, min_gap_s=0.2)
            r = _write_dataframe(sws, r, gap_w, header_fmt, cell_fmt)

            r = _write_section_header(sws, r, "Artifact events", section_fmt)
            _write_dataframe(sws, r, extract_artifact_intervals(sr.layer4), header_fmt, cell_fmt)

    return path
