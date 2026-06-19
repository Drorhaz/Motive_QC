"""Layer 4: artifact candidate screening with gap-safe kinematics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from motive_qc.analysis_scope import (
    analysis_labeled_marker_names,
    body_group_excluded,
)
from motive_qc.core import LOGGER, MotiveSession, QCMessage, QCResult
from motive_qc.plots import (
    plot_artifact_events_timeline,
    plot_artifact_timeline,
    plot_velocity_artifact_histogram,
)
from motive_qc.segments import (
    compute_accelerations,
    compute_speeds,
    frames_near_gap,
    marker_valid_segments,
    robust_mad_threshold,
    segment_positions,
)


def _artifacts_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("artifacts", {})


def velocity_mad_sigma(config: dict[str, Any]) -> float:
    """MAD multiplier (σ) used for Layer 4 velocity artifact detection."""
    return float(_artifacts_config(config).get("velocity_mad_multiplier", 11.0))


def velocity_percentile_threshold(config: dict[str, Any]) -> float:
    """Percentile floor paired with MAD in Layer 4 velocity artifact detection."""
    return float(_artifacts_config(config).get("velocity_percentile_threshold", 99.97))


def artifact_candidate_severity_note(config: dict[str, Any]) -> str:
    """Human-readable explanation of artifact heatmap severity colors."""
    art_cfg = _artifacts_config(config)
    sigma = velocity_mad_sigma(config)
    seg_pct = float(art_cfg.get("rigid_body", {}).get("max_segment_length_change_pct", 18.0))
    return (
        f"Each cell is the strongest artifact candidate on that marker at that time. "
        f"Velocity/acceleration MAD (σ={sigma:g}): <strong>minor</strong> = metric 1.0–1.25× "
        f"threshold, <strong>moderate</strong> = 1.25–2.0×, <strong>severe</strong> = ≥2.0×. "
        f"Constant-hold and single-frame-spike candidates use the same ratio tiers on their "
        f"respective metrics. <strong>Segment swap</strong> (rigid-body pair length): "
        f"<strong>moderate</strong> = distance deviates &gt;{seg_pct:g}% from the session "
        f"median, <strong>severe/swap</strong> = deviation ≥50% (likely marker identity error). "
        f"When several methods fire on the same frame, the highest severity is shown."
    )


def _load_marker_pair_map(config: dict[str, Any]) -> pd.DataFrame | None:
    """Load config/marker_pair_map.csv if present (declared rigid-body pairs)."""
    from motive_qc.core import base_dir_from_config

    rb_cfg = _artifacts_config(config).get("rigid_body", {})
    rel = rb_cfg.get("marker_pair_map", "config/marker_pair_map.csv")
    try:
        path = base_dir_from_config(config) / rel
    except Exception:
        path = Path(rel)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "include_in_qc" in df.columns:
        df = df[df["include_in_qc"].astype(str).str.lower().isin({"true", "1", "yes"})]
    return df


def _bootstrap_marker_pairs(
    session: MotiveSession,
    config: dict[str, Any],
    marker_names: list[str],
    name_to_idx: dict[str, int],
    coords: np.ndarray,
    max_pair_distance_m: float,
) -> list[tuple[str, str, str, str]]:
    """Auto-pair in-analysis labeled markers within each body region by proximity.

    Pairs every in-group marker pair whose robust (median) separation is finite and
    within ``max_pair_distance_m`` -- a reasonable rigid-body proxy when no explicit
    marker_pair_map.csv is provided.
    """
    inv = session.marker_inventory
    rb_cfg = _artifacts_config(config).get("rigid_body", {})
    # A rigid pair has a near-constant separation. Use std/mean (penalizes
    # movement excursions, unlike MAD) and pair each marker only with its single
    # most-rigid SAME-SIDE in-group partner. This rejects cross-body pairs
    # (e.g. LHeel-RHeel) and intermittent-movement pairs that would otherwise
    # flood the swap flag with false positives.
    max_cv = float(rb_cfg.get("max_rigid_cv", 0.05))
    labeled = set(analysis_labeled_marker_names(inv, config))
    sub = inv[inv["marker_name"].isin(labeled)]
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str, str, str]] = []
    for region, grp in sub.groupby("body_region_group"):
        names = [m for m in grp["marker_name"] if m in name_to_idx]
        for a in names:
            best_b, best_cv = None, max_cv
            for b in names:
                if b == a or _marker_side(b) != _marker_side(a):
                    continue
                dist = np.linalg.norm(
                    coords[:, name_to_idx[a], :] - coords[:, name_to_idx[b], :], axis=1
                )
                mean = float(np.nanmean(dist))
                if not np.isfinite(mean) or mean <= 0 or mean > max_pair_distance_m:
                    continue
                cv = float(np.nanstd(dist)) / mean
                if cv < best_cv:
                    best_cv, best_b = cv, b
            if best_b is None:
                continue
            key = tuple(sorted((a, best_b)))
            if key in seen:
                continue
            seen.add(key)
            short_a, short_b = key[0].split(":")[-1], key[1].split(":")[-1]
            pairs.append((f"{short_a}__{short_b}", key[0], key[1], str(region)))
    return pairs


def _marker_side(marker_name: str) -> str:
    """Best-effort body side from a marker name: 'L', 'R', or '' (midline)."""
    short = marker_name.split(":")[-1]
    low = short.lower()
    if low.startswith("l") and len(short) > 1 and short[1].isupper():
        return "L"
    if low.startswith("r") and len(short) > 1 and short[1].isupper():
        return "R"
    if "left" in low:
        return "L"
    if "right" in low:
        return "R"
    return ""


def detect_segment_length_violations(
    session: MotiveSession,
    config: dict[str, Any],
    gap_events: pd.DataFrame,
    candidate_id_start: int = 0,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Vectorized rigid-body distance (marker-swap) check.

    For each declared/auto pair compute frame-wise Euclidean distance with one
    ``np.linalg.norm`` call, establish a robust ``np.nanmedian`` baseline (gaps
    ignored), and flag frames deviating beyond ``max_segment_length_change_pct``
    even when velocity is sub-sigma. No per-frame Python loops.
    """
    rb_cfg = _artifacts_config(config).get("rigid_body", {})
    empty_qc = pd.DataFrame(
        columns=[
            "pair_name",
            "marker_a",
            "marker_b",
            "body_region_group",
            "median_distance_m",
            "n_frames_violating",
            "pct_frames_violating",
            "n_events",
        ]
    )
    if not rb_cfg.get("enabled", True):
        return [], empty_qc

    marker_names = list(session.coordinates.coords["marker"].values)
    name_to_idx = {m: i for i, m in enumerate(marker_names)}
    coords = session.coordinates.values  # (n_frames, n_markers, 3)
    frames = session.coordinates.coords["frame"].values
    times = session.time_seconds.values

    pair_map = _load_marker_pair_map(config)
    max_pair_distance_m = float(rb_cfg.get("max_pair_distance_m", 0.5))
    pairs: list[tuple[str, str, str, str]] = []
    if pair_map is not None and not pair_map.empty:
        inv = session.marker_inventory.set_index("marker_name")
        for _, r in pair_map.iterrows():
            a, b = str(r.get("marker_a")), str(r.get("marker_b"))
            if a in name_to_idx and b in name_to_idx:
                region = (
                    inv.loc[a, "body_region_group"] if a in inv.index else r.get("body_region", "")
                )
                pairs.append((str(r.get("pair_name", f"{a}__{b}")), a, b, str(region)))
    else:
        pairs = _bootstrap_marker_pairs(
            session, config, marker_names, name_to_idx, coords, max_pair_distance_m
        )

    max_change = float(rb_cfg.get("max_segment_length_change_pct", 18.0)) / 100.0
    candidates: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    candidate_id = candidate_id_start

    for pair_name, a, b, region in pairs:
        pa = coords[:, name_to_idx[a], :]
        pb = coords[:, name_to_idx[b], :]
        dist = np.linalg.norm(pa - pb, axis=1)
        median = float(np.nanmedian(dist))
        if not np.isfinite(median) or median <= 0:
            continue
        deviation = np.abs(dist - median) / median
        violating = np.isfinite(dist) & (deviation > max_change)
        viol_idx = np.where(violating)[0]
        qc_rows.append(
            {
                "pair_name": pair_name,
                "marker_a": a,
                "marker_b": b,
                "body_region_group": region,
                "median_distance_m": round(median, 6),
                "n_frames_violating": int(viol_idx.size),
                "pct_frames_violating": round(
                    100.0 * viol_idx.size / len(dist), 6
                )
                if len(dist)
                else 0.0,
                "n_events": 0,
            }
        )
        for idx in viol_idx:
            candidate_id += 1
            ratio = 1.0 + float(deviation[idx])
            severity = "severe" if deviation[idx] >= 0.5 else "moderate"
            candidates.append(
                {
                    "candidate_id": f"C{candidate_id:06d}",
                    "marker_name": pair_name,
                    "body_region_group": region,
                    "frame": int(frames[idx]),
                    "time_seconds": round(float(times[idx]), 6),
                    "method": "segment_length_violation",
                    "severity": severity,
                    "metric_value": round(float(dist[idx]), 6),
                    "threshold": round(median, 6),
                    "near_gap": False,
                    "recommended_status": "manual_review",
                }
            )
    qc_df = pd.DataFrame(qc_rows) if qc_rows else empty_qc
    return candidates, qc_df


def _detect_constant_holds(
    session: MotiveSession,
    marker_name: str,
    inv_row: pd.Series,
    config: dict[str, Any],
    candidate_id: int,
    gap_events: pd.DataFrame,
) -> tuple[list[dict[str, Any]], int]:
    art_cfg = _artifacts_config(config)
    hold_cfg = art_cfg.get("constant_position_hold", {})
    min_frames = int(hold_cfg.get("min_repeated_frames", 3))
    tol = float(hold_cfg.get("tolerance_m", 1e-6))
    candidates: list[dict[str, Any]] = []

    for start_idx, end_idx in marker_valid_segments(session, marker_name):
        if end_idx - start_idx + 1 < min_frames:
            continue
        pos = segment_positions(session, marker_name, start_idx, end_idx)
        frames = session.coordinates.coords["frame"].values
        i = 0
        while i < len(pos):
            j = i + 1
            while j < len(pos) and np.linalg.norm(pos[j] - pos[i]) <= tol:
                j += 1
            run_len = j - i
            if run_len >= min_frames:
                frame_idx = start_idx + i
                frame = int(frames[frame_idx])
                time_s = float(session.time_seconds.iloc[frame_idx])
                candidate_id += 1
                candidates.append(
                    _candidate_row(
                        candidate_id,
                        marker_name,
                        inv_row,
                        frame,
                        time_s,
                        "constant_position_hold",
                        float(run_len),
                        float(min_frames),
                        gap_events,
                        config,
                    )
                )
            i = max(j, i + 1)
    return candidates, candidate_id


def _detect_single_frame_spikes(
    session: MotiveSession,
    marker_name: str,
    inv_row: pd.Series,
    config: dict[str, Any],
    candidate_id: int,
    gap_events: pd.DataFrame,
) -> tuple[list[dict[str, Any]], int]:
    art_cfg = _artifacts_config(config)
    spike_cfg = art_cfg.get("single_frame_spike", {})
    return_tol = float(spike_cfg.get("return_near_original_tolerance_m", 0.005))
    min_jump = float(spike_cfg.get("min_jump_distance_m", 0.05))
    dt = 1.0 / float(session.metadata["effective_frame_rate_hz"])
    candidates: list[dict[str, Any]] = []

    for start_idx, end_idx in marker_valid_segments(session, marker_name):
        if end_idx - start_idx < 2:
            continue
        pos = segment_positions(session, marker_name, start_idx, end_idx)
        frames = session.coordinates.coords["frame"].values
        for local_i in range(1, len(pos) - 1):
            d_prev = np.linalg.norm(pos[local_i] - pos[local_i - 1])
            d_return = np.linalg.norm(pos[local_i + 1] - pos[local_i - 1])
            if d_prev >= min_jump and d_return <= return_tol:
                frame_idx = start_idx + local_i
                frame = int(frames[frame_idx])
                candidate_id += 1
                candidates.append(
                    _candidate_row(
                        candidate_id,
                        marker_name,
                        inv_row,
                        frame,
                        float(session.time_seconds.iloc[frame_idx]),
                        "single_frame_spike",
                        float(d_prev),
                        min_jump,
                        gap_events,
                        config,
                    )
                )
    return candidates, candidate_id


def _detect_velocity_acceleration(
    session: MotiveSession,
    marker_name: str,
    inv_row: pd.Series,
    config: dict[str, Any],
    candidate_id: int,
    gap_events: pd.DataFrame,
) -> tuple[list[dict[str, Any]], int]:
    art_cfg = _artifacts_config(config)
    methods = art_cfg.get("methods", {})
    min_neighbors = int(art_cfg.get("minimum_valid_neighbors", 2))
    min_segment = min_neighbors + 2
    dt = 1.0 / float(session.metadata["effective_frame_rate_hz"])
    vel_mult = float(art_cfg.get("velocity_mad_multiplier", 8.0))
    acc_mult = float(art_cfg.get("acceleration_mad_multiplier", 8.0))
    vel_pct = float(art_cfg.get("velocity_percentile_threshold", 99.9))
    acc_pct = float(art_cfg.get("acceleration_percentile_threshold", 99.9))

    all_speeds: list[float] = []
    all_acc: list[float] = []
    segment_speeds: list[tuple[int, int, np.ndarray]] = []
    segment_acc: list[tuple[int, int, np.ndarray]] = []

    for start_idx, end_idx in marker_valid_segments(session, marker_name):
        if end_idx - start_idx + 1 < min_segment:
            continue
        pos = segment_positions(session, marker_name, start_idx, end_idx)
        speeds = compute_speeds(pos, dt)
        if speeds.size:
            all_speeds.extend(speeds.tolist())
            segment_speeds.append((start_idx, end_idx, speeds))
            acc = compute_accelerations(speeds, dt)
            if acc.size:
                all_acc.extend(acc.tolist())
                segment_acc.append((start_idx, end_idx, acc))

    candidates: list[dict[str, Any]] = []
    frames = session.coordinates.coords["frame"].values

    if methods.get("velocity_mad", True) and all_speeds:
        speed_arr = np.array(all_speeds)
        _, _, vel_mad_thr = robust_mad_threshold(speed_arr, vel_mult)
        vel_pct_thr = float(np.percentile(speed_arr, vel_pct))
        vel_threshold = max(vel_mad_thr, vel_pct_thr)
        for start_idx, end_idx, speeds in segment_speeds:
            for local_i, speed in enumerate(speeds):
                if speed < vel_threshold:
                    continue
                is_peak = True
                if local_i > 0 and speeds[local_i - 1] >= speed:
                    is_peak = False
                if local_i < len(speeds) - 1 and speeds[local_i + 1] > speed:
                    is_peak = False
                if not is_peak:
                    continue
                frame_idx = start_idx + local_i + 1
                frame = int(frames[frame_idx])
                candidate_id += 1
                candidates.append(
                    _candidate_row(
                        candidate_id,
                        marker_name,
                        inv_row,
                        frame,
                        float(session.time_seconds.iloc[frame_idx]),
                        "velocity_mad",
                        float(speed),
                        float(vel_threshold),
                        gap_events,
                        config,
                    )
                )

    if methods.get("acceleration_mad", True) and all_acc:
        acc_arr = np.abs(np.array(all_acc))
        _, _, acc_mad_thr = robust_mad_threshold(acc_arr, acc_mult)
        acc_pct_thr = float(np.percentile(acc_arr, acc_pct))
        acc_threshold = max(acc_mad_thr, acc_pct_thr)
        for start_idx, end_idx, acc in segment_acc:
            abs_acc = np.abs(acc)
            for local_i, a in enumerate(abs_acc):
                if a < acc_threshold:
                    continue
                is_peak = True
                if local_i > 0 and abs_acc[local_i - 1] >= a:
                    is_peak = False
                if local_i < len(abs_acc) - 1 and abs_acc[local_i + 1] > a:
                    is_peak = False
                if not is_peak:
                    continue
                frame_idx = start_idx + local_i + 2
                if frame_idx > end_idx:
                    continue
                frame = int(frames[frame_idx])
                candidate_id += 1
                candidates.append(
                    _candidate_row(
                        candidate_id,
                        marker_name,
                        inv_row,
                        frame,
                        float(session.time_seconds.iloc[frame_idx]),
                        "acceleration_mad",
                        float(a),
                        float(acc_threshold),
                        gap_events,
                        config,
                    )
                )
    return candidates, candidate_id


def _candidate_row(
    candidate_id: int,
    marker_name: str,
    inv_row: pd.Series,
    frame: int,
    time_s: float,
    method: str,
    metric_value: float,
    threshold: float,
    gap_events: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    art_cfg = _artifacts_config(config)
    near_gap = frames_near_gap(
        frame,
        gap_events,
        int(art_cfg.get("max_frames_after_gap_for_velocity", 0)),
    )
    ratio = metric_value / threshold if threshold > 0 else 0.0
    severity = "minor"
    if ratio >= 2.0:
        severity = "severe"
    elif ratio >= 1.25:
        severity = "moderate"
    rec = "document" if severity == "minor" else "caution"
    return {
        "candidate_id": f"C{candidate_id:06d}",
        "marker_name": marker_name,
        "body_region_group": inv_row["body_region_group"],
        "frame": frame,
        "time_seconds": round(time_s, 6),
        "method": method,
        "severity": severity,
        "metric_value": round(metric_value, 6),
        "threshold": round(threshold, 6),
        "near_gap": near_gap,
        "recommended_status": rec,
    }


def _event_class(duration_frames: int, method: str) -> str:
    if duration_frames == 1:
        return "single_frame_spike" if method == "single_frame_spike" else "single_frame"
    if duration_frames <= 5:
        return "short_burst"
    return "sustained"


def cluster_artifact_events(
    candidates: pd.DataFrame,
    frame_rate: float,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "marker_name",
                "body_region_group",
                "method",
                "start_frame",
                "end_frame",
                "duration_frames",
                "duration_seconds",
                "event_class",
                "severity",
                "near_gap",
                "methods_in_event",
            ]
        )

    severity_rank = {"minor": 0, "moderate": 1, "severe": 2}

    def _flush_chunk(
        marker: str,
        method: str,
        chunk_start: int,
        prev: int,
        chunk: pd.DataFrame,
        eid: int,
    ) -> tuple[dict[str, Any], int]:
        dur = prev - chunk_start + 1
        sev = max(chunk["severity"].tolist(), key=lambda s: severity_rank.get(s, 0))
        eid += 1
        return (
            {
                "event_id": f"E{eid:06d}",
                "marker_name": marker,
                "body_region_group": chunk.iloc[0]["body_region_group"],
                "method": method,
                "start_frame": chunk_start,
                "end_frame": prev,
                "start_time_s": round(float(chunk["time_seconds"].min()), 6),
                "end_time_s": round(float(chunk["time_seconds"].max()), 6),
                "duration_frames": dur,
                "duration_seconds": round(dur / frame_rate, 6),
                "event_class": _event_class(dur, method),
                "severity": sev,
                "near_gap": bool(chunk["near_gap"].any()),
                "methods_in_event": method,
                "peak_metric_value": round(float(chunk["metric_value"].max()), 6)
                if "metric_value" in chunk.columns
                else None,
            },
            eid,
        )

    events: list[dict[str, Any]] = []
    event_id = 0
    for (marker, method), group in candidates.sort_values("frame").groupby(
        ["marker_name", "method"], sort=False
    ):
        g = group.reset_index(drop=True)
        frames = g["frame"].astype(int).to_numpy()
        run_start = 0
        for i in range(1, len(frames) + 1):
            if i == len(frames) or frames[i] != frames[i - 1] + 1:
                chunk = g.iloc[run_start:i]
                row, event_id = _flush_chunk(
                    marker, method, int(frames[run_start]), int(frames[i - 1]), chunk, event_id
                )
                events.append(row)
                run_start = i

    events_df = pd.DataFrame(events)
    vel_frames = set(
        candidates.loc[candidates["method"] == "velocity_mad", "frame"].astype(int).tolist()
    )
    acc_frames = set(
        candidates.loc[candidates["method"] == "acceleration_mad", "frame"].astype(int).tolist()
    )
    overlap_frames = vel_frames & acc_frames
    if overlap_frames:
        for idx, row in events_df.iterrows():
            fr = set(range(int(row["start_frame"]), int(row["end_frame"]) + 1))
            if row["method"] == "velocity_mad" and fr & overlap_frames:
                if any(
                    (events_df["method"] == "acceleration_mad")
                    & (events_df["start_frame"] <= row["end_frame"])
                    & (events_df["end_frame"] >= row["start_frame"])
                ):
                    events_df.at[idx, "methods_in_event"] = "velocity+acceleration"
    return events_df


def build_artifact_session_summary(
    candidates: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    vel_frames: set[int] = set()
    acc_frames: set[int] = set()
    if not candidates.empty:
        vel_frames = set(
            candidates.loc[candidates["method"] == "velocity_mad", "frame"].astype(int).tolist()
        )
        acc_frames = set(
            candidates.loc[candidates["method"] == "acceleration_mad", "frame"].astype(int).tolist()
        )
    n_both = len(vel_frames & acc_frames)

    if events.empty:
        recommendation = "No artifact events detected."
        n_events = 0
        by_class = {}
    else:
        n_events = len(events)
        by_class = events["event_class"].value_counts().to_dict()
        n_single = int(by_class.get("single_frame_spike", 0) + by_class.get("single_frame", 0))
        n_sustained = int(by_class.get("sustained", 0))
        if n_single > n_events * 0.7:
            recommendation = (
                "Most detections are single-frame spikes; review sustained events and "
                "critical-region overlaps first."
            )
        elif n_sustained > 0:
            recommendation = (
                f"{n_sustained} sustained artifact event(s) detected; priority visual review."
            )
        else:
            recommendation = "Review short-burst artifact events overlapping planned analysis windows."

    row = {
        "n_frame_candidates": len(candidates),
        "n_events": n_events,
        "n_single_frame_events": int(
            events[events["event_class"].isin(["single_frame", "single_frame_spike"])].shape[0]
        )
        if not events.empty
        else 0,
        "n_short_burst_events": int((events["event_class"] == "short_burst").sum())
        if not events.empty
        else 0,
        "n_sustained_events": int((events["event_class"] == "sustained").sum())
        if not events.empty
        else 0,
        "n_frames_velocity_candidate": len(vel_frames),
        "n_frames_acceleration_candidate": len(acc_frames),
        "n_frames_both_velocity_and_acceleration": n_both,
        "recommendation": recommendation,
    }
    return pd.DataFrame([row])


UNLABELED_BODY_GROUP = "unlabeled"


def list_velocity_histogram_groups(
    session: MotiveSession, config: dict[str, Any] | None = None
) -> list[str]:
    """Body-region groups for per-segment velocity histograms (labeled markers only)."""
    inv = session.marker_inventory
    excluded = set()
    if config:
        from motive_qc.analysis_scope import excluded_body_groups

        excluded = excluded_body_groups(config)
    groups = sorted(
        g
        for g in inv.loc[inv["is_labeled"], "body_region_group"].unique()
        if g and str(g) != UNLABELED_BODY_GROUP and str(g) not in excluded
    )
    return ["all_labeled", *groups]


def flagged_velocity_speeds(
    artifact_candidates: pd.DataFrame,
    session: MotiveSession,
    body_region_group: str | None = None,
    artifact_events: pd.DataFrame | None = None,
) -> np.ndarray:
    vel = pd.DataFrame()
    if not artifact_candidates.empty and "method" in artifact_candidates.columns:
        vel = artifact_candidates[artifact_candidates["method"] == "velocity_mad"]
    elif artifact_events is not None and not artifact_events.empty:
        vel = artifact_events[artifact_events["method"] == "velocity_mad"].copy()
        if "peak_metric_value" in vel.columns:
            vel = vel.rename(columns={"peak_metric_value": "metric_value"})
    if vel.empty or "metric_value" not in vel.columns:
        return np.array([], dtype=float)
    if body_region_group and body_region_group != "all_labeled":
        inv = session.marker_inventory.set_index("marker_name")
        markers = set(
            inv[
                inv["is_labeled"] & (inv["body_region_group"] == body_region_group)
            ].index
        )
        vel = vel[vel["marker_name"].isin(markers)]
    return vel["metric_value"].astype(float).values


def collect_session_velocity_distribution(
    session: MotiveSession,
    config: dict[str, Any],
    body_region_group: str | None = None,
) -> dict[str, Any]:
    """Pool gap-safe labeled-marker speeds for histogram / threshold review."""
    art_cfg = _artifacts_config(config)
    min_neighbors = int(art_cfg.get("minimum_valid_neighbors", 2))
    min_segment = min_neighbors + 2
    dt = 1.0 / float(session.metadata["effective_frame_rate_hz"])
    vel_mult = float(art_cfg.get("velocity_mad_multiplier", 11.0))
    vel_pct = float(art_cfg.get("velocity_percentile_threshold", 99.97))

    inventory = session.marker_inventory
    if body_region_group and body_region_group != "all_labeled":
        if body_group_excluded(body_region_group, config):
            labeled: list[str] = []
        else:
            labeled = inventory.loc[
                inventory["is_labeled"]
                & (inventory["body_region_group"] == body_region_group),
                "marker_name",
            ].tolist()
    else:
        labeled = analysis_labeled_marker_names(inventory, config)
    all_speeds: list[float] = []
    for marker in labeled:
        for start_idx, end_idx in marker_valid_segments(session, marker):
            if end_idx - start_idx + 1 < min_segment:
                continue
            pos = segment_positions(session, marker, start_idx, end_idx)
            speeds = compute_speeds(pos, dt)
            if speeds.size:
                all_speeds.extend(speeds.tolist())

    speed_arr = np.array(all_speeds, dtype=float)
    median, mad, vel_mad_thr = robust_mad_threshold(speed_arr, vel_mult)
    vel_pct_thr = float(np.percentile(speed_arr, vel_pct)) if speed_arr.size else 0.0
    vel_threshold = max(vel_mad_thr, vel_pct_thr) if speed_arr.size else 0.0

    return {
        "speeds_m_s": speed_arr,
        "median_m_s": median,
        "mad_m_s": mad,
        "vel_mad_threshold_m_s": vel_mad_thr,
        "vel_percentile_threshold_m_s": vel_pct_thr,
        "vel_threshold_m_s": vel_threshold,
        "vel_percentile_config": vel_pct,
        "vel_mad_multiplier": vel_mult,
        "n_speed_samples": int(speed_arr.size),
        "length_units": session.metadata.get("length_units"),
        "body_region_group": body_region_group or "all_labeled",
        "n_markers": len(labeled),
    }


def run_layer4_artifacts(
    session: MotiveSession,
    layer2_result: QCResult,
    config: dict[str, Any],
    verbose: bool = False,
) -> QCResult:
    art_cfg = _artifacts_config(config)
    if not art_cfg.get("enabled", False):
        return QCResult(layer_name="layer4", status="skipped", session=session)

    length_units = session.metadata.get("length_units")
    require_units = art_cfg.get("require_known_units", True)
    messages = list(session.validation_messages)
    if not length_units:
        msg = QCMessage(
            "WARNING",
            "UNITS_REQUIRED_FOR_ARTIFACTS",
            "Length units unknown; velocity/acceleration artifact screening skipped.",
        )
        messages.append(msg)
        if require_units:
            return QCResult(
                layer_name="layer4",
                status="skipped",
                messages=messages,
                session=session,
            )

    if verbose:
        LOGGER.info("Running Layer 4 artifact candidate screening")

    gap_events = layer2_result.tables.get("gap_events", pd.DataFrame())
    inventory = session.marker_inventory
    labeled = analysis_labeled_marker_names(inventory, config)
    methods = art_cfg.get("methods", {})

    all_candidates: list[dict[str, Any]] = []
    candidate_id = 0

    for marker in labeled:
        inv = inventory.set_index("marker_name").loc[marker]
        if methods.get("constant_position_hold", True):
            cands, candidate_id = _detect_constant_holds(
                session, marker, inv, config, candidate_id, gap_events
            )
            all_candidates.extend(cands)
        if methods.get("single_frame_spike", True):
            cands, candidate_id = _detect_single_frame_spikes(
                session, marker, inv, config, candidate_id, gap_events
            )
            all_candidates.extend(cands)
        if methods.get("velocity_mad", True) or methods.get("acceleration_mad", True):
            cands, candidate_id = _detect_velocity_acceleration(
                session, marker, inv, config, candidate_id, gap_events
            )
            all_candidates.extend(cands)

    segment_candidates, segment_length_qc = detect_segment_length_violations(
        session, config, gap_events, candidate_id_start=candidate_id
    )
    if segment_candidates:
        all_candidates.extend(segment_candidates)

    artifact_candidates = pd.DataFrame(all_candidates)
    frame_rate = float(session.metadata["effective_frame_rate_hz"])
    artifact_events = cluster_artifact_events(artifact_candidates, frame_rate)

    if not segment_length_qc.empty and not artifact_events.empty:
        swap_counts = (
            artifact_events[artifact_events["method"] == "segment_length_violation"]
            .groupby("marker_name")
            .size()
        )
        segment_length_qc["n_events"] = (
            segment_length_qc["pair_name"].map(swap_counts).fillna(0).astype(int)
        )
    artifact_session_summary = build_artifact_session_summary(
        artifact_candidates, artifact_events
    )

    if artifact_candidates.empty:
        artifact_summary = pd.DataFrame(
            columns=["marker_name", "method", "candidate_count"]
        )
    else:
        artifact_summary = (
            artifact_candidates.groupby(["marker_name", "method"])
            .size()
            .reset_index(name="candidate_count")
        )

    figures: dict[str, Path] = {}
    plots_cfg = config.get("outputs", {}).get("plots", {})
    if plots_cfg.get("artifact_velocity_histogram", True):
        velocity_dist = collect_session_velocity_distribution(session, config)
        flagged_speeds = flagged_velocity_speeds(
            artifact_candidates, session, artifact_events=artifact_events
        )
        figures["artifact_velocity_histogram"] = plot_velocity_artifact_histogram(
            velocity_dist, flagged_speeds, config
        )
        for group in list_velocity_histogram_groups(session, config):
            if group == "all_labeled":
                continue
            group_dist = collect_session_velocity_distribution(session, config, group)
            group_flagged = flagged_velocity_speeds(
                artifact_candidates, session, group, artifact_events=artifact_events
            )
            figures[f"artifact_velocity_histogram__{group}"] = (
                plot_velocity_artifact_histogram(
                    group_dist, group_flagged, config, body_region_group=group
                )
            )
    if plots_cfg.get("artifact_timeline", False):
        if not artifact_events.empty:
            figures["artifact_timeline"] = plot_artifact_events_timeline(
                artifact_events, config
            )
        elif not artifact_candidates.empty:
            figures["artifact_timeline"] = plot_artifact_timeline(
                artifact_candidates, config
            )

    tables: dict[str, pd.DataFrame] = {
        "artifact_events": artifact_events,
        "artifact_session_summary": artifact_session_summary,
        "artifact_summary_by_marker": artifact_summary,
        "artifact_candidates": artifact_candidates,
        "segment_length_qc": segment_length_qc,
    }

    return QCResult(
        layer_name="layer4",
        status="pass",
        tables=tables,
        figures=figures,
        messages=messages,
        session=session,
    )
