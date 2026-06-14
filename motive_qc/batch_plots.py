"""Batch-level plots for Layer 6 HTML report (heatmap grid + artifact histograms)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from motive_qc.analysis_scope import excluded_body_groups
from motive_qc.artifacts import (
    collect_session_velocity_distribution,
    flagged_velocity_speeds,
    velocity_mad_sigma,
    velocity_percentile_threshold,
)
from motive_qc.plots import (
    _draw_velocity_artifact_histogram,
    plot_artifact_candidate_heatmap_on_ax,
    plot_gap_severity_heatmap_on_ax,
)


def _session_label(sr: Any) -> str:
    return f"{sr.session_row.get('subject_id', '')}_{sr.session_row.get('session_id', '')}"


def _top_artifact_segments(
    sr: Any,
    config: dict[str, Any],
    n: int = 3,
) -> list[str]:
    by_seg = pd.DataFrame()
    if sr.layer5:
        by_seg = sr.layer5.tables.get("artifacts_by_segment", pd.DataFrame())
    if by_seg.empty and sr.layer4:
        events = sr.layer4.tables.get("artifact_events", pd.DataFrame())
        if events.empty or "body_region_group" not in events.columns:
            return []
        excl = excluded_body_groups(config)
        if excl:
            events = events[~events["body_region_group"].astype(str).isin(excl)]
        counts = events["body_region_group"].value_counts()
        return [str(x) for x in counts.head(n).index.tolist()]
    excl = excluded_body_groups(config)
    work = by_seg.copy()
    if excl and "body_region_group" in work.columns:
        work = work[~work["body_region_group"].astype(str).isin(excl)]
    work = work.sort_values("n_events", ascending=False)
    return work["body_region_group"].astype(str).head(n).tolist()


def plot_batch_missing_heatmaps_grid(
    session_results: list[Any],
    plots_dir: Path,
    config: dict[str, Any],
    dpi: int = 150,
) -> Path | None:
    """2×3 grid of gap-severity heatmaps (in-analysis markers only)."""
    ok = [sr for sr in session_results if getattr(sr, "batch_status", "") == "ok" and sr.layer1]
    if not ok:
        return None

    n = len(ok)
    ncols = min(3, n)
    nrows = int((n + ncols - 1) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    if n == 1:
        axes = [axes]
    else:
        axes = list(axes.flatten()) if hasattr(axes, "flatten") else list(axes)

    for idx, sr in enumerate(ok):
        ax = axes[idx]
        session = sr.layer1.session
        gap_events = sr.layer2.tables.get("gap_events", pd.DataFrame()) if sr.layer2 else pd.DataFrame()
        plot_gap_severity_heatmap_on_ax(ax, session, gap_events, config, _session_label(sr))

    for j in range(len(ok), len(axes)):
        axes[j].axis("off")

    from matplotlib.patches import Patch

    fig.legend(
        handles=[
            Patch(facecolor="#ffffff", edgecolor="#ccc", label="present"),
            Patch(facecolor="#ecc94b", label="gap 0.2–0.5 s"),
            Patch(facecolor="#e53e3e", label="gap ≥0.5 s"),
        ],
        loc="lower center",
        ncol=3,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle("Missing-frame heatmaps (in-analysis labeled markers)", fontsize=12, y=1.02)
    fig.tight_layout()
    path = plots_dir / "batch_missing_heatmaps_6sessions.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_session_artifact_heatmap(
    sr: Any,
    plots_dir: Path,
    config: dict[str, Any],
    dpi: int = 150,
) -> Path | None:
    """Single-session artifact candidate heatmap (full width)."""
    if getattr(sr, "batch_status", "") != "ok" or sr.layer1 is None or sr.layer4 is None:
        return None
    session = sr.layer1.session
    candidates = sr.layer4.tables.get("artifact_candidates", pd.DataFrame())
    sigma = velocity_mad_sigma(config)
    label = _session_label(sr)

    fig, ax = plt.subplots(figsize=(14, max(6, len(session.marker_inventory) * 0.12)))
    plot_artifact_candidate_heatmap_on_ax(
        ax, session, candidates, config, label, sigma=sigma, show_marker_labels=True
    )
    from matplotlib.patches import Patch

    seg_pct = float(config.get("artifacts", {}).get("rigid_body", {}).get("max_segment_length_change_pct", 18))
    ax.legend(
        handles=[
            Patch(facecolor="#ffffff", edgecolor="#ccc", label="none"),
            Patch(facecolor="#feebc8", label="minor (1.0–1.25× threshold)"),
            Patch(facecolor="#ed8936", label="moderate (1.25–2.0×)"),
            Patch(facecolor="#c53030", label=f"severe (≥2.0× MAD σ={sigma:g}) / swap (≥50% seg)"),
        ],
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=7,
        frameon=True,
    )
    fig.text(
        0.5,
        0.01,
        (
            f"Segment swap moderate = pair length >{seg_pct:g}% from session median. "
            "Max severity per frame when multiple methods fire."
        ),
        ha="center",
        fontsize=7,
        color="#4a5568",
    )
    fig.subplots_adjust(right=0.82, bottom=0.08)
    sess_dir = plots_dir / "sessions"
    sess_dir.mkdir(parents=True, exist_ok=True)
    path = sess_dir / f"{label}_artifact_heatmap.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_session_top3_artifact_histograms(
    sr: Any,
    plots_dir: Path,
    config: dict[str, Any],
    dpi: int = 150,
) -> Path | None:
    """Top-3 body segments by artifact burden — velocity histogram panels."""
    if getattr(sr, "batch_status", "") != "ok" or sr.layer1 is None or sr.layer4 is None:
        return None
    session = sr.layer1.session
    n_seg = int(config.get("outputs", {}).get("batch", {}).get("plot_top_artifact_segments", 3))
    segments = _top_artifact_segments(sr, config, n=n_seg)
    if not segments:
        segments = ["all_labeled"][:1]

    candidates = sr.layer4.tables.get("artifact_candidates", pd.DataFrame())
    events = sr.layer4.tables.get("artifact_events", pd.DataFrame())

    fig, axes = plt.subplots(1, len(segments), figsize=(5 * len(segments), 4))
    if len(segments) == 1:
        axes = [axes]
    for ax, seg in zip(axes, segments):
        dist = collect_session_velocity_distribution(session, config, seg)
        flagged = flagged_velocity_speeds(candidates, session, seg, artifact_events=events)
        _draw_velocity_artifact_histogram(ax, dist, flagged, seg)

    label = _session_label(sr)
    sigma = velocity_mad_sigma(config)
    pct = velocity_percentile_threshold(config)
    fig.suptitle(
        f"Artifact velocity histograms — {label}\n"
        f"(threshold: MAD σ={sigma:g}, velocity percentile={pct:g})",
        fontsize=11,
    )
    fig.tight_layout()
    sess_dir = plots_dir / "sessions"
    sess_dir.mkdir(parents=True, exist_ok=True)
    path = sess_dir / f"{label}_artifact_histograms.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def write_batch_session_plots(
    session_results: list[Any],
    plots_dir: Path,
    config: dict[str, Any],
) -> list[Path]:
    """Write all batch plot deliverables; return list of written paths."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(config.get("outputs", {}).get("dpi", 150))
    written: list[Path] = []

    grid = plot_batch_missing_heatmaps_grid(session_results, plots_dir, config, dpi=dpi)
    if grid:
        written.append(grid)

    for sr in session_results:
        p = plot_session_artifact_heatmap(sr, plots_dir, config, dpi=dpi)
        if p:
            written.append(p)
        p = plot_session_top3_artifact_histograms(sr, plots_dir, config, dpi=dpi)
        if p:
            written.append(p)

    return written
