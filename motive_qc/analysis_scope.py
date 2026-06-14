"""Scope markers/body groups included in QC calculations.

Single source of truth for the three-way marker partition used everywhere
downstream:

* ``labeled_in_analysis`` -- real subject markers that feed missingness, gaps,
  artifacts, masks, and the preprocessing verdict (the denominator).
* ``unlabeled``           -- name-based unlabeled tracks, reported only.
* ``quarantined``         -- labeled markers removed from analysis because they
  belong to a losing/phantom skeleton, are never solved
  (>= ``never_solved_missing_pct`` missing), or are a duplicate marker copy.
  Reported, never counted as missing data.

The partition is computed once at parse time (``compute_marker_analysis_flags``)
and stored on ``marker_inventory`` so gaps/windows/artifacts/masks all agree.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from motive_qc.core import QCMessage
from motive_qc.marker_meta import parse_marker_identity


def excluded_body_groups(config: dict[str, Any]) -> set[str]:
    groups = config.get("markers", {}).get("exclude_body_groups_from_analysis", [])
    return {str(g) for g in groups}


def never_solved_threshold(config: dict[str, Any]) -> float:
    return float(config.get("markers", {}).get("never_solved_missing_pct", 95.0))


def skeleton_overlap_min_markers(config: dict[str, Any]) -> int:
    return int(config.get("markers", {}).get("skeleton_overlap_min_markers", 10))


def _ensure_identity_columns(inventory: pd.DataFrame) -> pd.DataFrame:
    inventory = inventory.copy()
    if "skeleton_prefix" not in inventory.columns or "canonical_short_name" not in inventory.columns:
        identities = inventory["marker_name"].map(parse_marker_identity)
        inventory["skeleton_prefix"] = [i[0] for i in identities]
        inventory["canonical_short_name"] = [i[1] for i in identities]
    return inventory


def _score_skeleton(
    marker_names: list[str],
    missing_percent_by_marker: dict[str, float],
) -> tuple[float, int, int]:
    """Higher is better: mean coverage, then count of tracked markers, then size."""
    coverages = [100.0 - float(missing_percent_by_marker.get(n, 100.0)) for n in marker_names]
    n_tracked = sum(1 for c in coverages if c > 0.0)
    return (float(np.mean(coverages)), n_tracked, len(marker_names))


def _cluster_competing_skeletons(
    profiles: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> list[set[str]]:
    """Group skeleton prefixes that share enough canonical anatomical marker names."""
    skels = [s for s in profiles if s]
    if len(skels) < 2:
        return []

    min_overlap = skeleton_overlap_min_markers(config)
    visited: set[str] = set()
    clusters: list[set[str]] = []

    for skel in skels:
        if skel in visited:
            continue
        cluster = {skel}
        queue = [skel]
        visited.add(skel)
        while queue:
            current = queue.pop()
            canon_a = profiles[current]["canonicals"]
            for other in skels:
                if other in visited:
                    continue
                canon_b = profiles[other]["canonicals"]
                inter = canon_a & canon_b
                threshold = max(
                    min_overlap,
                    int(0.5 * min(len(canon_a), len(canon_b))),
                )
                if len(inter) >= threshold:
                    cluster.add(other)
                    visited.add(other)
                    queue.append(other)
        if len(cluster) > 1:
            clusters.append(cluster)
    return clusters


def _resolve_competing_skeletons(
    labeled: pd.DataFrame,
    missing_percent_by_marker: dict[str, float],
    config: dict[str, Any],
) -> tuple[dict[str, str], str | None, list[dict[str, Any]]]:
    """Quarantine entire losing skeletons when multiple labeled sets overlap."""
    reasons: dict[str, str] = {}
    events: list[dict[str, Any]] = []
    chosen: str | None = None

    if not bool(config.get("markers", {}).get("detect_duplicate_marker_sets", True)):
        return reasons, chosen, events
    if "skeleton_prefix" not in labeled.columns or "canonical_short_name" not in labeled.columns:
        return reasons, chosen, events

    profiles: dict[str, dict[str, Any]] = {}
    for skel, grp in labeled.groupby("skeleton_prefix"):
        skel_key = str(skel).strip()
        if not skel_key:
            continue
        names = grp["marker_name"].tolist()
        profiles[skel_key] = {
            "names": names,
            "canonicals": set(grp["canonical_short_name"].astype(str)),
            "score": _score_skeleton(names, missing_percent_by_marker),
        }

    if len(profiles) < 2:
        if len(profiles) == 1:
            chosen = next(iter(profiles))
        return reasons, chosen, events

    for cluster in _cluster_competing_skeletons(profiles, config):
        ranked = sorted(cluster, key=lambda s: profiles[s]["score"], reverse=True)
        winner = ranked[0]
        losers = ranked[1:]
        if chosen is None:
            chosen = winner
        elif winner == chosen or profiles[winner]["score"] > profiles.get(chosen, {}).get("score", (0, 0, 0)):
            chosen = winner

        for loser in losers:
            for name in profiles[loser]["names"]:
                reasons[name] = "phantom_skeleton"
            events.append(
                {
                    "chosen_skeleton": winner,
                    "ignored_skeleton": loser,
                    "n_markers_quarantined": len(profiles[loser]["names"]),
                    "winner_mean_coverage_pct": round(profiles[winner]["score"][0], 4),
                    "loser_mean_coverage_pct": round(profiles[loser]["score"][0], 4),
                }
            )

    if chosen is None and len(profiles) == 1:
        chosen = next(iter(profiles))
    return reasons, chosen, events


def compute_marker_quarantine(
    inventory: pd.DataFrame,
    missing_percent_by_marker: dict[str, float],
    config: dict[str, Any],
) -> tuple[dict[str, str], str | None, list[dict[str, Any]]]:
    """Map labeled marker -> quarantine reason.

    Order of precedence:
    1. Competing skeleton sets -> quarantine entire losing skeleton (``phantom_skeleton``)
    2. Never-solved individual markers (``never_solved``)
    3. Residual canonical-short duplicates (``duplicate_marker_set``)
    """
    reasons: dict[str, str] = {}
    skeleton_events: list[dict[str, Any]] = []
    chosen_skeleton: str | None = None

    if inventory.empty or "is_labeled" not in inventory.columns:
        return reasons, chosen_skeleton, skeleton_events

    inventory = _ensure_identity_columns(inventory)
    excluded = excluded_body_groups(config)
    labeled = inventory[inventory["is_labeled"].astype(bool)].copy()
    if "body_region_group" in labeled.columns and excluded:
        labeled = labeled[~labeled["body_region_group"].astype(str).isin(excluded)]
    if labeled.empty:
        return reasons, chosen_skeleton, skeleton_events

    skel_reasons, chosen_skeleton, skeleton_events = _resolve_competing_skeletons(
        labeled, missing_percent_by_marker, config
    )
    reasons.update(skel_reasons)

    threshold = never_solved_threshold(config)
    for name in labeled["marker_name"]:
        if name in reasons:
            continue
        if float(missing_percent_by_marker.get(name, 0.0)) >= threshold:
            reasons[name] = "never_solved"

    detect_dup = bool(
        config.get("markers", {}).get("detect_duplicate_marker_sets", True)
    )
    if detect_dup and "canonical_short_name" in labeled.columns:
        active = labeled[~labeled["marker_name"].isin(reasons)]
        for _short, grp in active.groupby("canonical_short_name"):
            names = list(grp["marker_name"])
            if len(set(names)) <= 1:
                continue
            ordered = sorted(
                names, key=lambda n: float(missing_percent_by_marker.get(n, 0.0))
            )
            for dup in ordered[1:]:
                reasons.setdefault(dup, "duplicate_marker_set")

    return reasons, chosen_skeleton, skeleton_events


def compute_marker_analysis_flags(
    inventory: pd.DataFrame,
    valid: np.ndarray,
    marker_names: list[str],
    config: dict[str, Any],
    messages: list[QCMessage] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach scope columns to ``inventory`` and return partition metadata."""
    inventory = _ensure_identity_columns(inventory.copy())
    n_frames = int(valid.shape[0]) if valid.ndim == 2 else 0
    if n_frames > 0:
        missing_counts = (~valid).sum(axis=0)
        missing_pct = 100.0 * missing_counts / n_frames
    else:
        missing_pct = np.zeros(len(marker_names), dtype=float)
    missing_by_name = {
        name: float(missing_pct[i]) for i, name in enumerate(marker_names)
    }

    reasons, chosen_skeleton, skeleton_events = compute_marker_quarantine(
        inventory, missing_by_name, config
    )
    excluded = excluded_body_groups(config)

    inventory["marker_missing_percent"] = (
        inventory["marker_name"].map(missing_by_name).astype(float).round(6)
    )
    inventory["quarantine_reason"] = (
        inventory["marker_name"].map(reasons).fillna("").astype(str)
    )

    def _in_analysis(row: pd.Series) -> bool:
        if not bool(row.get("is_labeled", False)):
            return False
        if str(row.get("body_region_group")) in excluded:
            return False
        if str(row.get("quarantine_reason") or ""):
            return False
        return True

    inventory["included_in_analysis"] = inventory.apply(_in_analysis, axis=1)

    partition_meta = {
        "analysis_skeleton_prefix": chosen_skeleton or "",
        "skeleton_selection_events": skeleton_events,
    }

    if messages is not None:
        _emit_partition_messages(
            inventory, reasons, config, messages, skeleton_events, chosen_skeleton
        )
    return inventory, partition_meta


def _emit_partition_messages(
    inventory: pd.DataFrame,
    reasons: dict[str, str],
    config: dict[str, Any],
    messages: list[QCMessage],
    skeleton_events: list[dict[str, Any]] | None = None,
    chosen_skeleton: str | None = None,
) -> None:
    phantom = sorted(n for n, r in reasons.items() if r == "phantom_skeleton")
    never_solved = sorted(n for n, r in reasons.items() if r == "never_solved")
    duplicates = sorted(n for n, r in reasons.items() if r == "duplicate_marker_set")

    if skeleton_events:
        ignored = sorted({e["ignored_skeleton"] for e in skeleton_events})
        messages.append(
            QCMessage(
                "WARNING",
                "PHANTOM_SKELETON_QUARANTINED",
                f"Competing labeled skeleton(s) detected; using '{chosen_skeleton}' for "
                f"analysis and quarantining phantom skeleton(s): {', '.join(ignored)}.",
                {
                    "chosen_skeleton": chosen_skeleton,
                    "ignored_skeletons": ignored,
                    "events": skeleton_events,
                    "n_quarantined": len(phantom),
                },
            )
        )
    elif phantom:
        messages.append(
            QCMessage(
                "WARNING",
                "PHANTOM_SKELETON_QUARANTINED",
                f"{len(phantom)} marker(s) quarantined from phantom skeleton copy(ies).",
                {"markers": phantom[:20]},
            )
        )

    if never_solved:
        messages.append(
            QCMessage(
                "WARNING",
                "MARKERS_QUARANTINED_NEVER_SOLVED",
                f"{len(never_solved)} labeled marker(s) quarantined as never-solved "
                f"(>= {never_solved_threshold(config)}% missing); excluded from missingness.",
                {"markers": never_solved[:20]},
            )
        )
    if duplicates:
        messages.append(
            QCMessage(
                "WARNING",
                "DUPLICATE_MARKER_SET_DETECTED",
                f"Duplicate labeled marker set detected; {len(duplicates)} phantom "
                "copy(ies) quarantined (most complete copy kept).",
                {"markers": duplicates[:20]},
            )
        )
    expected = config.get("markers", {}).get("expected_labeled_marker_count")
    if expected:
        n_labeled = int(inventory["is_labeled"].astype(bool).sum())
        if n_labeled > int(expected):
            messages.append(
                QCMessage(
                    "WARNING",
                    "LABELED_MARKER_COUNT_EXCEEDS_EXPECTED",
                    f"{n_labeled} labeled markers exceed expected {int(expected)}; "
                    "possible duplicate/phantom skeleton.",
                    {"n_labeled": n_labeled, "expected": int(expected)},
                )
            )


def body_group_excluded(group: str | None, config: dict[str, Any]) -> bool:
    if group is None:
        return False
    return str(group) in excluded_body_groups(config)


def marker_excluded_from_analysis(inv_row: pd.Series, config: dict[str, Any]) -> bool:
    if not bool(inv_row.get("is_labeled", False)):
        return False
    return body_group_excluded(inv_row.get("body_region_group"), config)


def analysis_labeled_marker_names(inventory: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    """Labeled markers included in gap, mask, artifact, and window calculations.

    Quarantine-aware: when ``included_in_analysis`` is present it is the single
    source of truth (excludes never-solved/duplicate markers); otherwise falls
    back to labeled minus excluded body groups.
    """
    if "included_in_analysis" in inventory.columns:
        return inventory.loc[
            inventory["included_in_analysis"].astype(bool), "marker_name"
        ].tolist()
    labeled = inventory.loc[inventory["is_labeled"]]
    excluded = excluded_body_groups(config)
    if excluded and "body_region_group" in labeled.columns:
        labeled = labeled[~labeled["body_region_group"].astype(str).isin(excluded)]
    return labeled["marker_name"].tolist()


def quarantined_marker_names(inventory: pd.DataFrame) -> list[str]:
    if "quarantine_reason" not in inventory.columns:
        return []
    mask = inventory["quarantine_reason"].astype(str).str.len() > 0
    return inventory.loc[mask, "marker_name"].tolist()


def filter_gap_events_for_analysis(
    gap_events: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    excluded = excluded_body_groups(config)
    if gap_events.empty or not excluded or "body_region_group" not in gap_events.columns:
        return gap_events
    return gap_events[~gap_events["body_region_group"].astype(str).isin(excluded)].copy()


def filter_artifact_events_for_analysis(
    events: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    excluded = excluded_body_groups(config)
    if events.empty or not excluded or "body_region_group" not in events.columns:
        return events
    return events[~events["body_region_group"].astype(str).isin(excluded)].copy()


def filter_marker_quality_for_analysis(
    marker_quality: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    if marker_quality.empty:
        return marker_quality
    if "included_in_analysis" in marker_quality.columns:
        return marker_quality[marker_quality["included_in_analysis"]].copy()
    excluded = excluded_body_groups(config)
    labeled = marker_quality[marker_quality["is_labeled"]]
    if not excluded or "body_region_group" not in labeled.columns:
        return labeled.copy()
    return labeled[~labeled["body_region_group"].astype(str).isin(excluded)].copy()


def analysis_scope_messages(config: dict[str, Any]) -> list[QCMessage]:
    excluded = sorted(excluded_body_groups(config))
    if not excluded:
        return []
    return [
        QCMessage(
            "INFO",
            "BODY_GROUPS_EXCLUDED_FROM_ANALYSIS",
            f"Body groups excluded from QC calculations: {', '.join(excluded)}.",
        )
    ]
