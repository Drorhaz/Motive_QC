"""Gap-safe valid segment utilities for kinematics and spectral analysis."""

from __future__ import annotations

from typing import Any

import numpy as np

from motive_qc.core import MotiveSession


def iter_valid_segment_indices(valid: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive (start_idx, end_idx) index pairs for contiguous valid runs."""
    segments: list[tuple[int, int]] = []
    n = len(valid)
    idx = 0
    while idx < n:
        if not valid[idx]:
            idx += 1
            continue
        start = idx
        while idx < n and valid[idx]:
            idx += 1
        segments.append((start, idx - 1))
    return segments


def marker_valid_segments(session: MotiveSession, marker_name: str) -> list[tuple[int, int]]:
    valid = session.valid_marker_frame.sel(marker=marker_name).values.astype(bool)
    return iter_valid_segment_indices(valid)


def segment_positions(
    session: MotiveSession, marker_name: str, start_idx: int, end_idx: int
) -> np.ndarray:
    coords = session.coordinates.sel(marker=marker_name).values
    return coords[start_idx : end_idx + 1, :]


def compute_speeds(positions: np.ndarray, dt: float) -> np.ndarray:
    """Speed at frames 1..n-1 for an in-segment position array (n, 3)."""
    if len(positions) < 2 or dt <= 0:
        return np.array([], dtype=float)
    disp = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return disp / dt


def compute_accelerations(speeds: np.ndarray, dt: float) -> np.ndarray:
    if len(speeds) < 2 or dt <= 0:
        return np.array([], dtype=float)
    return np.diff(speeds) / dt


def robust_mad_threshold(values: np.ndarray, multiplier: float) -> tuple[float, float, float]:
    if values.size == 0:
        return 0.0, 0.0, 0.0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad == 0:
        mad = float(np.std(values)) if values.size > 1 else 0.0
    threshold = median + multiplier * mad
    return median, mad, threshold


def frames_near_gap(
    frame: int,
    gap_events,
    max_frames: int,
    labeled_only: bool = True,
) -> bool:
    if gap_events is None or gap_events.empty or max_frames <= 0:
        return False
    gaps = gap_events
    if labeled_only:
        gaps = gaps[gaps["is_labeled"]]
    for _, gap in gaps.iterrows():
        if abs(frame - gap["gap_start_frame"]) <= max_frames:
            return True
        if abs(frame - gap["gap_end_frame"]) <= max_frames:
            return True
    return False
