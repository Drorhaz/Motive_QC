"""Layer 4 spectral screening: noise description and smoothing-suspicion flags."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from motive_qc.core import LOGGER, MotiveSession, QCMessage, QCResult
from motive_qc.segments import compute_speeds, marker_valid_segments, segment_positions


def _spectral_config(config: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "enabled": False,
        "require_known_units": True,
        "min_segment_frames": 60,
        "hf_band_hz": [15.0, 60.0],
        "rolloff_percentile": 95,
        "movement_band_hz": [0.5, 12.0],
        "smoothing_suspicion": {
            "hf_power_ratio_threshold": 0.02,
            "peer_zscore_threshold": 2.5,
        },
    }
    user = config.get("spectral_screen", {})
    merged = {**defaults, **user}
    if "smoothing_suspicion" in user:
        merged["smoothing_suspicion"] = {
            **defaults["smoothing_suspicion"],
            **user["smoothing_suspicion"],
        }
    return merged


def _welch_psd(signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(signal)
    if n < 8:
        return np.array([]), np.array([])
    nperseg = min(256, max(8, n // 4))
    window = np.hanning(nperseg)
    step = nperseg // 2
    accum = None
    count = 0
    for start in range(0, n - nperseg + 1, step):
        seg = signal[start : start + nperseg] - np.mean(signal[start : start + nperseg])
        fft = np.fft.rfft(seg * window)
        psd = (np.abs(fft) ** 2) / (fs * np.sum(window**2))
        if accum is None:
            accum = psd
        else:
            accum += psd
        count += 1
    if accum is None or count == 0:
        return np.array([]), np.array([])
    psd = accum / count
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    return freqs, psd


def _band_power(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> float:
    if freqs.size == 0:
        return 0.0
    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    if not mask.any():
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


def _spectral_rolloff(freqs: np.ndarray, psd: np.ndarray, percentile: float) -> float:
    if freqs.size == 0:
        return 0.0
    cumulative = np.cumsum(psd)
    total = cumulative[-1]
    if total <= 0:
        return 0.0
    target = total * (percentile / 100.0)
    idx = int(np.searchsorted(cumulative, target))
    idx = min(idx, len(freqs) - 1)
    return float(freqs[idx])


def _marker_spectral_metrics(
    session: MotiveSession,
    marker_name: str,
    spec_cfg: dict[str, Any],
) -> dict[str, Any] | None:
    fs = float(session.metadata["effective_frame_rate_hz"])
    min_frames = int(spec_cfg["min_segment_frames"])
    hf_band = tuple(spec_cfg["hf_band_hz"])
    mov_band = tuple(spec_cfg.get("movement_band_hz", [0.5, 12.0]))
    rolloff_pct = float(spec_cfg["rolloff_percentile"])

    segment_psds: list[tuple[np.ndarray, np.ndarray]] = []
    for start_idx, end_idx in marker_valid_segments(session, marker_name):
        if end_idx - start_idx + 1 < min_frames:
            continue
        pos = segment_positions(session, marker_name, start_idx, end_idx)
        dt = 1.0 / fs
        speeds = compute_speeds(pos, dt)
        if speeds.size < min_frames - 1:
            continue
        freqs, psd = _welch_psd(speeds, fs)
        if freqs.size:
            segment_psds.append((freqs, psd))

    if not segment_psds:
        return None

    ref_freqs = segment_psds[0][0]
    stacked = np.vstack([np.interp(ref_freqs, f, p) for f, p in segment_psds])
    median_psd = np.median(stacked, axis=0)

    mov_power = _band_power(ref_freqs, median_psd, mov_band)
    hf_power = _band_power(ref_freqs, median_psd, hf_band)
    total_power = _band_power(ref_freqs, median_psd, (ref_freqs[1], ref_freqs[-1]))
    hf_ratio = hf_power / mov_power if mov_power > 0 else 0.0
    rolloff = _spectral_rolloff(ref_freqs, median_psd, rolloff_pct)
    hf_mask = (ref_freqs >= hf_band[0]) & (ref_freqs <= hf_band[1])
    noise_floor_db = (
        10.0 * np.log10(np.median(median_psd[hf_mask]) + 1e-12) if hf_mask.any() else 0.0
    )

    return {
        "marker_name": marker_name,
        "n_segments_used": len(segment_psds),
        "hf_power_ratio": round(hf_ratio, 6),
        "spectral_rolloff_hz": round(rolloff, 6),
        "noise_floor_db": round(noise_floor_db, 6),
        "movement_band_power": round(mov_power, 6),
        "total_power": round(total_power, 6),
        "freqs": ref_freqs,
        "psd": median_psd,
    }


def run_spectral_screen(
    session: MotiveSession,
    config: dict[str, Any],
    verbose: bool = False,
) -> QCResult:
    spec_cfg = _spectral_config(config)
    if not spec_cfg.get("enabled", False):
        return QCResult(layer_name="spectral", status="skipped", session=session)

    if spec_cfg.get("require_known_units", True) and not session.metadata.get("length_units"):
        return QCResult(
            layer_name="spectral",
            status="skipped",
            messages=[
                QCMessage(
                    "WARNING",
                    "SPECTRAL_UNITS_MISSING",
                    "Spectral screening skipped because length units are unknown.",
                )
            ],
            session=session,
        )

    if verbose:
        LOGGER.info("Running spectral screening")

    inventory = session.marker_inventory
    labeled = inventory.loc[inventory["is_labeled"], "marker_name"].tolist()
    susp_cfg = spec_cfg["smoothing_suspicion"]
    hf_thr = float(susp_cfg["hf_power_ratio_threshold"])
    z_thr = float(susp_cfg["peer_zscore_threshold"])

    rows: list[dict[str, Any]] = []
    psd_curves: list[tuple[np.ndarray, np.ndarray]] = []
    hf_ratios: list[float] = []

    for marker in labeled:
        metrics = _marker_spectral_metrics(session, marker, spec_cfg)
        if metrics is None:
            continue
        hf_ratios.append(metrics["hf_power_ratio"])
        psd_curves.append((metrics.pop("freqs"), metrics.pop("psd")))
        rows.append({k: v for k, v in metrics.items()})

    if not rows:
        return QCResult(layer_name="spectral", status="pass", session=session)

    summary_df = pd.DataFrame(rows)
    median_hf = float(np.median(summary_df["hf_power_ratio"]))
    mad_hf = float(np.median(np.abs(summary_df["hf_power_ratio"] - median_hf)))
    if mad_hf == 0:
        mad_hf = float(summary_df["hf_power_ratio"].std()) or 1e-9

    flags: list[bool] = []
    for _, row in summary_df.iterrows():
        z = abs(row["hf_power_ratio"] - median_hf) / mad_hf
        low_hf = row["hf_power_ratio"] < hf_thr
        peer_outlier = z > z_thr and row["hf_power_ratio"] < median_hf
        flags.append(bool(low_hf or peer_outlier))
    summary_df["smoothing_suspicion_flag"] = flags

    session_flag = bool(median_hf < hf_thr or summary_df["smoothing_suspicion_flag"].sum() >= 3)
    narrative_codes: list[str] = []
    if session_flag:
        narrative_codes.append("possible_prior_smoothing")
    if (summary_df["hf_power_ratio"] > median_hf + z_thr * mad_hf).any():
        narrative_codes.append("elevated_hf_noise_markers")

    export_notes = config.get("project", {}).get("notes", "").lower()
    metadata_mismatch = session_flag and "raw" in export_notes
    if metadata_mismatch:
        narrative_codes.append("export_metadata_mismatch")

    session_spectral = pd.DataFrame(
        [
            {
                "median_hf_power_ratio": round(median_hf, 6),
                "n_markers_analyzed": len(summary_df),
                "n_smoothing_suspicion_flags": int(summary_df["smoothing_suspicion_flag"].sum()),
                "session_smoothing_suspicion": session_flag,
                "export_metadata_mismatch": metadata_mismatch,
                "narrative_codes": ";".join(narrative_codes),
            }
        ]
    )

    figures: dict[str, Path] = {}
    if psd_curves and config.get("outputs", {}).get("plots", {}).get("session_psd_summary", True):
        figures["session_psd_summary"] = _plot_session_psd(psd_curves, spec_cfg, config)

    return QCResult(
        layer_name="spectral",
        status="pass",
        tables={
            "spectral_summary_by_marker": summary_df,
            "session_spectral_summary": session_spectral,
        },
        figures=figures,
        session=session,
    )


def _plot_session_psd(
    psd_curves: list[tuple[np.ndarray, np.ndarray]],
    spec_cfg: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    from motive_qc.core import plot_dir_from_config

    freqs = psd_curves[0][0]
    stacked = np.vstack([np.interp(freqs, f, p) for f, p in psd_curves])
    median = np.median(stacked, axis=0)
    q25 = np.percentile(stacked, 25, axis=0)
    q75 = np.percentile(stacked, 75, axis=0)
    hf_band = spec_cfg["hf_band_hz"]

    plot_dir = plot_dir_from_config(config)
    fmt = config.get("outputs", {}).get("plot_format", "png")
    dpi = config.get("outputs", {}).get("dpi", 300)
    path = plot_dir / f"session_psd_summary.{fmt}"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(freqs, q25, q75, alpha=0.3, color="#2b6cb0", label="IQR across markers")
    ax.plot(freqs, median, color="#2b6cb0", linewidth=1.5, label="Median PSD")
    ax.axvline(hf_band[0], color="#e53e3e", linestyle="--", linewidth=1, label="HF band")
    ax.axvline(hf_band[1], color="#e53e3e", linestyle="--", linewidth=1)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (speed)")
    ax.set_title("Session speed PSD summary (labeled markers, gap-safe segments)")
    ax.set_xlim(0, min(freqs[-1], hf_band[1] * 1.5))
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
