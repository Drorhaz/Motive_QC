"""Layer 6: cross-session batch orchestration."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from motive_qc.batch_metrics import extract_session_details, extract_session_metrics_row
from motive_qc.batch_report import write_batch_reports
from motive_qc.core import LOGGER, QCValidationError, __version__
from motive_qc.discovery import (
    apply_session_to_config,
    data_root_from_config,
    discover_sessions,
    validate_csv_header,
)
from motive_qc.pipeline import run_full_pipeline


@dataclass
class SessionBatchResult:
    session_row: dict[str, Any]
    batch_status: str
    run_output_dir: str = ""
    error_message: str = ""
    elapsed_seconds: float = 0.0
    layer1: Any = None
    layer2: Any = None
    layer3: Any = None
    layer4: Any = None
    layer5: Any = None
    metrics_row: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    batch_dir: Path
    eda_df: pd.DataFrame
    failures_df: pd.DataFrame
    session_results: list[SessionBatchResult]
    report_paths: dict[str, Path] = field(default_factory=dict)


def resolve_batch_output_dir(config: dict[str, Any]) -> Path:
    base_dir = Path(config.get("_base_dir", Path(config["_config_path"]).parent))
    root = config.get("paths", {}).get("batch_output_dir", "outputs/batch_runs")
    from motive_qc.core import resolve_path

    batch_root = resolve_path(base_dir, root)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = batch_root / f"batch_{ts}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    return batch_dir


def _progress_iter(items: list, config: dict[str, Any], desc: str):
    batch_cfg = config.get("batch", {})
    if batch_cfg.get("progress_bar", True):
        try:
            from tqdm import tqdm

            return tqdm(items, desc=desc)
        except ImportError:
            pass
    return items


def run_batch(
    config: dict[str, Any],
    *,
    subject_ids: list[str] | None = None,
    session_filter: list[str] | None = None,
    catalog_rows: list[dict[str, Any]] | None = None,
    verbose: bool = True,
) -> BatchResult:
    """Run L1-L5 for each discovered session and build Layer 6 executive report."""
    base_config = copy.deepcopy(config)
    if catalog_rows is not None:
        catalog = pd.DataFrame(catalog_rows)
    else:
        catalog = discover_sessions(base_config, subject_ids=subject_ids, session_filter=session_filter)

    if catalog.empty:
        data_root = data_root_from_config(base_config)
        raise FileNotFoundError(f"No CSV sessions found under {data_root}")

    batch_dir = resolve_batch_output_dir(base_config)
    continue_on_error = base_config.get("batch", {}).get("continue_on_error", True)

    session_results: list[SessionBatchResult] = []
    failure_rows: list[dict[str, Any]] = []
    eda_rows: list[dict[str, Any]] = []
    all_top: list[pd.DataFrame] = []
    all_art: list[pd.DataFrame] = []
    all_vel: list[pd.DataFrame] = []
    pointers: list[dict[str, Any]] = []

    rows = catalog.to_dict("records")
    iterator = _progress_iter(rows, base_config, "Batch QC")

    for i, row in enumerate(iterator, start=1):
        subj = row["subject_id"]
        sess = row["session_id"]
        file_name = row["file_name"]
        csv_path = Path(row["csv_path"])

        if verbose:
            LOGGER.info("[%d/%d] %s %s — validating header", i, len(rows), subj, sess)

        ok, err = validate_csv_header(csv_path)
        if not ok:
            failure_rows.append({**row, "error_message": err, "stage": "header_validation"})
            session_results.append(
                SessionBatchResult(session_row=row, batch_status="failed", error_message=err)
            )
            if not continue_on_error:
                break
            continue

        session_config = apply_session_to_config(base_config, row)
        t0 = time.perf_counter()
        try:
            if verbose:
                LOGGER.info("[%d/%d] %s %s — running L1-L5", i, len(rows), subj, sess)
            layer1, layer2, layer3, layer4, layer5, _files = run_full_pipeline(
                session_config, verbose=verbose
            )
            elapsed = time.perf_counter() - t0
            run_dir = session_config.get("_run_output_dir", "")

            metrics = extract_session_metrics_row(
                row,
                layer1,
                layer2,
                layer3,
                layer4,
                layer5,
                session_config,
                batch_status="ok",
                run_output_dir=run_dir,
            )
            details = extract_session_details(row, layer1, layer2, layer3, layer4, session_config)

            eda_rows.append(metrics)
            all_top.append(details["top_markers"])
            all_art.append(details["artifact_types"])
            all_vel.append(details["velocity_by_segment"])

            pointers.append(
                {
                    "subject_id": subj,
                    "session_id": sess,
                    "file_name": file_name,
                    "run_output_dir": run_dir,
                    "batch_status": "ok",
                    "key_metrics": {
                        k: metrics[k]
                        for k in (
                            "raw_qc_preprocessing_status",
                            "missing_percent_labeled",
                            "n_artifact_events",
                            "window_yield_0p5s_pct",
                        )
                        if k in metrics
                    },
                }
            )

            session_results.append(
                SessionBatchResult(
                    session_row=row,
                    batch_status="ok",
                    run_output_dir=run_dir,
                    elapsed_seconds=elapsed,
                    layer1=layer1,
                    layer2=layer2,
                    layer3=layer3,
                    layer4=layer4,
                    layer5=layer5,
                    metrics_row=metrics,
                )
            )
            if verbose:
                LOGGER.info(
                    "[%d/%d] %s %s — ok (%.1fs) → %s",
                    i,
                    len(rows),
                    subj,
                    sess,
                    elapsed,
                    run_dir,
                )

        except (QCValidationError, Exception) as exc:
            elapsed = time.perf_counter() - t0
            msg = f"{type(exc).__name__}: {exc}"
            LOGGER.error("[%d/%d] %s %s — FAILED: %s", i, len(rows), subj, sess, msg)
            failure_rows.append({**row, "error_message": msg, "stage": "pipeline"})
            fail_metrics = {
                "subject_id": subj,
                "session_id": sess,
                "file_name": file_name,
                "batch_status": "failed",
                "error_message": msg,
                "run_output_dir": "",
            }
            eda_rows.append(fail_metrics)
            session_results.append(
                SessionBatchResult(
                    session_row=row,
                    batch_status="failed",
                    error_message=msg,
                    elapsed_seconds=elapsed,
                )
            )
            if not continue_on_error:
                break

    eda_df = pd.DataFrame(eda_rows)
    failures_df = pd.DataFrame(failure_rows)
    top_markers = pd.concat(all_top, ignore_index=True) if all_top else pd.DataFrame()
    artifact_types = pd.concat(all_art, ignore_index=True) if all_art else pd.DataFrame()
    velocity_df = pd.concat(all_vel, ignore_index=True) if all_vel else pd.DataFrame()

    report_paths = write_batch_reports(
        batch_dir,
        eda_df,
        top_markers,
        artifact_types,
        velocity_df,
        failures_df,
        base_config,
        pointers,
        session_results=session_results,
    )

    if verbose:
        LOGGER.info("Batch complete → %s", batch_dir)
        LOGGER.info("Executive report: %s", report_paths.get("md"))

    return BatchResult(
        batch_dir=batch_dir,
        eda_df=eda_df,
        failures_df=failures_df,
        session_results=session_results,
        report_paths=report_paths,
    )
