"""Output writing and notebook display helpers."""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from motive_qc.core import CSV_TABLE_EXCLUDE, LOGGER, QCMessage, QCResult, __version__
from motive_qc.output_tiers import (
    resolve_run_output_dir,
    should_write_plot,
    should_write_table,
)
from motive_qc.report import build_qc_report_markdown, write_markdown_summary_l12, write_reason_codes_file


def messages_to_dataframe(messages: list[QCMessage]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "severity": m.severity,
                "code": m.code,
                "message": m.message,
                "context": str(m.context),
                "suggested_action": m.suggested_action,
            }
            for m in messages
        ]
    )


def flatten_config(config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def _walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key.startswith("_"):
                    continue
                _walk(f"{prefix}.{key}" if prefix else key, child)
        else:
            rows.append({"key": prefix, "value": str(value)})

    _walk("", config)
    return pd.DataFrame(rows)


def _collect_tables(*results: QCResult | None) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for result in results:
        if result is None:
            continue
        for name, df in result.tables.items():
            tables[name] = df
    return tables


def _collect_figures(*results: QCResult | None) -> dict[str, Path]:
    figures: dict[str, Path] = {}
    for result in results:
        if result is None:
            continue
        figures.update(result.figures)
    return figures


def write_excel_workbook(
    path: Path,
    tables: dict[str, pd.DataFrame],
    messages: list[QCMessage],
    config: dict[str, Any],
) -> None:
    sheet_map = {
        "session_summary": "session_summary",
        "marker_inventory": "marker_inventory",
        "marker_quality_summary": "marker_quality",
        "gap_events": "gap_events",
        "gap_summary_by_marker": "gap_by_marker",
        "gap_summary_by_group": "gap_by_group",
        "unlabeled_marker_summary": "unlabeled_summary",
        "unlabeled_frame_counts": "unlabeled_frames",
        "frame_qc_mask": "frame_qc_mask",
        "frame_quality_summary": "frame_quality",
        "group_quality_summary": "group_quality",
        "window_quality_summary": "window_summary",
        "artifact_events": "artifact_events",
        "artifact_session_summary": "artifact_session",
        "artifact_candidates": "artifact_candidates",
        "artifact_summary_by_marker": "artifact_summary",
        "analysis_frame_mask": "analysis_frame_mask",
        "qc_intervals": "qc_intervals",
        "analysis_mask_summary": "analysis_mask_summary",
    }
    window_sheets = {
        k: k.replace("window_quality_", "window_")
        for k in tables
        if k.startswith("window_quality_") and k != "window_quality_summary"
    }
    sheet_map.update(window_sheets)

    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        for table_key, sheet_name in sheet_map.items():
            if table_key in tables and should_write_table(table_key, config):
                name = sheet_name[:31]
                tables[table_key].to_excel(writer, sheet_name=name, index=False)
        messages_to_dataframe(messages).to_excel(writer, sheet_name="validation_messages", index=False)
        flatten_config(config).to_excel(writer, sheet_name="config_summary", index=False)


def _write_manifest(
    path: Path,
    written: list[Path],
    tables: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> None:
    manifest = {
        "motive_qc_version": __version__,
        "run_time": datetime.now().isoformat(timespec="seconds"),
        "output_dir": config.get("_run_output_dir"),
        "tier": config.get("outputs", {}).get("tier", "essential"),
        "session_id": config.get("project", {}).get("session_id"),
        "n_files": len(written),
        "table_row_counts": {k: len(v) for k, v in tables.items() if should_write_table(k, config)},
        "files": [str(p.name) for p in written],
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_outputs(
    layer1_result: QCResult,
    layer2_result: QCResult,
    config: dict[str, Any],
    verbose: bool = False,
    layer3_result: QCResult | None = None,
    layer4_result: QCResult | None = None,
    layer5_result: QCResult | None = None,
) -> list[Path]:
    output_dir = resolve_run_output_dir(config)
    tables_dir = output_dir / "tables"
    written: list[Path] = []

    tables = _collect_tables(layer2_result, layer3_result, layer4_result, layer5_result)
    messages = layer1_result.messages + layer2_result.messages
    for result in (layer3_result, layer4_result, layer5_result):
        if result:
            messages.extend(result.messages)

    if config["outputs"].get("write_csv_tables", True):
        for name, df in tables.items():
            if name in CSV_TABLE_EXCLUDE:
                continue
            if not should_write_table(name, config):
                continue
            path = tables_dir / f"{name}.csv"
            df.to_csv(path, index=False)
            written.append(path)
            if verbose:
                LOGGER.info("Wrote %s", path)

    if config["outputs"].get("write_config_used", True):
        config_path = output_dir / "config_used.yaml"
        clean_config = copy.deepcopy(config)
        for key in ("_config_path", "_base_dir", "_run_output_dir"):
            clean_config.pop(key, None)
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(clean_config, handle, sort_keys=False)
        written.append(config_path)

    stop = int(config.get("reporting", {}).get("stop_after_layer", 2))
    if config["outputs"].get("write_text_summary", True):
        if stop >= 5 and layer5_result is not None:
            report_path = output_dir / "qc_report.md"
            report_path.write_text(
                build_qc_report_markdown(
                    layer1_result,
                    layer2_result,
                    layer3_result,
                    layer4_result,
                    tables,
                    messages,
                    config,
                ),
                encoding="utf-8",
            )
            written.append(report_path)
            reason_path = output_dir / "qc_reason_codes.md"
            write_reason_codes_file(reason_path)
            written.append(reason_path)
        else:
            summary_path = output_dir / "qc_report_summary.md"
            write_markdown_summary_l12(
                summary_path, layer1_result, layer2_result, messages, config
            )
            written.append(summary_path)

    if config["outputs"].get("write_excel_workbook", True):
        excel_path = output_dir / "qc_report.xlsx"
        write_excel_workbook(excel_path, tables, messages, config)
        written.append(excel_path)

    figures = _collect_figures(layer2_result, layer3_result, layer4_result)
    for name, fig_path in figures.items():
        if should_write_plot(name, config):
            written.append(fig_path)

    manifest_path = output_dir / "RUN_MANIFEST.json"
    _write_manifest(manifest_path, written, tables, config)
    written.append(manifest_path)

    return written


def write_validation_log(
    layer1_result: QCResult,
    layer2_result: QCResult,
    config: dict[str, Any],
    log_path: str | Path = "docs/VALIDATION_LOG.md",
    decision: str = "pending",
    validated_by: str = "",
    notes: str = "",
    layer3_decision: str = "pending",
    layer4_decision: str = "pending",
    layer5_decision: str = "pending",
) -> Path:
    base_dir = Path(config.get("_base_dir", Path(config["_config_path"]).parent))
    path = Path(log_path)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    session = layer1_result.session
    assert session is not None
    md = session.metadata
    summary = layer2_result.tables["session_summary"].iloc[0]
    run_dir = config.get("_run_output_dir", "n/a")
    content = f"""# Validation Log

## v0.5 - Layers 1-5 (refined)

Input file: {md['input_file']}
Run output: {run_dir}
Date run: {datetime.now().isoformat(timespec='seconds')}
Preprocessing status: {summary.get('raw_qc_preprocessing_status', 'n/a')}

Layer 1 decision: pending
Layer 2 decision: pending
Layer 3 decision: {layer3_decision}
Layer 4 decision: {layer4_decision}
Layer 5 decision: {layer5_decision}
Validated by: {validated_by}
Notes: {notes}
Decision: {decision}
"""
    path.write_text(content, encoding="utf-8")
    return path


def display_layer1_outputs(result: QCResult) -> dict[str, pd.DataFrame]:
    return {
        "session_summary": result.tables["session_summary"],
        "marker_inventory": result.tables["marker_inventory"],
    }


def display_layer2_outputs(result: QCResult) -> dict[str, Any]:
    top_n = 20
    gap_events = result.tables.get("gap_events", pd.DataFrame())
    longest_gaps = (
        gap_events.sort_values("duration_seconds", ascending=False).head(top_n)
        if not gap_events.empty
        else gap_events
    )
    return {
        "session_summary": result.tables["session_summary"],
        "marker_quality_summary": result.tables["marker_quality_summary"],
        "gap_events_top": longest_gaps,
        "gap_summary_by_marker": result.tables["gap_summary_by_marker"],
        "gap_summary_by_group": result.tables["gap_summary_by_group"],
        "unlabeled_marker_summary": result.tables.get("unlabeled_marker_summary"),
        "frame_qc_mask_head": result.tables.get("frame_qc_mask", pd.DataFrame()).head(20),
        "figures": result.figures,
    }
