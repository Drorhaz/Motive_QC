#!/usr/bin/env python3
"""Motive raw marker QC pipeline — CLI entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from motive_qc import load_config, run_full_pipeline, run_layers_1_2
from motive_qc.core import __version__
from motive_qc.core import QCValidationError, resolve_path
from motive_qc.marker_meta import (
    build_marker_columns,
    find_header_rows,
    read_csv_header,
)
from motive_qc.core import QCMessage

# Backward-compatible re-exports for notebooks
from motive_qc import (  # noqa: F401
    MotiveSession,
    QCResult,
    display_layer1_outputs,
    display_layer2_outputs,
    run_layer1_parse,
    run_layer2_gaps,
    run_layer3_windows,
    run_layer4_artifacts,
    run_layer5_report,
    run_spectral_screen,
    collect_session_velocity_distribution,
    flagged_velocity_speeds,
    list_velocity_histogram_groups,
    write_outputs,
    write_validation_log,
)

LOGGER = logging.getLogger("motive_raw_qc")


def _configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Motive raw marker QC (Layers 1-5)")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and parse metadata only")
    parser.add_argument("--verbose", action="store_true", help="Print progress messages")
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    config["_base_dir"] = config_path.parent

    try:
        if args.dry_run:
            input_path = resolve_path(config_path.parent, config["paths"]["input_csv"])
            if not input_path.exists():
                raise FileNotFoundError(f"Input CSV not found: {input_path}")
            metadata_row, header_rows, _ = read_csv_header(input_path)
            header_map = find_header_rows(header_rows)
            messages: list[QCMessage] = []
            marker_records, _, _ = build_marker_columns(header_rows, header_map, config, messages)
            print(f"Dry run OK: {input_path.name}")
            print(f"  Markers found: {len(marker_records)}")
            print(f"  Capture rate: {metadata_row.get('Capture Frame Rate')}")
            print(f"  Export rate: {metadata_row.get('Export Frame Rate')}")
            return 0

        stop = int(config.get("reporting", {}).get("stop_after_layer", 2))
        if stop <= 2:
            layer1, layer2, files = run_layers_1_2(config, verbose=args.verbose)
            final_status = layer2.status
        else:
            *_, layer5, files = run_full_pipeline(config, verbose=args.verbose)
            final_status = layer5.status if layer5 else "pass"

        if args.verbose:
            run_dir = config.get("_run_output_dir", config["paths"]["output_dir"])
            print(f"Wrote {len(files)} output files to {run_dir}")
        return 0 if final_status != "fail" else 1
    except Exception as exc:
        LOGGER.error("%s: %s", type(exc).__name__, exc)
        if args.verbose:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
