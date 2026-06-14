#!/usr/bin/env python3
"""Motive QC Layer 6 — cross-session batch processing and executive EDA report."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from motive_qc import discover_sessions, load_config, run_batch
from motive_qc.core import __version__
from motive_qc.discovery import data_root_from_config

LOGGER = logging.getLogger("motive_batch_qc")


def _configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Motive QC Layer 6 — batch cross-session QC and executive EDA report"
    )
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--discover", action="store_true", help="List discoverable sessions and exit")
    parser.add_argument("--subject", action="append", help="Subject folder(s) to process (e.g. 671)")
    parser.add_argument("--all-subjects", action="store_true", help="Process all subject folders under data/")
    parser.add_argument(
        "--sessions",
        help="Comma-separated session_id filter (e.g. T2_P1_R1,T3_P1_R2)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print progress messages")
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    config["_base_dir"] = config_path.parent

    if args.discover:
        if args.subject:
            discover_subject_ids = args.subject
        elif args.all_subjects:
            discover_subject_ids = None
        else:
            discover_subject_ids = None  # discover scans all subjects by default
    elif args.all_subjects:
        subject_ids = None
    elif args.subject:
        subject_ids = args.subject
    else:
        subject_ids = [config["project"]["subject_id"]]

    session_filter = None
    if args.sessions:
        session_filter = [s.strip() for s in args.sessions.split(",") if s.strip()]

    if args.discover:
        catalog = discover_sessions(config, subject_ids=discover_subject_ids, session_filter=session_filter)
        data_root = data_root_from_config(config)
        print(f"Motive_QC {__version__} — discovery under {data_root}")
        print(f"Found {len(catalog)} session(s):\n")
        if catalog.empty:
            return 1
        print(catalog.to_string(index=False))
        return 0

    try:
        result = run_batch(
            config,
            subject_ids=subject_ids,
            session_filter=session_filter,
            verbose=args.verbose,
        )
        print(f"\nBatch complete: {result.batch_dir}")
        print(f"Executive report: {result.report_paths.get('md')}")
        print(f"PI workbook: {result.report_paths.get('workbook')}")
        print(f"Sessions: {len(result.eda_df)} total, {len(result.failures_df)} failure(s)")
        return 1 if not result.failures_df.empty else 0
    except Exception as exc:
        LOGGER.error("%s: %s", type(exc).__name__, exc)
        if args.verbose:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
