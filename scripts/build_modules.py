#!/usr/bin/env python3
"""Generate motive_qc modules from motive_raw_qc.py (one-time refactor)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "motive_raw_qc.py"
lines = SRC.read_text(encoding="utf-8").splitlines()

REPLACEMENTS = [
    ("_resolve_path", "resolve_path"),
    ("_read_csv_header", "read_csv_header"),
    ("_find_header_rows", "find_header_rows"),
    ("_build_marker_columns", "build_marker_columns"),
    ("_filter_markers", "filter_markers"),
    ("_metadata_float", "metadata_float"),
    ("_metadata_int", "metadata_int"),
    ("_build_layer1_session_summary", "build_layer1_session_summary"),
    ("_seconds_to_key", "seconds_to_key"),
    ("_gap_threshold_labels", "gap_threshold_labels"),
    ("_crossed_thresholds", "crossed_thresholds"),
    ("_severity_label", "severity_label"),
    ("_recommended_status", "recommended_status"),
    ("_count_gaps_ge", "count_gaps_ge"),
    ("_marker_quality_label", "marker_quality_label"),
    ("_detect_gaps_for_marker", "detect_gaps_for_marker"),
    ("_build_gap_summary_by_group", "build_gap_summary_by_group"),
    ("_session_gap_timeline_metrics", "session_gap_timeline_metrics"),
    ("_evaluate_preprocessing_status", "evaluate_preprocessing_status"),
    ("_update_session_summary_layer2", "update_session_summary_layer2"),
    ("_frame_qc_mask_config", "frame_qc_mask_config"),
    ("_plot_", "plot_"),
    ("_messages_to_dataframe", "messages_to_dataframe"),
    ("_flatten_config", "flatten_config"),
    ("_critical_region_large_gaps", "critical_region_large_gaps"),
    ("_write_markdown_summary", "write_markdown_summary"),
    ("_write_excel_workbook", "write_excel_workbook"),
]


def extract(start_line: int, end_line: int) -> str:
    return "\n".join(lines[start_line - 1 : end_line])


def apply_replacements(text: str) -> str:
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    return text


def write(name: str, header: str, start: int, end: int) -> None:
    body = apply_replacements(extract(start, end))
    path = ROOT / "motive_qc" / name
    path.write_text(header + "\n" + body + "\n", encoding="utf-8")
    print(f"Wrote {path} ({end - start + 1} lines)")


PARSE_HEADER = '''"""Layer 1: parse Motive raw CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from motive_qc.core import (
    LOGGER,
    MotiveCSVParseError,
    MotiveSession,
    QCMessage,
    QCResult,
    QCValidationError,
    SchemaValidationError,
    resolve_path,
)
from motive_qc.marker_meta import (
    build_marker_columns,
    filter_markers,
    find_header_rows,
    metadata_float,
    metadata_int,
    read_csv_header,
)
'''

GAPS_HEADER = '''"""Layer 2: gaps, masks, unlabeled burden."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from motive_qc.core import LOGGER, MotiveSession, QCMessage, QCResult
from motive_qc.parse import build_layer1_session_summary
from motive_qc.plots import generate_layer2_plots
'''

PLOTS_HEADER = '''"""Plotting helpers for Layers 2-5."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from motive_qc.core import MotiveSession, plot_dir_from_config, resolve_path
'''

IO_HEADER = '''"""Output writing and notebook display helpers."""

from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from motive_qc.core import CSV_TABLE_EXCLUDE, LOGGER, QCMessage, QCResult, resolve_path
from motive_qc.report import write_markdown_summary_l12
'''

if __name__ == "__main__":
    write("parse.py", PARSE_HEADER, 517, 865)
    write("gaps.py", GAPS_HEADER, 873, 1582)
    write("plots.py", PLOTS_HEADER, 1590, 1857)
    # io will be hand-written due to circular imports
    print("Done (plots/gaps/parse). io.py manual.")
