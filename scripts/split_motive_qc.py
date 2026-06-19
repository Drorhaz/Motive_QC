#!/usr/bin/env python3
"""One-time helper to verify package imports after refactor."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import motive_qc  # noqa: E402
from motive_qc import load_config, run_layers_1_2  # noqa: E402

if __name__ == "__main__":
  config = load_config(ROOT / "config.yaml")
  config["_base_dir"] = ROOT
  config["reporting"]["stop_after_layer"] = 2
  l1, l2, files = run_layers_1_2(config, verbose=True)
  print(f"OK: layer1={l1.status} layer2={l2.status} files={len(files)}")
