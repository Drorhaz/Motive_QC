# Motive_QC

**Version:** 0.6.0

Reproducible quality-control pipeline for **raw OptiTrack Motive marker XYZ CSV exports** — before gap filling, smoothing, skeleton solving, or BVH export. The pipeline parses capture files, quantifies missingness and gaps, screens kinematic artifact **candidates**, judges fixed-duration analysis windows for PCA/jPCA planning, and writes shareable QC reports.

**In scope:** raw marker XYZ QC, gap structure, frame/window warnings, artifact event screening, analysis masks and intervals.

**Out of scope:** modifying the CSV, gap filling, smoothing/filtering, BVH parsing, automatic frame deletion, PCA/jPCA execution.

Full specification: [`docs/PROJECT_SPEC_MOTIVE_QC.md`](docs/PROJECT_SPEC_MOTIVE_QC.md)

---

## Quick start

### Requirements

- Python 3.10+
- Dependencies in [`requirements.txt`](requirements.txt): pandas, numpy, xarray, PyYAML, matplotlib, openpyxl, xlsxwriter, ipywidgets, tqdm

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### Run one session (CLI)

1. Place Motive CSV under `data/` (e.g. `data/671/671_T2_P1_R1_Take ...csv`).
2. Edit [`config.yaml`](config.yaml): set `paths.input_csv`, `project.subject_id`, `project.session_id`.
3. Run:

```bash
python motive_raw_qc.py --config config.yaml --verbose
```

Dry-run (parse header only):

```bash
python motive_raw_qc.py --config config.yaml --dry-run
```

Outputs land in `outputs/runs/{session_id}_{YYYYMMDD_HHMMSS}/` (gitignored).

### Run batch across sessions (Layer 6)

Discover CSVs under `data/{subject_id}/`:

```bash
python motive_batch_qc.py --config config.yaml --discover
```

Batch one subject, all subjects, or explicit sessions:

```bash
python motive_batch_qc.py --config config.yaml --subject 671 --verbose
python motive_batch_qc.py --config config.yaml --all-subjects --verbose
python motive_batch_qc.py --config config.yaml --subject 671 --sessions T2_P1_R1,T3_P1_R2 --verbose
```

Layer 6 runs L1–L5 sequentially per CSV and writes an executive package under `outputs/batch_runs/batch_{YYYYMMDD_HHMMSS}/`:

| Deliverable | Purpose |
|-------------|---------|
| `dataset_eda_report.md` | PI-facing narrative summary |
| `dataset_eda_report.csv` / `.xlsx` | Cross-session comparison table (flat sheets) |
| `dataset_eda_workbook.xlsx` | **PI workbook**: structured Executive Summary + one tab per session (gap windows & artifact intervals in seconds) |
| `plots/batch_*.png` | Preprocessing, window yield, artifacts, missingness |
| `details/*.csv` | Top markers, artifact types, velocity by body segment |
| `sessions/*.json` | Pointers to per-session `outputs/runs/` folders |

Per-session full QC outputs remain in `outputs/runs/` (unchanged). Exit code `1` if any session failed (`failures.csv`).

### Interactive QC (notebooks)

| Notebook | Purpose |
|----------|---------|
| [`notebooks/01_raw_csv_qc_layers_1_2.ipynb`](notebooks/01_raw_csv_qc_layers_1_2.ipynb) | Layers 1–2: parse, gaps, frame mask, validation sign-off |
| [`notebooks/02_raw_csv_qc_layers_3_5.ipynb`](notebooks/02_raw_csv_qc_layers_3_5.ipynb) | Full L1–L5: artifacts, windows, report, tuning widgets, per-segment velocity histograms, **Layer 6 batch picker** |

Open notebook 02 from the project root so `config.yaml` resolves correctly.

---

## Pipeline architecture

**Execution order (v0.5+):** L1 → L2 → **L4** → **L3** → L5

Layer 6 (v0.6) orchestrates L1–L5 across many sessions and aggregates executive EDA — it does not re-run analysis logic.

```
CSV file(s) under data/{subject_id}/
   │
   ▼
Layer 1  Parse + marker inventory + session metadata
   │
   ▼
Layer 2  Gaps, missingness, frame QC mask, session preprocessing status
   │
   ▼
Layer 4  Kinematic artifact candidates → clustered events (labeled markers only)
   │
   ▼
Layer 3  Fixed windows (0.5 s, 1.0 s) — safe for PCA/jPCA?
   │
   ▼
Layer 5  qc_report.md, qc_intervals, analysis_frame_mask, Excel workbook
   │
   └─► Layer 6  Cross-session batch aggregator → dataset_eda_report (PI package)
```

| Layer | Module | Role |
|-------|--------|------|
| L1 | `motive_qc/parse.py` | Read Motive CSV; build `MotiveSession` (xarray coordinates) |
| L2 | `motive_qc/gaps.py` | Gap events, marker/session summaries, `frame_qc_mask` |
| L4 | `motive_qc/artifacts.py` | Gap-safe velocity/spike/hold screening → `artifact_events` |
| L3 | `motive_qc/windows.py` | Window bins + combined L2+L4 flags |
| L5 | `motive_qc/report.py` | Merged analysis mask, labeled-only intervals, markdown report |
| L6 | `motive_qc/batch.py` | Cross-session orchestration + executive EDA report |

Orchestration: [`motive_qc/pipeline.py`](motive_qc/pipeline.py)  
CLI entry: [`motive_raw_qc.py`](motive_raw_qc.py) (single session), [`motive_batch_qc.py`](motive_batch_qc.py) (batch)

---

## Repository layout

```
Motive_QC/
├── config.yaml              # Main tunable settings
├── motive_raw_qc.py         # Single-session CLI
├── motive_batch_qc.py       # Layer 6 batch CLI
├── motive_qc/               # Python package
│   ├── parse.py             # Layer 1
│   ├── gaps.py              # Layer 2
│   ├── artifacts.py         # Layer 4
│   ├── windows.py           # Layer 3
│   ├── report.py            # Layer 5
│   ├── discovery.py         # Layer 6 session discovery
│   ├── batch.py             # Layer 6 batch runner
│   ├── batch_metrics.py     # Layer 6 per-session EDA extraction
│   ├── batch_report.py      # Layer 6 PI report + plots
│   ├── batch_workbook.py    # Layer 6 PI Excel workbook
│   ├── plots.py             # Figures
│   ├── io.py                # Write CSV/Excel/report/manifest
│   ├── output_tiers.py      # essential vs full outputs
│   ├── reason_codes.py      # Interval reason glossary
│   ├── segments.py          # Gap-safe kinematics helpers
│   └── spectral.py          # Optional PSD screen (disabled by default)
├── notebooks/               # Interactive validation UI
├── data/                    # Input CSVs (by subject: 671/, 252/, …)
├── outputs/runs/            # Per-run outputs (gitignored)
├── outputs/batch_runs/      # Layer 6 batch packages (gitignored)
└── docs/                    # Spec, templates, validation log
```

---

## Configuration reference (`config.yaml`)

All scientific thresholds live in config — not hard-coded. Paths are relative to the config file directory unless absolute.

### `project`

| Key | Description |
|-----|-------------|
| `subject_id` | Participant ID (e.g. `671`) |
| `session_id` | Session label (e.g. `T2_P1_R1`) — used in output folder names |
| `motive_version` | Documentation string |
| `analysis_stage` | e.g. `raw_csv_before_preprocessing` |
| `notes` | Free text stored in reports |

### `paths`

| Key | Default | Description |
|-----|---------|-------------|
| `input_csv` | — | Path to Motive marker XYZ CSV |
| `output_dir` | `outputs/runs` | Root for run folders |
| `data_root` | `data` | Root for subject folders (`data/{subject_id}/*.csv`) |
| `batch_output_dir` | `outputs/batch_runs` | Root for Layer 6 batch packages |
| `exclude_globs` | `archive/**`, etc. | Paths excluded from discovery |
| `include_globs` | `{subject_id}/*.csv` | Per-subject CSV glob template |
| `use_timestamp_subfolder` | `true` | If true: `{output_dir}/{session_id}_{YYYYMMDD_HHMMSS}/` |

### `batch` (Layer 6)

| Key | Default | Description |
|-----|---------|-------------|
| `continue_on_error` | `true` | Log failure and continue remaining sessions |
| `sort_by` | `subject_id`, `session_id` | Catalog sort order |
| `progress_bar` | `true` | Use `tqdm` if installed |

### `time`

| Key | Description |
|-----|-------------|
| `infer_frame_rate_from_file` | Use Capture/Export Frame Rate from CSV header |
| `frame_rate_hz_override` | Force frame rate (null = infer) |
| `require_capture_export_rate_match` | Fail if capture ≠ export rate |
| `allow_time_column_tolerance_seconds` | Tolerance for Time column vs frame index |

### `parsing`

Controls CSV validation strictness: XYZ triplets, duplicate markers/frames, monotonic frames, blank → NaN, etc.

### `markers`

| Key | Description |
|-----|-------------|
| `include_labeled_markers` / `include_unlabeled_markers` | Which tracks to load |
| `include_unlabeled_in_session_missing_percent` | Whether unlabeled gaps count in session missing % |
| `exclude_body_groups_from_analysis` | e.g. `[fingers]` — excluded from gaps, artifacts, windows, masks, and batch EDA |
| `exclude_markers` / `include_only_markers` | Optional filters |

### `marker_groups`

Keyword lists mapping marker names to body regions (`head_neck`, `pelvis_waist`, `fingers`, …). Used in gap summaries, window flags, and interval body groups.

### `gaps`

| Key | Description |
|-----|-------------|
| `thresholds_seconds` | Named durations: `moderate_gap` (0.2 s), `large_gap` (0.5 s), `severe_gap` (1.0 s), … |
| `report_all_gaps` | Emit every continuous missing run |
| `primary_report_thresholds_seconds` | Thresholds highlighted in reports |

### `quality_labels`

Rule-based **marker** labels (`clean`, `minor_issue`, `caution`, `poor`) and **session** preprocessing status (`acceptable_for_preprocessing`, `caution_for_preprocessing`, `poor_if`). Drives `raw_qc_preprocessing_status` in `session_summary.csv`.

### `frame_qc_mask` (Layer 2)

Per-frame `use` / `caution` / `exclude_or_review` from missingness and gap overlap at frame level.

### `frame_quality` (Layer 3 input)

Per-frame missing % and `critical_groups` list for torso/pelvis/head/legs sensitivity.

### `windows` (Layer 3)

| Key | Default | Description |
|-----|---------|-------------|
| `window_lengths_seconds` | `[0.5, 1.0]` | Analysis window sizes |
| `use_non_overlapping_windows` | `true` | Non-overlapping bins along timeline |
| `flag_if_gap_at_least_seconds` | `0.2` | Gap overlap → caution |
| `flag_if_large_gap_at_least_seconds` | `0.5` | Large gap overlap → exclude/review |
| `flag_if_missing_marker_percent_above` | `10.0` | High missing % in window |

**Window labels:** `use` | `caution` | `exclude_or_review`

### `artifacts` (Layer 4)

Tuned for **expressive movement** (e.g. Gaga) — flags extreme outliers, not normal fast motion.

| Key | Default | Description |
|-----|---------|-------------|
| `require_known_units` | `true` | Skip kinematics if length units unknown |
| `methods.velocity_mad` | `true` | Robust velocity peak detection |
| `methods.acceleration_mad` | `false` | Acceleration peaks (off for dance) |
| `methods.single_frame_spike` | `true` | Out-and-back jump detector |
| `methods.constant_position_hold` | `false` | Frozen XYZ hold detector |
| `velocity_mad_multiplier` | `11.0` | MAD σ — **higher = looser** (fewer flags) |
| `velocity_percentile_threshold` | `99.97` | Percentile floor — **higher = looser** |
| `single_frame_spike.min_jump_distance_m` | `0.10` | Minimum spike jump (meters) |
| `constant_position_hold.min_repeated_frames` | `5` | Frames at identical position |
| `minimum_valid_neighbors` | `2` | Min valid frames each side for kinematics |
| `max_frames_after_gap_for_velocity` | `0` | Exclude speeds near gap boundaries |

**Threshold rule (velocity):** per marker, `threshold = max(median + σ×MAD, percentile)`; flag only **local speed peaks** above threshold on **gap-safe segments**.

### `spectral_screen`

Optional PSD-based smoothing-suspicion screen. **`enabled: false`** in v0.5 (module retained for compatibility).

### `outputs`

| Key | Default | Description |
|-----|---------|-------------|
| `tier` | `essential` | `essential` or `full` (see Outputs below) |
| `write_frame_level_artifacts` | `false` | Write `artifact_candidates.csv` (frame-level) |
| `write_csv_tables` / `write_excel_workbook` / `write_text_summary` | `true` | Output toggles |
| `plots.*` | — | Per-plot enable flags |
| `plot_format` | `png` | Figure format |
| `dpi` | `300` | Figure resolution |

### `reporting`

| Key | Default | Description |
|-----|---------|-------------|
| `stop_after_layer` | `5` | `2` = L1–L2 only; `5` = full pipeline |
| `top_n_problem_markers` | `20` | Report truncation |
| `min_interval_frames` | `1` | Minimum frames for `qc_intervals` rows |

---

## Outputs

Each run creates a timestamped folder:

```
outputs/runs/{session_id}_{YYYYMMDD_HHMMSS}/
├── RUN_MANIFEST.json       # File list + row counts
├── config_used.yaml        # Config snapshot
├── qc_report.md            # Human-readable report
├── qc_reason_codes.md      # Reason code glossary
├── qc_report.xlsx          # Workbook (all tables)
├── tables/
│   └── *.csv
└── plots/
    └── *.png
```

### Essential tier (default)

**Tables:** `session_summary`, `gap_events`, `window_quality_summary`, `window_quality_0p5s`, `artifact_events`, `artifact_session_summary`, `qc_intervals`, `analysis_mask_summary`

**Plots:** `gap_timeline`, `window_quality_timeline`, `artifact_timeline`, `artifact_velocity_histogram` (+ per body-segment histograms `artifact_velocity_histogram_{group}.png`)

Set `outputs.tier: full` for all tables and plots (marker inventory, heatmaps, frame masks, etc.).

### Layer 6 batch package

```
outputs/batch_runs/batch_{YYYYMMDD_HHMMSS}/
├── BATCH_MANIFEST.json
├── dataset_eda_report.csv
├── dataset_eda_report.md          # PI deliverable
├── dataset_eda_report.xlsx
├── dataset_eda_workbook.xlsx      # PI workbook (ExecutiveSummary + per-session tabs)
├── failures.csv                   # if any session failed
├── config_snapshot.yaml
├── plots/
│   ├── batch_preprocessing_status.png
│   ├── batch_window_yield.png
│   ├── batch_artifact_events.png
│   └── batch_missingness.png
├── sessions/
│   └── {subject_id}_{session_id}.json
└── details/
    ├── top_markers_by_session.csv
    ├── artifact_type_distribution.csv
    └── velocity_by_body_segment.csv
```

### Key output tables

| File | Content |
|------|---------|
| `session_summary.csv` | One row: missing %, gap counts, preprocessing status |
| `gap_events.csv` | Every continuous missing interval per marker |
| `artifact_events.csv` | Clustered artifact **events** (duration, body group, method) |
| `artifact_session_summary.csv` | Event counts by class + recommendation text |
| `window_quality_0p5s.csv` | Per 0.5 s window: gap/artifact overlap, label, reasons |
| `qc_intervals.csv` | Caution/exclude intervals for sharing (labeled markers only) |
| `analysis_mask_summary.csv` | Frame counts by `use` / `caution` / `exclude_or_review` |

---

## Artifact detection policy

Two stages: **candidates** (frames) → **events** (clusters).

### Detection methods

| Method | What it finds |
|--------|----------------|
| `velocity_mad` | Local 3D speed peak above robust threshold |
| `single_frame_spike` | Jump away ≥ `min_jump_distance_m` and return near origin |
| `constant_position_hold` | Same XYZ for ≥ `min_repeated_frames` |
| `acceleration_mad` | Local acceleration peak (off by default) |

Candidates are **screening only** — not confirmed Motive filter artifacts. Visual review is expected.

### Event clustering

Consecutive candidate frames (same marker + same method) merge into one event:

| Duration | `event_class` |
|----------|----------------|
| 1 frame | `single_frame` or `single_frame_spike` |
| 2–5 frames | `short_burst` |
| >5 frames | `sustained` |

Each event has `start_frame`, `end_frame`, `duration_seconds`, `body_region_group`, `severity` (`minor` / `moderate` / `severe`).

### Impact on windows and intervals

- Any artifact event in a 0.5 s window → **caution**
- **Sustained** or **severe** event in window → **exclude_or_review**
- `qc_intervals` use **labeled markers only**; unlabeled body groups and unlabeled-only gap intervals are excluded from the report table

---

## Velocity calculation

No smoothing or Savitzky–Golay — raw QC on exported coordinates.

On each **gap-safe** contiguous valid segment:

```
speed = ||p[t+1] - p[t]|| / Δt
```

- Positions in meters (from CSV units)
- `Δt = 1 / frame_rate_hz`
- Speed is **not** computed across missing frames

Per-segment velocity histograms (notebook + plots) show the speed distribution, threshold lines, and flagged peaks for tuning `velocity_mad_multiplier` and `velocity_percentile_threshold`.

---

## Reason codes

Interval and window `reason_codes` map to plain language via [`motive_qc/reason_codes.py`](motive_qc/reason_codes.py). Written to `qc_reason_codes.md` each run.

Examples: `GAP_OVERLAP`, `LARGE_GAP_OVERLAP`, `ARTIFACT_EVENT_IN_WINDOW`, `SUSTAINED_ARTIFACT_IN_WINDOW`, `CRITICAL_GROUP_GAP`, `ELEVATED_MISSING`.

---

## Tuning tips (expressive movement)

Defaults target Gaga-style fast motion (fewer false artifact flags):

- Increase **Vel MAD σ** (e.g. 11–15) → looser
- Increase **Vel pct** (e.g. 99.97–99.99) → looser
- Keep **acceleration_mad** off unless investigating accel-specific glitches
- Increase **Spike jump m** if short real jumps are flagged
- Use notebook 02 **Re-run L4–L5** after slider changes
- Compare per-segment **velocity histograms** before tightening thresholds

**MAD σ:** higher = looser (fewer flags). **Percentile:** higher = looser.

---

## Python API (selected)

```python
from motive_qc import load_config, run_full_pipeline

config = load_config("config.yaml")
config["_base_dir"] = Path(".").resolve()
layer1, layer2, layer3, layer4, layer5, files = run_full_pipeline(config, verbose=True)
```

Layer-by-layer: `run_layer1_parse`, `run_layer2_gaps`, `run_layer4_artifacts`, `run_layer3_windows`, `run_layer5_report`, `write_outputs`.

Batch (Layer 6):

```python
from motive_qc import discover_sessions, run_batch, load_config

config = load_config("config.yaml")
config["_base_dir"] = Path(".").resolve()
catalog = discover_sessions(config)  # all subjects under data/
result = run_batch(config, subject_ids=["671"], verbose=True)
# PI report: result.report_paths["md"]
```

Histogram helpers: `collect_session_velocity_distribution`, `list_velocity_histogram_groups`, `flagged_velocity_speeds`.

---

## Data folder convention

```
data/
  671/          # subject 671 sessions
  252/          # subject 252 sessions
  archive/      # excluded by default (exclude_globs)
```

Filename pattern: `{subject}_{T#_P#_R#}_Take {date} {time}.csv`

Use **notebook 02 session picker**, `motive_batch_qc.py --discover`, or `discover_sessions(config)` to list sessions. **Load selected session** updates `config` paths before G0 cells.

---

## Validation workflow

1. **Notebook 01** — approve parse + gaps (Layer 1–2).
2. **Notebook 02** — run full pipeline; review artifact events, window flags, `qc_intervals`; use **Run batch for PI** for cross-session executive report.
3. Sign off via validation log widget → `docs/VALIDATION_LOG.md`.

The PI receives `outputs/batch_runs/batch_*/dataset_eda_report.md` and **`dataset_eda_workbook.xlsx`** (structured tabs with gap/artifact timelines per session) plus CSV/Excel and comparison plots — no notebook or CLI required on their side.

The pipeline **does not** modify raw CSV data or automatically exclude frames from analysis — it produces evidence and recommendations for human decisions before Motive preprocessing and PCA/jPCA.

---

## License / citation

Academic QC tooling for OptiTrack Motive raw exports. Cite session `qc_report.md`, `config_used.yaml`, and `RUN_MANIFEST.json` for reproducibility.

For implementation details and future layers (BVH, parallel batch), see [`docs/PROJECT_SPEC_MOTIVE_QC.md`](docs/PROJECT_SPEC_MOTIVE_QC.md).
