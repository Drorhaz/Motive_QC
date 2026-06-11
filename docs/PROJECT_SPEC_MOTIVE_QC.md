# PROJECT_SPEC_MOTIVE_QC.md

**Project:** Motive Raw Marker Quality-Control / EDA Pipeline  
**Primary data source:** OptiTrack Motive CSV export containing raw reconstructed marker XYZ trajectories  
**Motive version to record:** Motive:Body 3.4.0.2  
**Development mode:** Layered, validation-gated implementation  
**Current implementation scope:** **Implement Layers 1-2 only. Stop after Layer 2 and wait for researcher validation.**

---

## 0. Non-Negotiable Principles

This project is a scientific quality-control pipeline, not a cosmetic plotting script. It must prioritize reproducibility, transparency, and explicit failure behavior.

### Core rules

1. **No silent failures.**  
   Every assumption must be checked. Every failed assumption must either raise a clear exception or be recorded as an explicit warning in the output reports.

2. **No hidden data mutation.**  
   The raw CSV must never be modified in place. The pipeline may derive tables, masks, plots, and reports, but it must not gap fill, smooth, trim, interpolate, filter, relabel, resample, or transform coordinates unless a later validated layer explicitly implements that behavior.

3. **No overclaiming.**  
   The report must not say that data are "guaranteed raw." It may say:  
   **"The exported file is consistent with raw reconstructed Motive marker XYZ data, with missing values preserved, according to the checks performed."**

4. **Marker-level QC and solved movement analysis are separate.**  
   Raw Motive CSV marker trajectories are the source for the raw-data quality certificate. A later preprocessed BVH export may be used for continuous movement analysis, but BVH is a solved skeleton representation and must not be treated as raw marker data.

5. **Backend does all computation. Notebook is the interactive front-end.**  
   All parsing, math, validation, event detection, table generation, and plotting logic must live in backend Python modules or scripts. The Jupyter Notebook should only orchestrate steps, display outputs, and collect researcher decisions.

6. **Stop at validation gates.**  
   Cursor or any code-generation assistant must not implement later layers before the current layer outputs are manually reviewed and approved.

---

## 1. Project Purpose

The goal is to create a reproducible QC/EDA pipeline for **raw OptiTrack Motive marker XYZ CSV exports** before any gap filling, smoothing, trimming, skeleton solving, or BVH export.

The pipeline must produce a transparent raw-data certificate showing whether a recording segment is suitable for later preprocessing in Motive and subsequent continuous kinematic analyses such as PCA, jPCA, or jcvPCA.

### Intended scientific workflow

1. Export raw reconstructed marker data from Motive as CSV.
2. Run this QC/EDA pipeline on the raw CSV.
3. Confirm that the analyzed frames have no unacceptable gaps, missing-data clusters, spikes, or artifacts.
4. Document which frames, markers, and body regions require caution.
5. Apply documented Motive preprocessing later, such as gap filling and smoothing.
6. Export the preprocessed solved BVH later.
7. Run a second validation stage on the processed movement representation before PCA/jPCA/jcvPCA.

### Session preprocessing QC semantics

`raw_qc_preprocessing_status` (`acceptable` / `caution` / `poor`) uses **timeline-based** labeled-gap severity, not gap count alone. The primary metric is **union gap time**: unique seconds on the recording timeline where any labeled marker has a continuous gap at or above the moderate threshold (default 0.2 s). Per-marker gap durations are not summed (that would double-count overlapping gaps). Critical body-region single-gap duration and labeled missing percent are separate axes. Thresholds live in `quality_labels.session` in `config.yaml`.

### Target Minimal QC Report

The final human-readable QC report should follow `RAW_MOTIVE_MARKER_QC_REPORT_TEMPLATE.md`.

- The report should be **concise**, not exhaustive.
- It should contain only high-value sections:
  - session/export identity;
  - marker completeness and gap structure;
  - unlabeled-marker burden;
  - candidate artifact screening;
  - BVH analysis mask.
- The first implementation does **not** need to fully populate the artifact or BVH sections yet.
- **Layers 1-2** should generate the fields needed for report sections 1-3 and prepare the structure for section 5 via `frame_qc_mask.csv`.
- The report should be useful for sharing with supervisors/collaborators, not only for debugging.

### Important terminology

| Term | Meaning in this project |
|---|---|
| **Raw CSV** | Motive CSV export containing reconstructed marker XYZ trajectories before user-applied gap filling/smoothing. This is the QC source. |
| **Marker** | A physical or reconstructed 3D point trajectory. Marker-level data are not anatomical joint centers. |
| **Unlabeled marker** | A reconstructed marker trajectory not assigned to a known label. Included in master data, but summarized separately. |
| **Gap** | A continuous run of missing XYZ values for one marker. |
| **Solved data** | Rigid-body, skeleton, constraints, BVH, or other model-based outputs. Not raw marker trajectories. |
| **BVH** | Later processed/skeleton-solved movement representation. Not a marker-level raw-data source. |
| **Artifact candidate** | A suspicious kinematic event flagged by robust screening. Not a confirmed artifact until visually reviewed. |

---

## 2. Pipeline Architecture Overview

### 2.1 High-level data map

```text
Motive CSV export
    |
    v
Layer 1: Raw CSV ingestion + metadata + marker inventory
    |
    v
Canonical in-memory session object
    |
    v
Layer 2: Missingness + gap detection
    |
    v
[VALIDATION GATE: researcher reviews parser + gap outputs]
    |
    v
Layer 4: Artifact event screening (gap-safe kinematics)      [implemented in motive_qc v0.5]
    |
    v
Layer 3: Window safety verdict (L2 gaps + L4 events)         [implemented in motive_qc v0.5]
    |
    v
Layer 5: Publication-ready report package                    [implemented in motive_qc v0.5]
    |
    v
Later separate pipeline: processed BVH validation             [future, separate spec]
```

### 2.2 Recommended repository structure

```text
motive-qc/
├── PROJECT_SPEC_MOTIVE_QC.md
├── config.yaml
├── motive_qc/                  # package: parse, gaps, artifacts, windows, report, output_tiers
├── motive_raw_qc.py            # CLI entry point
├── requirements.txt
├── notebooks/
│   ├── 01_raw_csv_qc_layers_1_2.ipynb
│   └── 02_raw_csv_qc_layers_3_5.ipynb
├── data/
│   └── input_csv_here.csv
├── outputs/
│   └── runs/{session_id}_{timestamp}/   # essential or full tier per config
└── docs/
    ├── notes/
    └── VALIDATION_LOG.md
```

### 2.3 Separation of concerns

| Component | Responsibility | Should contain heavy computation? |
|---|---|---:|
| `config.yaml` | Tunable scientific and engineering settings | No |
| `motive_qc/` + `motive_raw_qc.py` | Layers 1-5 backend; CLI wrapper | Yes |
| Future modules, e.g. `motive_qc/parser.py` | Robust parsing logic | Yes |
| Future modules, e.g. `motive_qc/gaps.py` | Missingness and gap detection | Yes |
| Future modules, e.g. `motive_qc/plots.py` | Plot generation | Yes |
| Jupyter Notebook | Interactive orchestration, visualization, validation decisions | No |
| `VALIDATION_LOG.md` | Human approval/rejection record | No |

For the first implementation, a single backend script `motive_raw_qc.py` is acceptable. If the script grows beyond maintainable size, split it into a package after Layer 2 validation.

---

## 3. Canonical Data Structures and Schemas

The pipeline should convert Motive's CSV layout into a canonical internal representation. All downstream computations must use the canonical representation rather than repeatedly parsing CSV header rows.

### 3.1 Recommended internal session object

Use a typed dataclass or equivalent structure.

```text
MotiveSession
├── metadata: dict[str, Any]
├── frames: pandas.Index[int]
├── time_seconds: pandas.Series[float]
├── coordinates: xarray.DataArray
│   ├── dims: frame x marker x axis
│   ├── frame: integer Motive frame number
│   ├── marker: marker name string
│   └── axis: X, Y, Z
├── valid_marker_frame: xarray.DataArray[bool]
│   ├── dims: frame x marker
│   └── True only when X, Y, and Z are all valid finite values
├── marker_inventory: pandas.DataFrame
└── validation_messages: list[QCMessage]
```

If `xarray` is not used in the first implementation, the fallback structure is a pandas `DataFrame` with a column `MultiIndex`:

```text
column level 0: marker_name
column level 1: axis, one of X/Y/Z
index: frame number
values: float coordinates
```

However, `xarray.DataArray` is preferred because mocap data are naturally three-dimensional: **frame x marker x coordinate axis**.

### 3.2 Coordinate data schema

| Dimension | Type | Required? | Notes |
|---|---|---:|---|
| `frame` | integer | Yes | Motive frame number, expected continuous unless explicitly reported otherwise. |
| `marker` | string | Yes | Full marker name, e.g. `671:ChestTop` or `Unlabeled 12`. |
| `axis` | categorical string | Yes | Exactly `X`, `Y`, `Z`. |
| values | float | Yes | Position in Motive export units. Missing values must remain `NaN`. |

### 3.3 Metadata schema

Layer 1 must extract or explicitly mark as unknown the following metadata.

| Field | Type | Required behavior |
|---|---|---|
| `input_file` | string | Required. Absolute or relative path to raw CSV. |
| `file_stem` | string | Required. Used for output naming if batch mode is added later. |
| `motive_version` | string | From `config.yaml`; for this project record `Motive:Body 3.4.0.2`. |
| `capture_frame_rate_hz` | float or null | Extract from CSV if present; otherwise warn. |
| `export_frame_rate_hz` | float or null | Extract from CSV if present; otherwise use config override or fail. |
| `effective_frame_rate_hz` | float | Required for gap-duration conversion. Must be from export rate or config override. |
| `total_frames_metadata` | int or null | Extract from CSV if present; compare to observed rows. |
| `total_frames_observed` | int | Required. Number of data rows parsed. |
| `duration_seconds` | float | Required. Computed from observed frames and frame rate. |
| `start_frame` | int | Required. First frame in parsed data. |
| `end_frame` | int | Required. Last frame in parsed data. |
| `rotation_type` | string or null | Expected `XYZ` for marker position export; quaternions should be flagged if present. |
| `length_units` | string or null | Expected to be parsed if header contains it. Do not assume meters silently. |
| `coordinate_space` | string or null | Expected to be parsed if header contains it, e.g. global/local. |
| `axis_convention` | string or null | If unavailable, set `unknown` and warn; do not transform coordinates. |
| `raw_data_status` | string | `consistent_with_marker_xyz`, `ambiguous`, or `not_marker_xyz`. |

### 3.4 Marker inventory schema

`marker_inventory.csv` must have one row per marker trajectory.

| Column | Type | Description |
|---|---|---|
| `marker_name` | string | Full parsed marker name. |
| `marker_short_name` | string | Name without subject prefix if safely parsable. |
| `subject_or_asset_prefix` | string or null | Prefix before `:` if present, e.g. `671`. |
| `is_labeled` | bool | False for unlabeled markers. |
| `is_unlabeled` | bool | True for unlabeled markers. |
| `marker_type_raw` | string | Header type, e.g. `Marker`, if present. |
| `body_region_group` | string | Keyword-based group from config. |
| `x_column_source` | string or int | Original CSV column reference. |
| `y_column_source` | string or int | Original CSV column reference. |
| `z_column_source` | string or int | Original CSV column reference. |
| `has_x` | bool | True only if X column parsed. |
| `has_y` | bool | True only if Y column parsed. |
| `has_z` | bool | True only if Z column parsed. |
| `parse_status` | string | `ok`, `missing_axis`, `duplicate_name`, `non_marker_type`, or other explicit status. |

### 3.5 Raw-data constraints

For Layers 1-2, the pipeline must not perform any of the following:

- no coordinate transformations;
- no interpolation;
- no smoothing;
- no filtering;
- no marker relabeling;
- no deletion of frames;
- no deletion of markers;
- no gap filling;
- no rigid-body solving;
- no skeleton solving;
- no BVH parsing.

---

## 4. Config YAML Requirements

The configuration file must contain all tunable scientific thresholds and output settings. The Python code should not hide important thresholds inside functions.

### 4.1 Required `config.yaml` template

```yaml
project:
  project_name: "Motive raw marker QC"
  subject_id: "671"
  session_id: "T2_P1_R1"
  motive_version: "Motive:Body 3.4.0.2"
  analysis_stage: "raw_csv_before_preprocessing"
  notes: "Raw marker XYZ export before Motive gap filling, smoothing, skeleton solving, or BVH export."

paths:
  input_csv: "data/input_csv_here.csv"
  output_dir: "outputs/generated_by_script"

time:
  infer_frame_rate_from_file: true
  frame_rate_hz_override: null
  require_capture_export_rate_match: true
  allow_time_column_tolerance_seconds: 0.0005

parsing:
  require_marker_xyz_triplets: true
  accepted_marker_types:
    - "Marker"
  fail_on_duplicate_marker_names: true
  fail_on_missing_xyz_axis: true
  fail_on_non_numeric_coordinate_values: false
  convert_blank_cells_to_nan: true
  preserve_raw_frame_numbers: true

markers:
  include_labeled_markers: true
  include_unlabeled_markers: true
  include_unlabeled_in_master_tables: true
  summarize_unlabeled_separately: true
  include_unlabeled_in_session_missing_percent: false
  exclude_markers: []
  include_only_markers: []

marker_groups:
  head_neck:
    keywords: ["Head", "Neck"]
  torso_chest_back:
    keywords: ["Chest", "Back", "Sternum", "Spine"]
  pelvis_waist:
    keywords: ["Waist", "Pelvis", "Hip", "Sacrum"]
  shoulder_upper_arm:
    keywords: ["Shoulder", "UpperArm", "Arm"]
  elbow_forearm:
    keywords: ["Elbow", "Forearm"]
  wrist_hand:
    keywords: ["Wrist", "Hand"]
  fingers:
    keywords: ["Thumb", "Index", "Pinky", "Finger"]
  thigh_knee:
    keywords: ["Thigh", "Knee"]
  shank_ankle:
    keywords: ["Shin", "Shank", "Ankle"]
  foot:
    keywords: ["Heel", "Toe", "Foot"]
  unlabeled:
    keywords: ["Unlabeled"]
  unclassified:
    keywords: []

gaps:
  use_greater_equal_thresholds: true
  report_all_gaps: true
  report_single_frame_gaps: true
  thresholds_seconds:
    any_gap: 0.0
    tiny_gap: 0.025
    minor_gap: 0.100
    moderate_gap: 0.200
    large_gap: 0.500
    severe_gap: 1.000
  primary_report_thresholds_seconds: [0.1, 0.2, 0.5, 1.0]

quality_labels:
  marker:
    clean:
      max_missing_percent: 1.0
      max_large_gaps: 0
    minor_issue:
      max_missing_percent: 5.0
      max_large_gaps: 0
    caution:
      max_missing_percent: 10.0
      max_large_gaps: 1
    poor:
      missing_percent_above: 10.0
      large_gaps_above: 1
  session:
    acceptable_for_preprocessing:
      max_labeled_missing_percent: 5.0
      max_large_gaps_labeled: 0
    caution_for_preprocessing:
      max_labeled_missing_percent: 10.0
      max_large_gaps_labeled: 2

frame_quality:
  enabled: false  # Layer 3 only. Do not implement in Layers 1-2.
  missing_marker_percent_warn: 5.0
  missing_marker_percent_caution: 10.0
  missing_marker_percent_bad: 20.0
  critical_groups:
    - torso_chest_back
    - pelvis_waist
    - head_neck
    - thigh_knee
    - shank_ankle
    - foot

windows:
  enabled: false  # Layer 3 only. Do not implement in Layers 1-2.
  window_lengths_seconds: [0.5, 1.0]
  use_non_overlapping_windows: true
  step_seconds: null
  flag_if_gap_at_least_seconds: 0.2
  flag_if_large_gap_at_least_seconds: 0.5
  flag_if_missing_marker_percent_above: 10.0

artifacts:
  enabled: true  # Layer 4; set false to skip kinematic screening.
  methods:
    velocity_mad: true
    acceleration_mad: true
    single_frame_spike: true
    constant_position_hold: true
  velocity_mad_multiplier: 8.0
  acceleration_mad_multiplier: 8.0
  velocity_percentile_threshold: 99.9
  acceleration_percentile_threshold: 99.9
  minimum_valid_neighbors: 2
  single_frame_spike:
    return_near_original_tolerance_m: 0.005
    min_jump_distance_m: 0.050
  constant_position_hold:
    min_repeated_frames: 3
    tolerance_m: 0.000001

outputs:
  write_csv_tables: true
  write_excel_workbook: true
  write_text_summary: true
  write_html_report: false  # Layer 5 only unless explicitly unlocked.
  write_config_used: true
  plots:
    enabled: true
    marker_completeness: true
    gap_duration_histogram: true
    missing_data_heatmap_labeled: true
    missing_data_heatmap_unlabeled: true
    gap_timeline: false       # Layer 3 only.
    artifact_timeline: true  # Layer 4.
    session_psd_summary: true  # Spectral add-on.
    window_quality_timeline: false  # Layer 3 only.
  plot_format: "png"
  dpi: 300
  max_markers_per_heatmap: 80
  heatmap_downsample_max_frames: 5000

reporting:
  top_n_problem_markers: 20
  top_n_longest_gaps: 20
  include_methods_notes_stub: false
  stop_after_layer: 2
```

### 4.2 Config handling rules

| Rule | Required behavior |
|---|---|
| Missing config key | Raise `ConfigValidationError` unless a safe documented default exists. |
| Unknown config key | Warn explicitly; do not silently ignore unless under an allowed `extra` section. |
| Threshold units | Store gap thresholds in seconds; convert to frames using effective frame rate. |
| Threshold comparison | Use greater-than-or-equal (`>=`) because it is easier to explain in methods. |
| Config provenance | Copy the exact config used to `outputs/config_used.yaml`. |
| Runtime overrides | If notebook overrides any config value, write both original and final resolved config. |

---

## 5. Layer-by-Layer Technical Specification

## Layer 1 - Reliable Parser + Metadata + Marker Inventory

### Status

**Implement now.** This is part of the first Cursor task.

### Goal

Prove that the code reads the Motive CSV correctly and understands what data are present before any scientific interpretation is made.

### Input schema

| Item | Description |
|---|---|
| `config.yaml` | Required. Defines input path, output path, thresholds, grouping rules, and export settings. |
| Motive CSV | Required. Exported tracking data with header information. Expected to include marker XYZ positions. |

Expected Motive CSV content may include:

- metadata rows such as capture frame rate, export frame rate, total frames, units, rotation type, coordinate space;
- multi-row column headers describing data type, marker name, position/rotation, and axis;
- data rows with frame number, time, and marker coordinate columns.

The parser must be robust to minor header differences, but it must not silently guess if marker triplets cannot be identified.

### Output schema

#### `session_summary.csv`

One row per input CSV.

| Column | Type | Description |
|---|---|---|
| `file_name` | string | Input CSV basename. |
| `input_file` | string | Full or relative path. |
| `project_name` | string | From config. |
| `subject_id` | string | From config or parsed file name if allowed later. |
| `session_id` | string | From config or parsed file name if allowed later. |
| `motive_version` | string | From config. |
| `capture_frame_rate_hz` | float/null | Parsed from CSV. |
| `export_frame_rate_hz` | float/null | Parsed from CSV. |
| `effective_frame_rate_hz` | float | Used for all time conversion. |
| `frame_rate_status` | string | `ok`, `mismatch`, `missing_used_override`, or `missing_failed`. |
| `total_frames_metadata` | int/null | From CSV metadata, if present. |
| `total_frames_observed` | int | Number of parsed frames. |
| `frame_start` | int | First frame number. |
| `frame_end` | int | Last frame number. |
| `frame_continuity_status` | string | `continuous`, `missing_frames`, `duplicate_frames`, or `non_monotonic`. |
| `duration_seconds` | float | Observed duration. |
| `time_column_status` | string | `ok`, `missing`, `non_monotonic`, `inconsistent_with_frame_rate`, or `unknown`. |
| `length_units` | string/null | Parsed units. |
| `coordinate_space` | string/null | Parsed coordinate space. |
| `axis_convention` | string/null | Parsed if available; otherwise `unknown`. |
| `rotation_type` | string/null | Parsed if available. |
| `n_marker_triplets_total` | int | All marker XYZ triplets parsed. |
| `n_labeled_markers` | int | Labeled marker count. |
| `n_unlabeled_markers` | int | Unlabeled marker count. |
| `contains_marker_xyz` | bool | True if marker XYZ columns are present. |
| `contains_rigid_body_columns` | bool | True if solved rigid body data detected. |
| `contains_skeleton_columns` | bool | True if skeleton/bone data detected. |
| `contains_quaternion_columns` | bool | True if quaternion fields detected. |
| `raw_data_status` | string | `consistent_with_marker_xyz`, `ambiguous`, or `not_marker_xyz`. |
| `validation_status` | string | `pass`, `pass_with_warnings`, or `fail`. |
| `n_errors` | int | Count of validation errors. |
| `n_warnings` | int | Count of validation warnings. |

Layer 1 may leave Layer 2 columns blank or `not_computed` if the Excel workbook is already created. Do not invent missingness counts before Layer 2 runs.

#### `marker_inventory.csv`

Use the schema from Section 3.4.

#### `config_used.yaml`

Exact resolved configuration used for the run.

#### `qc_report_summary.txt`

A short text file containing:

- input file;
- frame count;
- frame rate;
- marker counts;
- whether data are consistent with marker XYZ;
- warnings/errors;
- explicit note that no preprocessing was applied by the script.

### Required plots

Layer 1 does not require plots. Optional: none.

### Kinematic and neuroscientific nuances

- Marker trajectories are reconstructed 3D positions, not anatomical joint centers.
- Rotation/quaternion fields indicate solved rigid-body or skeleton data, not raw marker XYZ. They should be flagged, not processed as marker coordinates.
- Coordinate system and units must be reported because later PCA/jPCA features can be affected by coordinate convention and scaling.
- No filtering or smoothing is allowed in Layer 1.

### Fail-safe and validation checks

| Check | Required behavior |
|---|---|
| Input CSV missing | Raise `FileNotFoundError` with path. |
| CSV unreadable | Raise parser exception with line/encoding context if possible. |
| No frame column | Fail. |
| Duplicate frame numbers | Fail unless config later allows. |
| Non-monotonic frames | Fail. |
| Missing frames | Warn or fail according to config; record exact missing frame ranges. |
| No marker XYZ triplets | Fail. |
| Marker missing X/Y/Z axis | Fail if `fail_on_missing_xyz_axis: true`. |
| Duplicate marker names | Fail if `fail_on_duplicate_marker_names: true`. |
| Capture/export rate mismatch | Fail if `require_capture_export_rate_match: true`; otherwise warn. |
| Units missing | Warn; do not assume meters silently. |
| Coordinate convention missing | Warn; do not transform coordinates. |
| Rigid body/skeleton/quaternion columns present | Warn and flag; do not include as raw marker data. |

### Acceptance checks before continuing

Researcher must verify:

- observed total frames match expected Motive export;
- effective frame rate is correct;
- marker count is plausible;
- labeled and unlabeled marker separation is correct;
- marker names match expected Motive marker names;
- no unexpected rigid-body/skeleton/quaternion data are being interpreted as raw markers;
- `raw_data_status` is acceptable.

### Stop condition

If Layer 1 fails, do not run Layer 2. Fix parsing/config first.

---

## Layer 2 - Missing Data and Gap Detection

### Status

**Implement now.** This is part of the first Cursor task.

### Goal

Quantify raw missingness and continuous gaps per marker without applying any correction.

### Input schema

| Input | Type | Dimensions |
|---|---|---|
| `MotiveSession.coordinates` | `xarray.DataArray` or pandas equivalent | frame x marker x axis |
| `MotiveSession.valid_marker_frame` | boolean array | frame x marker |
| `marker_inventory` | DataFrame | one row per marker |
| `config.gaps` | dict | thresholds in seconds |

A marker-frame is considered valid only when **X, Y, and Z are all finite numeric values**. If any axis is missing or non-finite, the entire marker-frame is invalid.

### Output schema

#### `marker_quality_summary.csv`

One row per marker.

| Column | Type | Description |
|---|---|---|
| `marker_name` | string | Full marker name. |
| `is_labeled` | bool | Labeled marker flag. |
| `is_unlabeled` | bool | Unlabeled marker flag. |
| `body_region_group` | string | Config-derived group. |
| `n_total_frames` | int | Total parsed frames. |
| `n_valid_frames` | int | Marker-frames with valid X/Y/Z. |
| `n_missing_frames` | int | Invalid/missing marker-frames. |
| `missing_percent` | float | `100 * missing / total`. |
| `n_gaps_total` | int | Number of continuous missing runs. |
| `n_single_frame_gaps` | int | Gaps with duration 1 frame. |
| `longest_gap_frames` | int | Longest continuous missing run. |
| `longest_gap_seconds` | float | Longest gap duration in seconds. |
| `mean_gap_frames` | float/null | Mean gap length if gaps exist. |
| `median_gap_frames` | float/null | Median gap length if gaps exist. |
| `n_gaps_ge_0p025s` | int | Count of gaps >= 0.025 s. |
| `n_gaps_ge_0p1s` | int | Count of gaps >= 0.1 s. |
| `n_gaps_ge_0p2s` | int | Count of gaps >= 0.2 s. |
| `n_gaps_ge_0p5s` | int | Count of gaps >= 0.5 s. |
| `n_gaps_ge_1p0s` | int | Count of gaps >= 1.0 s. |
| `first_missing_frame` | int/null | First missing frame, if any. |
| `last_missing_frame` | int/null | Last missing frame, if any. |
| `quality_label` | string | `clean`, `minor_issue`, `caution`, or `poor`. |
| `quality_reason` | string | Human-readable reason. |

#### `gap_events.csv`

One row per continuous marker gap.

| Column | Type | Description |
|---|---|---|
| `gap_id` | string/int | Unique gap identifier. |
| `marker_name` | string | Full marker name. |
| `is_labeled` | bool | Labeled marker flag. |
| `is_unlabeled` | bool | Unlabeled marker flag. |
| `body_region_group` | string | Marker group. |
| `gap_start_frame` | int | First missing frame in gap. |
| `gap_end_frame` | int | Last missing frame in gap. |
| `gap_start_time_seconds` | float | Start time relative to recording. |
| `gap_end_time_seconds` | float | End time relative to recording. |
| `duration_frames` | int | Inclusive duration in frames. |
| `duration_seconds` | float | `duration_frames / frame_rate_hz`. |
| `thresholds_crossed` | string | Semicolon-delimited labels crossed, e.g. `minor;moderate;large`. |
| `severity_label` | string | `single_frame`, `tiny`, `minor`, `moderate`, `large`, `severe`. |
| `prev_valid_frame` | int/null | Closest valid frame before gap. |
| `next_valid_frame` | int/null | Closest valid frame after gap. |
| `touches_start_or_end` | bool | True if gap is at first or last frame. |
| `recommended_status` | string | `document`, `caution`, or `potential_exclusion`. |

#### `gap_summary_by_marker.csv`

One row per marker, focused only on gap counts and duration thresholds. This may duplicate some `marker_quality_summary` fields for easy sharing.

#### `gap_summary_by_group.csv`

One row per body-region group.

| Column | Type | Description |
|---|---|---|
| `body_region_group` | string | Group label. |
| `n_markers` | int | Markers in group. |
| `n_labeled_markers` | int | Labeled markers in group. |
| `n_unlabeled_markers` | int | Unlabeled markers in group. |
| `total_missing_frames` | int | Sum across markers. |
| `mean_missing_percent` | float | Mean marker missing percent. |
| `max_missing_percent` | float | Worst marker missing percent. |
| `n_gaps_total` | int | Total gaps. |
| `n_gaps_ge_0p2s` | int | Moderate-or-longer gaps. |
| `n_gaps_ge_0p5s` | int | Large-or-longer gaps. |
| `n_gaps_ge_1p0s` | int | Severe gaps. |
| `longest_gap_frames` | int | Longest gap in group. |
| `longest_gap_seconds` | float | Longest gap in seconds. |
| `worst_marker` | string | Marker with highest missing percent or longest gap. |

#### Updated `session_summary.csv`

Layer 2 must update or regenerate session summary with:

| Column | Type | Description |
|---|---|---|
| `total_marker_frames_all` | int | frames x all markers. |
| `missing_marker_frames_all` | int | All marker missing frames including unlabeled. |
| `missing_percent_all` | float | Missing percent including unlabeled. |
| `total_marker_frames_labeled` | int | frames x labeled markers. |
| `missing_marker_frames_labeled` | int | Labeled marker missing frames. |
| `missing_percent_labeled` | float | Main missingness percentage for reporting. |
| `total_marker_frames_unlabeled` | int | frames x unlabeled markers. |
| `missing_marker_frames_unlabeled` | int | Unlabeled marker missing frames. |
| `missing_percent_unlabeled` | float | Unlabeled missingness percentage. |
| `n_gaps_total_all` | int | All marker gaps. |
| `n_gaps_total_labeled` | int | Labeled marker gaps. |
| `n_gaps_ge_0p2s_labeled` | int | Labeled gaps >= 0.2 s. |
| `n_gaps_ge_0p5s_labeled` | int | Labeled gaps >= 0.5 s. |
| `n_gaps_ge_1p0s_labeled` | int | Labeled gaps >= 1.0 s. |
| `longest_gap_marker_labeled` | string/null | Marker with longest labeled gap. |
| `longest_gap_seconds_labeled` | float | Longest labeled gap. |
| `raw_qc_preprocessing_status` | string | `acceptable`, `caution`, or `poor`, based on config. |
| `raw_qc_status_reason` | string | Human-readable reason. |

### Required plots for Layer 2

All plots must be written to `outputs/generated_by_script/plots/`.

| Plot file | Required? | Description |
|---|---:|---|
| `marker_completeness.png` | Yes | Bar plot showing percent valid per marker. Separate labeled/unlabeled visually if possible. |
| `gap_duration_histogram.png` | Yes | Distribution of gap durations in seconds, with threshold lines at 0.1, 0.2, 0.5, and 1.0 s. |
| `missing_data_heatmap_labeled.png` | Yes | Marker x frame missingness heatmap for labeled markers. Downsample frames if needed but document downsampling. |
| `missing_data_heatmap_unlabeled.png` | Optional but recommended | Same for unlabeled markers if present; separate to avoid dominating labeled summary. |

### Kinematic and neuroscientific nuances

- A gap is not always equally harmful. A torso marker gap may affect skeleton solving more than a short finger marker gap, depending on the downstream model.
- Missing values must remain missing in raw QC. Filling them before this layer invalidates the raw-data certificate.
- Long gaps should be documented because interpolation over long occlusions may distort movement structure and later PCA/jPCA trajectories.
- Unlabeled markers can naturally appear and disappear. They must be included in master tables but summarized separately from labeled markers to avoid misleading session-level quality statistics.
- For PCA/jPCA, missingness patterns matter not only by marker but also by frame/time. Layer 2 records event-level gaps; Layer 3 will later convert these to frame/window warnings.

### Fail-safe and validation checks

| Check | Required behavior |
|---|---|
| Missing coordinates are blank strings | Convert to `NaN` and record conversion. |
| Only one axis missing | Treat marker-frame as missing; record in optional diagnostic count. |
| Infinite values | Treat as invalid; record warning. |
| Non-numeric coordinate strings | Convert to `NaN` if config allows; otherwise fail. |
| Threshold conversion | Store both seconds and frame equivalent in report. |
| Zero gaps | Valid result; do not treat as failure. |
| Unlabeled markers absent | Valid result; report `n_unlabeled_markers = 0`. |
| Heatmap too large | Downsample for plotting only; never downsample data used for tables. |

### Acceptance checks before continuing

Researcher must verify:

- `marker_quality_summary.csv` marker count matches `marker_inventory.csv`;
- `gap_events.csv` correctly reports gap start/end frames;
- `duration_frames` is inclusive and matches expected counts;
- `duration_seconds = duration_frames / effective_frame_rate_hz`;
- gaps >= 0.2 s and >= 0.5 s are reproducible;
- labeled and unlabeled summaries are separate;
- plots are readable;
- no gap filling, smoothing, or filtering was performed.

### Stop condition

**After Layer 2 implementation and output generation, stop.**  
Do not implement Layer 3, Layer 4, Layer 5, BVH parsing, processed-data validation, or automated methods text until the researcher reviews and approves the Layer 1-2 outputs.

---

## Layer 3 - Body-Region Grouping + Frame/Window Warnings

### Status

**Implemented** in `motive_qc/windows.py` and notebook `02_raw_csv_qc_layers_3_5.ipynb`. **Runs after Layer 4** in v0.5 so window verdicts combine L2 gap overlap and L4 artifact events.

### Goal

Make the QC report useful for later PCA/jPCA/jcvPCA window selection by identifying risky fixed-duration windows using both gap evidence (L2) and artifact events (L4).

### Input schema

- Validated Layer 1 session object.
- Validated Layer 2 gap events.
- Configured marker groups.
- Configured window lengths, e.g. 0.5 s and 1.0 s.

### Outputs to implement later

| Output | Description |
|---|---|
| `frame_quality_summary.csv` | One row per frame: missing marker count, missing percent, affected groups, large-gap flag. |
| `window_quality_0p5s.csv` | Non-overlapping or configured windows of 0.5 s. |
| `window_quality_1s.csv` | Non-overlapping or configured windows of 1.0 s. |
| `group_quality_summary.csv` | Missingness and gap summary by anatomical region/body segment. |
| `gap_timeline_by_group.png` | Timeline of gaps grouped by body region. |
| `frame_missingness_timeline.png` | Missing marker count/percent by frame. |
| `window_quality_timeline.png` | Usable/caution/exclude flags over time. |

### Future validation checks

- Window boundaries must be explicit and reproducible.
- Window duration in frames must be reported.
- A gap overlapping two windows must be counted in both windows, with overlap duration reported.
- No final PCA/jPCA frame exclusions should be made automatically. The script only warns.

---

## Layer 4 - Artifact Candidate Screening

### Status

**Implemented** in `motive_qc/artifacts.py`. Primary output is **clustered artifact events** (`artifact_events.csv`, `artifact_session_summary.csv`). Frame-level `artifact_candidates.csv` is optional (`outputs.write_frame_level_artifacts`). Spectral screening is **disabled by default** in v0.5 (`spectral_screen.enabled: false`).

### Goal

Identify potential spikes, jumps, marker swaps, constant-position holds, and extreme velocity/acceleration events without overclaiming. Events include duration, body segment, and window linkage for review.

### Important scientific caution

Artifact detection must be labeled as **screening only**. A robust statistical threshold can identify suspicious frames, but visual review is required before declaring confirmed artifacts.

### Future methods

| Method | Intended detection |
|---|---|
| Robust velocity MAD | Extremely large frame-to-frame speed relative to marker-specific baseline. |
| Robust acceleration MAD | Sudden changes in velocity, potentially indicating spikes. |
| Single-frame spike detector | Jump out and return near original trajectory. |
| Constant-position hold detector | Repeated identical XYZ values suggesting hold-fill or export artifact. |
| Nearby-gap context | Whether spike occurs immediately before/after a gap, where interpolation risk is higher. |

### Future outputs

| Output | Description |
|---|---|
| `artifact_candidates.csv` | One row per candidate event. |
| `artifact_summary_by_marker.csv` | Counts by marker and method. |
| `velocity_outlier_timeline.png` | Candidate events over time. |
| `artifact_candidates_by_group.png` | Candidate events by body region. |

### Future validation checks

- Thresholds must be in config, not hard-coded.
- Candidate events must not be automatically removed or corrected.
- `NaN` propagation must be preserved around gaps.
- Velocity/acceleration must not be computed across missing data gaps as if continuous.

### Spectral screening (deprecated in v0.5)

Optional `spectral_screen` module remains in the package for backward compatibility but is **not run** when `enabled: false`. Prefer artifact event screening for kinematic QC.

---

## Layer 5 - Publication-Ready Report Package

### Status

**Implemented** in `motive_qc/report.py`.

### Goal

Create a shareable thesis/paper QC package after Layers 1-4 are validated.

### Future outputs

| Output | Description |
|---|---|
| `qc_report.xlsx` | Workbook containing all validated tables as sheets. |
| `qc_report_summary.txt` | Plain-language summary. |
| `qc_report.html` | Optional interactive or static report. |
| `plots/` | Publication-quality figures. |
| `methods_notes_stub.md` | Later optional methods paragraph template. |

### Report language rules

The report may say:

> "The raw Motive CSV export was consistent with reconstructed marker XYZ trajectories. Missing data and continuous gaps were quantified before preprocessing."

The report must not say:

> "The data are guaranteed raw" or "no filtering ever occurred."

---

## 6. Backend Script vs Frontend Notebook Mapping

The Jupyter Notebook must be a transparent step-by-step interface. It should import backend functions, run one layer at a time, display tables/plots, and stop for human validation.

### 6.1 Notebook cell plan for initial implementation

| Notebook cell | Backend call | Purpose | Expected output |
|---|---|---|---|
| 1 | imports only | Import backend functions and display package versions. | Version table. |
| 2 | `load_config(path)` | Load and validate YAML. | Resolved config display. |
| 3 | `run_layer1_parse(config)` | Parse CSV, metadata, marker inventory. | `MotiveSession`, `session_summary`, `marker_inventory`. |
| 4 | `display_layer1_outputs(result)` | Show parser outputs. | Tables in notebook. |
| 5 | researcher markdown cell | Human validation checkpoint. | Researcher writes pass/fail notes. |
| 6 | `run_layer2_gaps(session, config)` | Missingness and gap detection. | Marker/gap tables and plots. |
| 7 | `display_layer2_outputs(result)` | Show gap outputs. | Tables, plots. |
| 8 | researcher markdown cell | Human validation checkpoint. | Approval or correction request. |
| 9 | `write_validation_log(...)` | Save validation decision. | `docs/VALIDATION_LOG.md` updated. |

### 6.2 Backend result pattern

Backend functions must return structured results, not just write files.

```text
QCResult
├── layer_name: str
├── status: "pass" | "pass_with_warnings" | "fail"
├── tables: dict[str, pandas.DataFrame]
├── figures: dict[str, pathlib.Path]
├── files_written: list[pathlib.Path]
├── messages: list[QCMessage]
└── exception: Exception | null
```

```text
QCMessage
├── severity: "INFO" | "WARNING" | "ERROR"
├── code: str
├── message: str
├── context: dict[str, Any]
└── suggested_action: str | null
```

### 6.3 Exception behavior

| Exception | Meaning | Notebook behavior |
|---|---|---|
| `ConfigValidationError` | Invalid or incomplete config. | Display error table; stop. |
| `MotiveCSVParseError` | CSV cannot be parsed safely. | Display parser context; stop. |
| `SchemaValidationError` | Expected columns/axes are missing. | Display missing schema items; stop. |
| `QCValidationError` | Scientific validation failed. | Display reason and suggested action; stop. |

### 6.4 Example notebook error display behavior

The notebook should catch backend exceptions only to present them clearly. It must not hide them.

```text
try:
    layer1 = run_layer1_parse(config)
except QCValidationError as exc:
    display_error_panel(exc)
    raise
```

The notebook may show a friendly error panel, but it must still raise or stop execution so the user cannot accidentally continue after failure.

---

## 7. Required Outputs for Current Implementation: Layers 1-2

For the first Cursor task, produce the following directory structure.

```text
outputs/generated_by_script/
├── tables/
│   ├── session_summary.csv
│   ├── marker_inventory.csv
│   ├── marker_quality_summary.csv
│   ├── gap_events.csv
│   ├── gap_summary_by_group.csv
│   ├── unlabeled_marker_summary.csv
│   ├── frame_qc_mask.csv
│   └── unlabeled_frame_counts.csv   # optional
├── plots/
│   ├── marker_completeness.png
│   ├── gap_duration_histogram.png
│   ├── missing_data_heatmap_labeled.png
│   ├── gap_timeline.png
│   └── unlabeled_count_over_time.png
├── qc_report_summary.md
├── qc_report.xlsx
└── config_used.yaml
```

`gap_summary_by_marker` remains available in the Excel workbook (`gap_by_marker` sheet) but is not written as a separate CSV in the lean output set.

### Excel workbook requirements

`qc_report.xlsx` must contain at minimum:

| Sheet | Source table |
|---|---|
| `session_summary` | `session_summary.csv` |
| `marker_inventory` | `marker_inventory.csv` |
| `marker_quality` | `marker_quality_summary.csv` |
| `gap_events` | `gap_events.csv` |
| `gap_by_marker` | `gap_summary_by_marker.csv` |
| `gap_by_group` | `gap_summary_by_group.csv` |
| `unlabeled_summary` | `unlabeled_marker_summary.csv` |
| `unlabeled_frames` | `unlabeled_frame_counts.csv` |
| `frame_qc_mask` | `frame_qc_mask.csv` |
| `validation_messages` | generated from `QCMessage` list |
| `config_summary` | flattened config values |

Excel export is required because it is the easiest format to share with supervisors and collaborators. CSV files are still required for reproducible downstream analysis.

---

## 8. Documentation and Edge-Case Guardrails

### 8.1 Assumption matrix

| Assumption | Why it matters | How to test | If violated |
|---|---|---|---|
| File is a Motive CSV export | Parser depends on Motive header structure. | Detect frame/time columns and marker XYZ header rows. | Fail with `MotiveCSVParseError`. |
| Data contain marker XYZ triplets | Raw QC requires marker positions. | Verify X/Y/Z columns per marker. | Fail if no valid triplets. |
| Marker XYZ are reconstructed 3D positions | Gap/missingness is marker-level. | Check header type `Marker`; flag solved types. | Warn/fail depending on config. |
| Sampling is uniform | Gap duration and velocity need frame rate. | Compare frame/time differences to frame rate. | Fail or warn; do not compute duration silently. |
| Frame numbers are continuous | Missing frames vs missing markers are different issues. | Check expected integer sequence. | Warn/fail and report missing frame ranges. |
| Missing marker data are represented as blank/NaN | Gap detection depends on missingness. | Convert blanks to NaN and count. | Record conversion; fail on unexpected strings if configured. |
| All axes missing together for a gap | Partial axis missing may indicate corruption. | Count partial-axis invalid frames. | Treat marker-frame as missing; warn. |
| Units are known | Speeds/jumps depend on scale. | Parse header. | Warn; artifact layer should not run without known units. |
| Coordinate convention is known | Later transformations/PCA interpretation depend on axes. | Parse header if available. | Warn; do not transform. |
| Unlabeled markers are not comparable to labeled markers | Unlabeled tracks may appear/disappear. | Detect unlabeled names/types. | Include in master data, summarize separately. |
| Body-region grouping by keyword is approximate | CSV markers are not joints. | Store matched keyword and group. | Unmatched markers go to `unclassified`. |
| BVH is not raw marker data | BVH is solved skeleton representation. | Treat BVH as future processed output only. | Do not use BVH in raw marker QC. |

### 8.2 Anomalous frame tracking

Layer 1-2 should create only gap-level events. Layer 3 will later convert them to frame/window quality.

Future `anomalous_frame_tracker` should include:

| Field | Description |
|---|---|
| `frame` | Motive frame number. |
| `time_seconds` | Time from start. |
| `n_missing_labeled_markers` | Labeled missing count. |
| `n_missing_unlabeled_markers` | Unlabeled missing count. |
| `affected_body_groups` | Semicolon-delimited body groups. |
| `in_gap_ge_0p2s` | True if any moderate gap overlaps frame. |
| `in_gap_ge_0p5s` | True if any large gap overlaps frame. |
| `artifact_candidate_count` | Future Layer 4. |
| `frame_quality_label` | Future Layer 3. |
| `reason` | Human-readable reason. |

### 8.3 Quality score philosophy

A single numerical quality score can be helpful for dashboards, but it can also hide important details. Therefore:

- Layers 1-2 should prioritize transparent counts and event tables.
- Any future quality score must be decomposable into visible components.
- The report should never replace gap tables with only a score.

Recommended future quality labels:

| Label | Meaning |
|---|---|
| `usable` | No major missingness/gaps/artifact candidates under configured thresholds. |
| `usable_with_caution` | Minor/moderate issues present; document before analysis. |
| `exclude_or_review` | Large/severe gaps or artifact candidates requiring manual review. |

---

## 9. Neuroscientific and Kinematic Nuances

### 9.1 Raw marker QC vs solved movement analysis

Raw marker QC answers:  
**"Was the recorded marker evidence good enough to support later preprocessing and solving?"**

BVH or solved skeleton analysis answers:  
**"Is the final continuous movement representation suitable for the planned analysis?"**

These are related but not identical. The thesis/paper should report both stages separately if BVH is used for final analysis.

### 9.2 Rigid bodies, constraints, and solved markers

Rigid-body and skeleton constraints may provide continuous estimated positions even when physical markers are occluded. These solved positions are not equivalent to actual reconstructed marker locations. Therefore:

- Raw QC should use marker XYZ columns.
- Rigid-body, skeleton, bone, constraint, and quaternion fields must be flagged separately.
- These fields must not be mixed into marker missingness calculations.

### 9.3 Marker dropping and occlusion

A marker dropout should appear as missing XYZ values in the raw CSV. The QC pipeline must preserve this missingness. Continuous gaps must be reported with exact frame ranges so later preprocessing decisions can be justified.

### 9.4 Gap filling limits

Gap filling can be acceptable for short occlusions when surrounding trajectories support reliable reconstruction. However, long gaps are risky because interpolation may over-complete or distort movement. This pipeline must not decide whether a gap can be filled automatically; it should provide the evidence required for that decision.

### 9.5 Filtering and high-frequency movement nuance

Somatic movement can contain meaningful high-frequency components, especially in rapid corrective movements, tremor-like fluctuations, finger motion, or abrupt transitions. Later smoothing/filtering should therefore be documented with:

- Motive tool name;
- filter type if documented;
- cutoff/window/smoothing strength if available;
- whether filtering was applied to markers, skeleton, or both;
- order of operations relative to gap filling and skeleton solving.

Do not invent Motive filter details. Exact algorithmic descriptions must be verified from Motive:Body 3.4.0.2 documentation, export settings, or screenshots before they are written into thesis methods.

### 9.6 PCA/jPCA/jcvPCA sensitivity

PCA-like analyses can be distorted by:

- missing values;
- filled gaps that span a meaningful portion of the movement;
- sudden spikes;
- marker swaps;
- non-uniform sampling;
- inconsistent coordinate scaling;
- excessive smoothing that removes meaningful dynamics;
- analyzing windows that overlap major correction events.

This is why Layer 2 records exact gap events and Layer 3 will later create frame/window warnings.

---

## 10. Cursor Implementation Instructions

### 10.1 First Cursor task

Cursor must be instructed exactly:

```text
Read PROJECT_SPEC_MOTIVE_QC.md carefully.
Implement only Layer 1 and Layer 2.
Do not implement Layer 3, Layer 4, Layer 5, BVH parsing, coordinate transformations, filtering, smoothing, interpolation, or automatic artifact rejection.
Create config.yaml, motive_raw_qc.py, requirements.txt, and an optional starter notebook for Layers 1-2.
After implementation, summarize files created and stop.
```

### 10.2 What Cursor must implement now

| Item | Required now? |
|---|---:|
| YAML config loading and validation | Yes |
| Motive CSV parser | Yes |
| Metadata extraction | Yes |
| Marker inventory | Yes |
| Labeled/unlabeled classification | Yes |
| Marker grouping by config keywords | Yes |
| Missingness mask | Yes |
| Continuous gap detection | Yes |
| Session summary | Yes |
| Marker quality summary | Yes |
| Gap event table | Yes |
| Gap summaries by marker/group | Yes |
| Basic plots | Yes |
| Excel workbook | Yes |
| Text summary | Yes |
| Config copy | Yes |
| Notebook front-end | Recommended |

### 10.3 What Cursor must not implement yet

| Item | Reason |
|---|---|
| Frame/window quality summaries | Layer 3, requires validation of gap outputs first. |
| Velocity/acceleration artifact detection | Layer 4, threshold tuning later. |
| PCA/jPCA feature extraction | Outside raw QC and not validated yet. |
| BVH parsing | Later separate processed-data validation pipeline. |
| Gap filling | Must never occur in raw QC. |
| Smoothing/filtering | Must never occur in raw QC. |
| Coordinate transformations | Later only after explicit coordinate convention verification. |
| Automated exclusion decisions | The script warns; researcher decides. |
| Methods paragraph generation | Later, after settings are finalized. |

---

## 11. Validation Workflow

### 11.1 Human validation gate after Layer 1

Researcher checks:

- input file is correct;
- frame count is correct;
- frame rate is correct;
- units/coordinate metadata are correct or explicitly unknown;
- marker names are parsed correctly;
- labeled and unlabeled markers are separated correctly;
- no solved data are included as raw markers.

### 11.2 Human validation gate after Layer 2

Researcher checks:

- missingness counts are plausible;
- longest gaps match manual spot checks;
- gap start/end frames are inclusive and correct;
- gap thresholds use `>=` exactly as configured;
- labeled and unlabeled summaries are separated;
- heatmaps and histograms are readable;
- no correction was applied.

### 11.3 `VALIDATION_LOG.md` template

Create or update `docs/VALIDATION_LOG.md` after each run.

```text
## Validation Run

Version: v0.2
Input file:
Date run:
Motive version recorded:
Expected frame count:
Observed frame count:
Expected frame rate:
Observed/effective frame rate:
Expected marker count:
Observed marker count:
Expected major gaps:
Observed major gaps:
Layer 1 decision: approved / needs correction
Layer 2 decision: approved / needs correction
Validated by:
Notes:
Decision: approved / needs correction
```

---

## 12. Source Verification Notes

The pipeline should be grounded in official OptiTrack/Motive terminology, but the code must not rely on undocumented assumptions when the CSV itself can be inspected.

### Facts to preserve in documentation and methods

- Motive CSV marker export can include reconstructed marker XYZ positions.
- Unlabeled markers can be included in CSV exports and should be handled separately in summaries.
- Rigid-body constraints and bone constraints are solved/model-based positions and are distinct from actual raw marker positions.
- Motive Edit Tools include trimming, gap filling, smoothing, and swapping trajectories; these are preprocessing/editing operations and must not be performed in raw QC.
- BVH is a solved skeleton export format and is not the raw marker-data source for this QC pipeline.

### Required later verification before thesis methods text

Before writing the final methods section, verify and document from the exact project settings:

- Motive:Body version;
- raw CSV export options used;
- whether header information was included;
- units and axis convention;
- Motive gap filling settings;
- Motive smoothing/filter settings;
- skeleton solving settings;
- BVH export settings;
- final analysis frame windows.

---

## 13. Definition of Done for Version v0.2

Version v0.2 is complete only when all items below are true.

| Requirement | Status |
|---|---|
| `config.yaml` exists and validates. | Required |
| `motive_raw_qc.py` runs from command line with `--config config.yaml`. | Required |
| CSV parser extracts metadata and marker triplets. | Required |
| Labeled/unlabeled markers are classified. | Required |
| Marker groups are assigned. | Required |
| Missingness is computed from X/Y/Z finite values only. | Required |
| Continuous gaps are detected with inclusive start/end frames. | Required |
| Gap durations are reported in frames and seconds. | Required |
| Threshold counts use `>=`. | Required |
| Required CSV tables are written. | Required |
| Required plots are written. | Required |
| Excel workbook is written. | Required |
| `qc_report_summary.txt` is written. | Required |
| `config_used.yaml` is written. | Required |
| No smoothing, filtering, interpolation, gap filling, frame deletion, marker deletion, or coordinate transformation occurs. | Required |
| Cursor stops after Layer 2. | Required |

---

## 14. Command-Line Interface Requirement

The initial backend script should be executable as:

```bash
python motive_raw_qc.py --config config.yaml
```

Optional flags:

```bash
python motive_raw_qc.py --config config.yaml --dry-run
python motive_raw_qc.py --config config.yaml --verbose
```

Behavior:

| Flag | Expected behavior |
|---|---|
| `--config` | Required path to YAML config. |
| `--dry-run` | Validate config and input file, parse metadata only if safe, do not write full outputs. |
| `--verbose` | Print structured progress messages in addition to logs. |

All output files must be written under the configured output directory. The script must not write scattered files outside the output directory except for optional validation logs under `docs/` if explicitly configured.

---

## 15. Requirements File

Initial `requirements.txt` should include:

```text
pandas
numpy
pyyaml
matplotlib
openpyxl
xlsxwriter
xarray
```

Optional later:

```text
scipy
plotly
jinja2
```

Do not add optional dependencies until the layer that needs them is unlocked.

---

## 16. Final Reminder to Cursor

Implement conservatively. A smaller correct Layer 1-2 pipeline is better than a broad script that silently guesses.

**Implemented (v0.5):** Layers 1–5 with pipeline order L2→L4→L3→L5; artifact **events**; window verdicts combining gaps + artifacts; timestamped run folders (`outputs/runs/`); essential/full output tiers; `qc_reason_codes.md`; enriched `qc_intervals` with `reason_human` and `affected_body_groups`.  
**Still out of scope:** BVH parsing/mapping, filtering/smoothing/interpolation, Butterworth recommendations, PCA/jPCA, automatic exclusions.

