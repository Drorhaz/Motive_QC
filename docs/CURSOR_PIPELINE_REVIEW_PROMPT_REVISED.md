# Cursor Review Prompt: Refine the Motive Raw CSV QC Pipeline Without Overbuilding

## Purpose of this document

You are reviewing and refining an existing OptiTrack Motive raw marker QC/EDA project.

Your task is **not** to add every possible feature. Your task is to help produce a clean, efficient, scientifically defensible QC pipeline that serves the real analysis scope.

The project should produce a concise, easy-to-read QC report and the minimum set of tables/plots needed to justify later BVH-based analysis.

---

## Files you must read first

Read these files before proposing any changes:

1. `PROJECT_SPEC_MOTIVE_QC.md`  
   This is the **master technical plan / scientific contract**.

2. `motive_raw_qc.py`  
   This is the **current Layer 1-2 implementation foundation**. Review it before suggesting new features.

3. `RAW_MOTIVE_MARKER_QC_REPORT_TEMPLATE.md`  
   This is the **target minimal report shape**. The pipeline should eventually produce a report similar to this file.

Do not write code until you have reviewed all three and proposed a concise improvement plan.

---

## Scientific workflow to preserve

The intended workflow is:

```text
Raw Motive marker XYZ CSV
-> raw-data QC / EDA only
-> identify usable, caution, and exclude frames/windows
-> preprocess / solve / export BVH in Motive
-> analyze processed BVH while excluding or flagging frames/windows identified from raw CSV QC
```

Important distinctions:

- The raw CSV is **not** the final analysis dataset.
- The raw CSV is the **marker-level QC evidence source**.
- Final movement analysis will be performed on the **processed/skeleton-solved BVH representation**.
- The QC pipeline should create a concise raw-data certificate and a frame/window mask that can later be mapped to BVH frames.

---

## Non-negotiable terminology and scientific rules

1. Do **not** call BVH raw data.
2. Do **not** call position + quaternion data raw marker data.
3. Use **“raw marker-level Motive CSV export”** for marker XYZ data.
4. Use **“processed/skeleton-solved BVH representation”** for final analysis data.
5. Do not claim data are “guaranteed raw.” Use cautious wording such as:  
   **“consistent with raw Motive marker XYZ export, according to the checks performed.”**
6. The QC pipeline must not fill, smooth, trim, interpolate, filter, relabel, resample, delete, or correct raw data.
7. The QC pipeline detects, labels, summarizes, reports, and creates masks only.
8. Artifact detections, when implemented later, are **candidate warnings**, not confirmed artifacts unless visually/manual reviewed.
9. The implementation must be validation-gated: do not implement future layers before the current layer is reviewed and approved.

---

## Current code review summary: `motive_raw_qc.py`

The current `motive_raw_qc.py` is a good Layer 1-2 foundation, but it should **not yet be treated as the stable core**.

It is close, but key issues should be fixed before adding unlabeled-summary, frame-mask, artifact screening, BVH mapping, or report-generation layers.

| Area | Verdict |
|---|---|
| Dynamic parsing | Mostly good, but still somewhat fragile |
| Marker XYZ detection | Mostly correct, but duplicate-axis detection has a bug |
| Labeled/unlabeled separation | Good for common names, but should be more robust |
| Gap duration logic | Mostly correct; inclusive frames and `>=` thresholds are implemented |
| Output alignment with minimal report | Partial, not complete |
| Code complexity | Reasonable, not overbuilt |
| Silent dropping/mutation risk | Some silent conversions/drops need fixing |
| Ready as core? | Almost, but fix key issues first |

### Prioritize reviewing/fixing these before expanding

- Parser robustness: confirm header detection works across expected Motive CSV variants.
- Marker XYZ triplet detection: verify that only true marker XYZ columns are used.
- Duplicate axis/name detection: fix any logic that fails to detect duplicate marker-axis definitions.
- Labeled/unlabeled separation: make detection robust to common Motive unlabeled naming variants.
- Silent dropping risk: do not silently exclude columns, markers, axes, or data rows without logging a warning or failing clearly.
- Coordinate parsing: do not silently convert meaningful non-numeric values to NaN without recording diagnostics.
- Output alignment: make outputs serve the target minimal report, not just produce many tables.
- Keep code lean: do not add artifact/BVH/window modules before the core parser and gap outputs are validated.

---

## What should be added to the master plan

Add or recommend adding a short section to `PROJECT_SPEC_MOTIVE_QC.md` called:

```markdown
## Target Minimal QC Report
```

This section should state:

- The final human-readable report should follow `RAW_MOTIVE_MARKER_QC_REPORT_TEMPLATE.md`.
- The report should be concise, not exhaustive.
- The report should contain only high-value sections:
  - session/export identity;
  - marker completeness and gap structure;
  - unlabeled-marker burden;
  - candidate artifact screening;
  - BVH analysis mask.
- The first implementation does **not** need to fully populate the artifact or BVH sections yet.
- Layers 1-2 should generate the fields needed for report sections 1-3 and prepare the structure for section 5.
- The report should be useful for sharing with supervisors/collaborators, not only for debugging.

---

## What to keep from the previous review prompt

Keep these ideas, but apply them carefully and minimally.

| Addition | Keep? | Why |
|---|---:|---|
| Raw CSV is QC source; BVH is final analysis source | Yes | Central to the scientific workflow |
| Do not call BVH raw | Yes | Prevents methods overclaiming |
| Unlabeled-marker burden summary | Yes | Important and compact tracking-stability check |
| `unlabeled_frame_counts.csv` | Yes, maybe optional in first pass | Useful for overlap between unlabeled bursts and labeled gaps |
| `frame_qc_mask.csv` | Yes | Key bridge between raw QC and later BVH analysis |
| `bvh_frame_qc_mask.csv` | Later | Requires BVH export range/mapping details |
| Artifact candidate screening | Later | Important, but should not block clean raw CSV QC |
| “Do not overbuild” instruction | Yes | Essential to keep the project focused |
| Ask Cursor to critique before coding | Yes | Good validation workflow |

---

## What is too much right now

Postpone these until the core raw CSV QC is validated:

- full artifact candidate screening;
- velocity/acceleration outlier plots;
- `artifact_summary_by_marker.csv`;
- full `window_quality_summary.csv`;
- BVH parsing;
- BVH validation;
- post-processing distortion checks;
- HTML report;
- methods-text generation;
- PCA/jPCA/jcvPCA feature extraction;
- automated exclusion decisions.

These may be useful later, but they should not delay the first clean implementation.

---

## Recommended lean first implementation

The first implementation should produce enough to populate most of the minimal report template, but not the entire final system.

### First-pass tables

Required or strongly recommended:

1. `session_summary.csv`
2. `marker_inventory.csv`
3. `marker_quality_summary.csv`
4. `gap_events.csv`
5. `gap_summary_by_group.csv`
6. `unlabeled_marker_summary.csv`
7. `frame_qc_mask.csv`
8. `qc_report.xlsx`
9. `qc_report_summary.md` or `qc_report_summary.txt`
10. `config_used.yaml`

Optional, only if it does not bloat the first implementation:

- `gap_summary_by_marker.csv`
- `unlabeled_frame_counts.csv`

### First-pass plots

Keep plots few and readable:

1. `missing_data_heatmap_labeled.png`
2. `marker_completeness.png`
3. `gap_duration_histogram.png`
4. `gap_timeline.png`
5. `unlabeled_count_over_time.png`

Do not create many redundant plots. The goal is a concise expert EDA, not a dashboard.

---

## Target report shape

The pillar output is a concise report shaped like:

```text
RAW_MOTIVE_MARKER_QC_REPORT_TEMPLATE.md
```

The report should include these sections:

1. **Session and export identity**  
   What file was analyzed? What export type was detected? What frame rate, duration, marker counts, units, and coordinate space were found?

2. **Marker completeness and gap structure**  
   What was the labeled-marker missingness? How many gaps crossed important thresholds? What was the longest gap? Were critical regions affected?

3. **Unlabeled-marker burden**  
   How often did unlabeled markers appear? Were there bursts? Did they overlap labeled-marker gaps?

4. **Candidate artifact screening**  
   Later layer only. The first implementation may include this section as “not yet computed.”

5. **BVH analysis mask**  
   The first implementation should at least prepare a basic `frame_qc_mask.csv`. Full BVH-frame mapping can come later after BVH export settings are known.

---

## Minimal frame QC mask requirement

Because final analysis will be on BVH, the raw CSV QC should begin producing a simple frame-level mask.

First version may be simple:

| Column | Meaning |
|---|---|
| `raw_frame` | Motive frame number |
| `time_seconds` | Time in seconds |
| `n_missing_labeled_markers` | Count of missing labeled markers |
| `n_missing_unlabeled_markers` | Count of missing unlabeled markers, if available |
| `in_gap_ge_0p2s` | Whether frame overlaps any labeled gap >= 0.2 s |
| `in_gap_ge_0p5s` | Whether frame overlaps any labeled gap >= 0.5 s |
| `qc_status` | `use`, `caution`, or `exclude_or_review` |
| `reason_codes` | Semicolon-delimited reason codes |

Do not over-engineer this yet. It is only the first bridge toward later BVH masking.

Full `bvh_frame_qc_mask.csv` should be postponed until BVH export start/end frame and frame-rate mapping are known.

---

## Compact unlabeled-marker QC requirement

Add unlabeled-marker burden early, but keep it compact.

### `unlabeled_marker_summary.csv`

One row per session:

| Field | Meaning |
|---|---|
| `total_unlabeled_tracks` | Number of unlabeled marker tracks |
| `frames_with_any_unlabeled` | Number of frames containing unlabeled markers |
| `percent_frames_with_any_unlabeled` | Session-level unlabeled burden |
| `max_unlabeled_markers_in_frame` | Worst frame unlabeled count |
| `frame_of_max_unlabeled_count` | Frame where maximum occurred |
| `unlabeled_bursts_count` | Number of continuous unlabeled bursts |
| `longest_unlabeled_burst_sec` | Longest continuous unlabeled burst |
| `overlap_with_labeled_gaps` | Whether unlabeled detections overlap labeled gaps |

### Optional `unlabeled_frame_counts.csv`

One row per frame:

| Field | Meaning |
|---|---|
| `frame` | Raw Motive frame number |
| `time_seconds` | Time in seconds |
| `unlabeled_count` | Number of unlabeled markers present |
| `labeled_missing_count` | Number of labeled markers missing |
| `overlap_flag` | Unlabeled present while labeled markers are missing |

Recommended plot:

- `unlabeled_count_over_time.png`, ideally with major labeled gaps marked.

Do not deeply analyze each unlabeled trajectory unless it overlaps important labeled gaps.

---

## Artifact screening: later only

Artifact candidate screening is important for PCA/jPCA, but should not be implemented until the core raw CSV parser, missingness, gap, unlabeled, and frame-mask outputs are validated.

When implemented later:

- detections must be called **candidate artifacts**;
- no frames are automatically removed;
- thresholds must come from YAML;
- velocity/acceleration must not be computed across missing data gaps as if continuous;
- output should be concise, not a large artifact dashboard.

Recommended later wording:

> Candidate artifacts were detected by robust kinematic outlier screening and reviewed as potential QC warnings. These detections were not automatically removed.

---

## What Cursor should do now

After reading the master plan, current code, and report template, do the following **before writing code**:

1. Critique the current `motive_raw_qc.py` against the master plan and target report template.
2. Identify specific fixes required before treating it as stable Layer 1-2 core.
3. Identify which outputs already exist, which are missing, and which should be removed/postponed to avoid bloat.
4. Propose the smallest revised output set needed to populate `RAW_MOTIVE_MARKER_QC_REPORT_TEMPLATE.md`.
5. Propose exact changes needed to `PROJECT_SPEC_MOTIVE_QC.md`, especially the new `Target Minimal QC Report` section.
6. Propose exact changes needed to `config.yaml` for unlabeled summary and simple frame mask, without enabling artifact/BVH layers yet.
7. Propose a short implementation sequence.
8. Stop and wait for researcher approval before coding.

---

## Expected Cursor response format

Your response should include:

1. **Short critique of current code**  
   What is good, what is fragile, and what must be fixed first.

2. **Master plan refinements**  
   What to add/change in `PROJECT_SPEC_MOTIVE_QC.md`.

3. **Lean output list**  
   Required tables, optional tables, required plots, optional/later plots.

4. **Implementation order**  
   A small step-by-step plan that fixes the current core before adding anything.

5. **Postponed items**  
   Clearly list what should not be implemented yet.

6. **Approval checkpoint**  
   End by asking for approval before code changes.

---

## Final reminder

The master plan is `PROJECT_SPEC_MOTIVE_QC.md`.

The current implementation foundation is `motive_raw_qc.py`.

The required pillar report shape is `RAW_MOTIVE_MARKER_QC_REPORT_TEMPLATE.md`.

The goal is not maximum output. The goal is a **minimal, expert, easy-to-share QC report** plus a clean raw-to-BVH readiness bridge.

Do not overreach. Build the smallest scientifically defensible pipeline that serves the scope.
