Read these files carefully before making changes:

1. PROJECT_SPEC_MOTIVE_QC.md
2. CURSOR_PIPELINE_REVIEW_PROMPT_REVISED.md
3. RAW_MOTIVE_MARKER_QC_REPORT_TEMPLATE.md
4. motive_raw_qc.py

Important hierarchy:
- PROJECT_SPEC_MOTIVE_QC.md is the master technical plan.
- motive_raw_qc.py is the current Layer 1–2 implementation foundation.
- RAW_MOTIVE_MARKER_QC_REPORT_TEMPLATE.md is the target human-readable report shape.
- CURSOR_PIPELINE_REVIEW_PROMPT_REVISED.md is a refinement guide, not permission to overbuild.

Goal:
Refine the existing Layer 1–2 raw Motive CSV QC pipeline so it becomes a clean, reliable foundation for a concise expert EDA report. Do not expand into a large dashboard or advanced analysis system.

Scientific workflow:
Raw Motive marker XYZ CSV
→ raw-data QC / EDA only
→ identify usable, caution, and exclude frames/windows
→ later preprocess / solve / export BVH in Motive
→ later analyze processed BVH while excluding or flagging frames/windows identified from raw CSV QC.

Critical wording:
- Do not call BVH raw data.
- Do not treat position + quaternion data as raw marker XYZ.
- The raw CSV is the QC evidence source.
- BVH is the later processed/skeleton-solved analysis representation.
- The QC pipeline must detect, label, report, and create masks only.
- Do not gap-fill, smooth, interpolate, filter, relabel, delete frames, delete markers, or transform coordinates.

Current code review conclusion:
motive_raw_qc.py is a good Layer 1–2 foundation, but not yet a stable core. It is close, but fix key issues before adding artifact screening, BVH parsing, or advanced window logic.

Known review verdict:
- Dynamic parsing: mostly good, but still somewhat fragile.
- Marker XYZ detection: mostly correct, but duplicate-axis detection needs fixing.
- Labeled/unlabeled separation: good for common names, but should be more robust.
- Gap duration logic: mostly correct; inclusive frames and >= thresholds are implemented.
- Output alignment with minimal report: partial, not complete.
- Code complexity: reasonable, not overbuilt.
- Silent dropping/mutation risk: some silent conversions/drops need fixing.
- Ready as core: almost, but fix key issues first.

Your task now:
First review the code and propose the exact minimal changes needed. Then implement only the approved Layer 1–2 refinements listed below.

Implement now:
1. Make parser validation safer:
   - verify true XYZ triplets per marker;
   - fix duplicate-axis / duplicate-marker detection;
   - avoid silently skipping suspicious marker columns;
   - report excluded non-marker/solved/quaternion columns clearly.
2. Improve labeled/unlabeled robustness:
   - support common unlabeled marker naming variants;
   - keep unlabeled markers in inventory;
   - summarize unlabeled separately from labeled markers.
3. Add compact unlabeled-marker burden outputs:
   - unlabeled_marker_summary.csv
   - optionally unlabeled_frame_counts.csv if simple and useful.
4. Add initial frame_qc_mask.csv:
   - one row per raw CSV frame;
   - include frame, time, missing_labeled_count, missing_labeled_percent, large_gap_present, moderate_gap_present, unlabeled_present, unlabeled_count, qc_status, reason_codes.
   - Use only raw CSV QC evidence.
   - Do not create bvh_frame_qc_mask yet unless mapping fields are clearly configured.
5. Align outputs with RAW_MOTIVE_MARKER_QC_REPORT_TEMPLATE.md:
   - session identity;
   - marker completeness and gaps;
   - unlabeled-marker burden;
   - initial BVH-analysis mask structure.
   - Artifact section can remain “not implemented in this layer.”
6. Keep the output set lean.

Required first-pass outputs:
Tables:
- session_summary.csv
- marker_inventory.csv
- marker_quality_summary.csv
- gap_events.csv
- gap_summary_by_group.csv
- unlabeled_marker_summary.csv
- frame_qc_mask.csv
- qc_report.xlsx
- qc_report_summary.md or qc_report_summary.txt
- config_used.yaml

Plots:
- missing_data_heatmap_labeled.png
- marker_completeness.png
- gap_duration_histogram.png
- gap_timeline.png
- unlabeled_count_over_time.png

Do not implement yet:
- full artifact candidate screening;
- velocity/acceleration outlier plots;
- artifact_summary_by_marker.csv;
- full window_quality_summary.csv;
- BVH parsing;
- BVH validation;
- post-processing distortion checks;
- HTML report;
- PCA/jPCA analysis;
- methods-text generation;
- automatic exclusion decisions.

Also update PROJECT_SPEC_MOTIVE_QC.md:
Add a short section called “Target Minimal QC Report” explaining:
- the final human-readable report should follow RAW_MOTIVE_MARKER_QC_REPORT_TEMPLATE.md;
- the report should be concise, not exhaustive;
- it should contain only high-value sections: session identity, gap structure, unlabeled burden, candidate artifact screening, and BVH mask;
- the first implementation does not need to fully populate artifact or BVH sections yet;
- Layer 1–2 should generate the fields needed for sections 1–3 and prepare the structure for section 5.

Before coding:
1. Briefly summarize what you found in the current code.
2. List the exact files you will edit.
3. List the outputs you will add or remove.
4. Confirm that you will not implement postponed layers.

After coding:
1. Summarize changes.
2. Explain how to run the script.
3. List generated outputs.
4. State what still belongs to later layers.