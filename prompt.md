You are working on a research-grade optical motion capture QC pipeline for OptiTrack Motive CSV exports.

Read PROJECT_SPEC_MOTIVE_QC.md completely before writing code.

Your task is to implement ONLY Layer 1 and Layer 2 from the specification.

Do not implement Layer 3, Layer 4, or Layer 5 yet.
Do not implement frame/window quality summaries yet.
Do not implement artifact/spike detection yet.
Do not implement BVH validation yet.
Do not implement final HTML reporting yet.

Current implementation scope:
1. Reliable Motive CSV parser
2. Metadata extraction
3. Marker inventory
4. Labeled vs unlabeled marker classification
5. XYZ marker triplet validation
6. Frame continuity validation
7. Missing-data quantification
8. Continuous gap detection
9. Gap severity classification using config.yaml thresholds with ≥ logic
10. CSV outputs for Layers 1–2
11. Excel workbook output
12. Basic text QC summary
13. Required Layer 2 plots
14. Jupyter Notebook frontend that imports backend code and runs Layers 1–2 step by step

Required project structure:
motive-qc/
├── PROJECT_SPEC_MOTIVE_QC.md
├── config.yaml
├── motive_raw_qc.py
├── requirements.txt
├── notebooks/
│   └── 01_raw_csv_qc_layers_1_2.ipynb
├── data/
│   └── input_csv_here.csv
├── outputs/
│   └── generated_by_script/
└── docs/
    └── notes/

Backend/frontend rule:
All heavy lifting, parsing, validation, calculations, and file writing must live in Python backend code.
The Jupyter Notebook must only import backend functions, run them step by step, display tables/plots, and allow researcher validation.

Required output files for Layers 1–2:
outputs/generated_by_script/
├── tables/
│   ├── session_summary.csv
│   ├── marker_inventory.csv
│   ├── marker_quality_summary.csv
│   ├── gap_events.csv
│   ├── gap_summary_by_marker.csv
│   └── gap_summary_by_group.csv
├── plots/
│   ├── marker_completeness.png
│   ├── gap_duration_histogram.png
│   └── missing_data_heatmap_labeled.png
├── qc_report_summary.txt
├── qc_report.xlsx
└── config_used.yaml

The Excel workbook must contain at least these sheets:
- session_summary
- marker_inventory
- marker_quality_summary
- gap_events
- gap_summary_by_marker
- gap_summary_by_group
- config_used

The script must be runnable from command line:
python motive_raw_qc.py --config config.yaml

The notebook must be runnable cell by cell and must:
- load the config
- run parser/metadata validation
- show session_summary
- show marker_inventory
- run missingness/gap detection
- show marker_quality_summary
- show longest gaps from gap_events
- display the generated plots
- stop with a clear validation checklist for the researcher

Config requirements:
Create or update config.yaml with tunable settings for:
- input_csv
- output_dir
- project/session metadata
- frame rate override and inference
- gap thresholds in seconds: 0.025, 0.1, 0.2, 0.5, 1.0
- use_greater_equal_thresholds: true
- unlabeled marker handling
- marker grouping keywords
- Excel output settings
- plot settings

Strict validation behavior:
No silent failures.
If parsing assumptions fail, raise a clear exception.
If frame numbers are missing or discontinuous, raise or flag according to config.
If XYZ triplets cannot be identified reliably, stop and explain why.
If metadata frame rate is missing and no config override is provided, stop and explain why.
If output files cannot be written, stop and explain why.
Do not silently drop markers, columns, or frames.

Scientific language constraints:
Do not say the data are “guaranteed raw.”
Use wording like “consistent with raw Motive marker XYZ export.”
Do not treat BVH as raw marker data.
Do not treat rigid bodies, skeletons, quaternions, or constraints as raw marker trajectories.
Artifact detection is out of scope for this version and must not be implemented yet.

After implementation:
1. Summarize exactly which files were created or modified.
2. Summarize how to run the command-line script.
3. Summarize how to run the notebook.
4. List the validation checks implemented.
5. Stop. Do not proceed to Layer 3.