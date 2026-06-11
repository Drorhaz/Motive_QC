"""QC reason code glossary and human-readable mappings."""

from __future__ import annotations

REASON_GLOSSARY: dict[str, tuple[str, str]] = {
    "LARGE_GAP": (
        "Labeled gap >=0.5 s overlaps this interval",
        "Review before gap-fill; consider BVH exclusion",
    ),
    "LARGE_GAP_OVERLAP": (
        "Labeled gap >=0.5 s overlaps this analysis window",
        "Review before PCA/jPCA window selection",
    ),
    "MODERATE_GAP": (
        "Labeled gap >=0.2 s overlaps interval",
        "Document; check Motive gap-fill settings",
    ),
    "GAP_OVERLAP": (
        "Labeled gap >=0.2 s overlaps analysis window",
        "Caution for window-based analysis",
    ),
    "HIGH_MISSING": (
        ">20% labeled markers missing in window",
        "Tracking/occlusion check",
    ),
    "ELEVATED_MISSING": (
        ">10% labeled markers missing in window",
        "Document elevated missingness",
    ),
    "ELEVATED_MISSING_LABELED": (
        "Elevated labeled marker missingness at frame level",
        "Check occlusion or marker dropout",
    ),
    "HIGH_MISSING_LABELED": (
        "High labeled marker missingness at frame level",
        "Priority review",
    ),
    "CRITICAL_GROUP_GAP": (
        "Gap in critical body region (torso/pelvis/head/legs)",
        "PCA-relevant caution",
    ),
    "ARTIFACT_CANDIDATE": (
        "Short kinematic outlier event in window",
        "Visual review recommended",
    ),
    "SEVERE_ARTIFACT_CANDIDATE": (
        "Severe or sustained kinematic outlier",
        "Priority visual review; may exclude",
    ),
    "ARTIFACT_EVENT_IN_WINDOW": (
        "One or more artifact events overlap analysis window",
        "Review event duration and body segment",
    ),
    "SUSTAINED_ARTIFACT_IN_WINDOW": (
        "Sustained artifact event (>5 frames) in window",
        "Likely exclude for PCA/jPCA",
    ),
    "UNLABELED_PRESENT": (
        "Unlabeled markers present (if flagged in config)",
        "Tracking stability indicator only",
    ),
}


def reason_codes_to_human(reason_codes: str) -> str:
    if not reason_codes or isinstance(reason_codes, float):
        return ""
    parts = [p.strip() for p in str(reason_codes).split(";") if p.strip()]
    human: list[str] = []
    for code in parts:
        entry = REASON_GLOSSARY.get(code)
        human.append(entry[0] if entry else code)
    return "; ".join(human)


def primary_reason_code(reason_codes: str) -> str:
    if not reason_codes:
        return ""
    return str(reason_codes).split(";")[0].strip()


def build_reason_codes_markdown() -> str:
    lines = [
        "# QC reason codes",
        "",
        "| Code | Meaning | Typical action |",
        "|---|---|---|",
    ]
    for code, (meaning, action) in sorted(REASON_GLOSSARY.items()):
        lines.append(f"| `{code}` | {meaning} | {action} |")
    return "\n".join(lines) + "\n"
