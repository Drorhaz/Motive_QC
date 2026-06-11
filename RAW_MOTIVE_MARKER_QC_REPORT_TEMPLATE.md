# Raw Motive Marker QC Report

## 1. Session and export identity

This report summarizes raw marker-level quality control for one Motive CSV export prior to preprocessing, skeleton solving, and BVH-based analysis.

| Field | Value |
|---|---|
| Session ID | `<session_id>` |
| Input file | `<filename>` |
| Motive version | `<motive_version>` |
| Export type detected | `<marker_xyz / mixed / solved>` |
| Frame rate | `<Hz>` |
| Frame range | `<start_frame>-<end_frame>` |
| Duration | `<seconds>` |
| Units | `<units>` |
| Labeled markers | `<n>` |
| Unlabeled marker tracks | `<n>` |
| Overall QC status | `<pass / caution / fail>` |

**Interpretation:** The input file was used as the raw marker-level QC source. Final movement analysis is planned on the processed BVH representation, using the frame/window QC mask derived here.

---

## 2. Marker completeness and gap structure

| Metric | Value |
|---|---:|
| Labeled marker missingness | `<%>` |
| Markers with any missing frames | `<n>` |
| Total continuous labeled-marker gaps | `<n>` |
| Gaps >=0.2 s | `<n>` |
| Gaps >=0.5 s | `<n>` |
| Longest labeled-marker gap | `<seconds>` |
| Critical body-region large gaps | `<yes/no>` |

**Key finding:** `<short sentence: e.g., Missingness was low and long gaps were limited to non-analysis regions / large torso gaps require caution.>`

**Figures:**

- Missing-data heatmap
- Gap duration histogram
- Gap timeline for moderate/large gaps

---

## 3. Unlabeled-marker burden

| Metric | Value |
|---|---:|
| Frames with any unlabeled marker | `<n>` |
| Percent frames with unlabeled markers | `<%>` |
| Max unlabeled markers in one frame | `<n>` |
| Longest unlabeled burst | `<seconds>` |
| Overlap with labeled marker gaps | `<yes/no>` |

**Interpretation:** Unlabeled markers were treated as a tracking-stability indicator, not as primary kinematic variables.

**Figure:** Unlabeled marker count over time.

---

## 4. Candidate artifact screening

| Metric | Value |
|---|---:|
| Velocity outlier candidates | `<n>` |
| Acceleration outlier candidates | `<n>` |
| Hold-fill candidates | `<n>` |
| Impossible-value candidates | `<n>` |
| Severe candidates | `<n>` |

**Interpretation:** Candidate artifacts were detected using robust kinematic outlier screening. These detections were not automatically removed and should be visually reviewed if they overlap planned analysis windows.

**Figure:** Artifact candidate timeline.

---

## 5. BVH analysis mask

The raw CSV QC was converted into a frame/window mask for later BVH analysis.

| Status | Number of frames/windows | Meaning |
|---|---:|---|
| Use | `<n>` | No relevant raw QC issue |
| Corrected acceptable | `<n>` | Minor issue expected to be acceptable after documented Motive preprocessing |
| Caution | `<n>` | Moderate gap, unlabeled burst, or artifact candidate |
| Exclude | `<n>` | Large gap or severe candidate affecting important movement data |

### Exclusion/caution intervals

| Start frame | End frame | Duration | Status | Reason | Recommended BVH action |
|---:|---:|---:|---|---|---|
| `<frame>` | `<frame>` | `<sec>` | `<caution/exclude>` | `<reason>` | `<action>` |

---

## Final QC conclusion

`<One concise conclusion, for example:>`

Raw marker-level QC showed that the session was suitable for documented Motive preprocessing and BVH-based movement-pattern analysis, with predefined caution/exclusion intervals listed above. The BVH analysis should use the generated frame/window mask and should not treat the BVH file as raw marker data.
