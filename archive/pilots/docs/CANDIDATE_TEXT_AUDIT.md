# FDA text cleanup before the CPU screen

The three datasets were prepared and the approved CPU screen completed successfully.
This audit concerns the fixed seed-42 samples described in CANDIDATE_SCREEN_PREPARATION.md.

## Approved edits, applied

Masked explicit recall-severity references with the same `[recall severity]` placeholder,
independently of the target label. Keep all other product and recall wording.
Keep legitimate medical-device regulatory classes, laser classes, and clinical classes.

Examples from the selected records:

| Original wording | Cleaned wording |
|---|---|
| Class II Recall - Worldwide Distribution | [recall severity] Recall - Worldwide Distribution |
| this recall is now being reclassified by the Agency as a Class I | this recall is now being reclassified by the Agency as a [recall severity] |
| FDA Class 2 Voluntary Recall | FDA [recall severity] Voluntary Recall |
| Class II medical device | Keep unchanged: this describes the device, not recall severity. |

The Device sample contains 11 selected records with explicit recall-severity references:

| CSV data row (zero-based) | Field |
|---:|---|
| 3052 | reason_for_recall |
| 3455 | reason_for_recall |
| 3806 | reason_for_recall |
| 12556 | reason_for_recall |
| 14282 | reason_for_recall |
| 2466 | distribution_pattern |
| 6070 | distribution_pattern |
| 7289 | distribution_pattern |
| 7344 | distribution_pattern |
| 11884 | distribution_pattern |
| 13879 | distribution_pattern |

Food row 2842, reason_for_recall, cites an upstream Class 2 recall. Its own target is
Class I; mask the reference consistently rather than inspecting whether it matches the label.

Device row 12727 contains a pasted administrative review template inside reason_for_recall.
Removed the span beginning `Pre-Review Reminders` and ending
`Doc ID 03534.05.06 2 of 8`. The surrounding sentence becomes:

> The system does not provide a warning or notification to alert the user that the configured units have reverted to the default units.

These edits affected 12 Device rows and 1 Food row. Row identifiers, targets, class proportions,
structured features, and missing-value counts were checked against saved before-edit copies
and are unchanged. The prepared CSVs now contain the cleaned text. Before/after changes are
recorded in data/raw/candidate_screens/fda_*_severity.text_changes.json.

The remaining 62 Device field values containing class phrases were reviewed as legitimate
device regulatory, laser, or clinical class information, rather than explicit recall severity.
No class phrases remain in the Food sample.

## Other checks

- All three prepared datasets have no duplicate complete feature rows.
- Each FDA dataset has one pair with identical combined text across different event IDs.
  The structured features differ; retain these independent source records under the agreed event rule.
- All FDA event IDs are unique in the prepared samples. IDs are outside the model CSVs.
- LendingClub retains the real home_ownership category `NONE`; it is not converted to missing.
- Partially missing ancillary fields are retained for the existing training-fold preprocessing.
- All six official split definitions contain every class on both sides. This checks data compatibility only; no learners were fitted.

The proposed local screen remains 2 learners x 3 conditions x 3 folds x 3 datasets = 54 evaluations.
The user approved the screen, and all 54 evaluations completed successfully. See results/new_candidate_screen_comparison.md.
