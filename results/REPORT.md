# Reproduction report

## Decision #1: result integrity

The official UCI split reproduces at 72.88--73.75% accuracy, not the thesis's 87--93%.
The thesis matrices use 1,200 rows whereas the official test set has 1,036. The authors
must decide whether they can supply the thesis `download.csv` for forensic comparison.
Until then, the official split is the only defensible primary result.

Exact combined-review text overlaps occur in 16 official test rows; 20 test rows have a
near-duplicate training text under token-set Jaccard >= 0.90. There are no exact complete
row overlaps. This is a diagnostic, not a reason to silently remove rows or replace the
headline results. The original vectorizer and encoder are fitted before CV, so CV and
learning-curve estimates have preprocessing leakage. A fold-contained sklearn Pipeline is
required only as a separately labelled supplementary analysis.

The exact-overlap diagnostic retains 1,020 test rows. Accuracy changes from 72.88% to
72.55% for LR, 73.75% to 73.33% for RF, and 73.75% to 73.33% for SVC. These are
diagnostic side-by-side values, not replacement headline results. Full metrics are in
`deduplicated_test_diagnostic.json`.

## What was reproduced

- Official UCI Drug Reviews (Druglib.com) TSVs loaded with `sep='\\t'`: 3,107 train and
  1,036 test rows.
- Thesis binning, feature construction, three models, grids, `cv=3`, seed 42, and balanced
  class weights were retained.
- Best official-split settings: LR C=1/l2/saga; RF depth=30, leaf=2, split=2,
  estimators=300; SVC C=10/rbf.

## Reporting fixes applied

1. Corrected per-class report labels by explicitly passing No, Mild, Severe labels.
2. Aligned ROC score columns with `model.classes_`; corrected macro/micro and per-class AUC.
3. Corrected Figure 4.2 description: appendix code measures character length (`len`), not
   word count.
4. Removed the unsupported claim that Random Forest is best. On this official split RF and
   SVC tie in accuracy, while SVC has slightly lower macro-F1.

## Known defects and limitations

- Logistic Regression reaches `max_iter=1000` without convergence for some fits; this was
  logged and not changed because 1,000 is the thesis setting.
- `sideEffectsReview` is text written about the target outcome, creating text-label proximity.
- Labels are self-reported, data are small and imbalanced, and no clinical validation exists.

## Decisions required from the authors

1. Provide the exact thesis `download.csv` and any original environment details.
2. Decide whether the official split replaces thesis headline values in the paper.
3. Approve a clearly labelled supplementary fold-contained Pipeline CV analysis.
4. Approve a clearly labelled de-duplicated-test diagnostic, if desired.
5. Review the reference audit before submission.

## Supplementary experiments

These analyses do not replace the official-split headline results. They are saved as
`supplementary_5fold_pipeline_cv.csv`, `supplementary_test_metrics.json`,
`supplementary_bootstrap_accuracy.json`, `supplementary_mcnemar_svc_vs_rf.json`, and
`supplementary_ablation.csv`.

- Fold-contained stratified five-fold CV gives mean accuracy 0.702 (LR), 0.709 (RF), and
  0.718 (SVC), with macro-F1 means 0.697, 0.694, and 0.702 respectively.
- Bootstrap 95% accuracy intervals are LR [0.701, 0.756], RF [0.711, 0.763], and SVC
  [0.709, 0.764] using 2,000 fixed-seed replicates.
- McNemar's exact two-sided test for SVC versus RF has 84 SVC-only correct and 84
  RF-only correct cases (p=1.0).
- The text-plus-categorical ablation is strongest for each model; text-only and
  categorical-only values are reported without tuning or cherry-picking.
