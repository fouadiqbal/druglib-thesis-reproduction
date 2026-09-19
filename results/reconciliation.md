# Thesis-to-official-split reconciliation

## Data split

The thesis confusion matrices each total 1,200 test rows. The official UCI Druglib.com
test file has 1,036 rows. Therefore its reported tables cannot be a direct result of the
official test split used in this reproduction. No effort was made to tune or select a setup
to reproduce those values.

## Table 5.1 and Figure 4.7

| Model | Thesis accuracy | Official split rerun | Difference (percentage points) |
|---|---:|---:|---:|
| Random Forest | 87.33% / 0.87 | 73.75% | -13.58 |
| Logistic Regression | 87.00% / 0.87 | 72.88% | -14.12 |
| SVC | 92.83% / 0.93 | 73.75% | -19.08 |

The thesis Table 5.1 precision, recall, and F1 columns are **macro averages**, not
weighted averages. For example, the RF confusion matrix in Table 5.2 reconstructs to
macro P/R/F1 0.862/0.878/0.869, which rounds to 0.86/0.88/0.87. Its weighted values
instead round to 0.88/0.87/0.87.

## Tables 5.2--5.4

All three thesis matrices total 1,200, while every official-split rerun matrix totals 1,036.
The rerun matrices are saved as `confusion_matrix_original_*.csv` and
`confusion_matrix_corrected_*.csv`. The corrected versions use the explicit label order
No, Mild, Severe. The original classification reports were mislabelled because sklearn's
alphabetical class order is Mild, No, Severe.

The thesis SVM table has 32 where its figure has 31; this source conflict is retained as a
thesis discrepancy. Its discussion also calls off-diagonal values 11 and 20 correct
predictions, which is false.

## Tables 5.5--5.7 ROC AUC

The original reconstruction repeats the appendix's label/probability-column mismatch and
produces low AUCs for displayed No and Mild classes. Corrected per-class AUCs are:

| Model | Mild | No | Severe | Macro | Micro |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.809 | 0.882 | 0.931 | 0.875 | 0.886 |
| Random Forest | 0.814 | 0.869 | 0.933 | 0.873 | 0.889 |
| SVC | 0.811 | 0.883 | 0.931 | 0.875 | 0.895 |

The SVC ROC estimator is separately refit with `probability=True`, as in the appendix;
small probability-related variation is expected. Full values are in `roc_auc_corrected.json`.
