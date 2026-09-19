"""Report-only diagnostic: remove official test rows with exact train combined-review matches.

This does not retrain or replace the official-split headline result. It consumes saved
predictions from original_pipeline.py and documents the consequence of the exact overlap.
"""
from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
RAW, RESULTS = ROOT / "data" / "raw", ROOT / "results"
review_cols = ["benefitsReview", "commentsReview", "sideEffectsReview"]
labels = ["No Side Effects", "Mild Side Effects", "Severe Side Effects"]
norm = lambda x: " ".join(str(x).lower().split())
train = pd.read_csv(RAW / "drugLibTrain_raw.tsv", sep="\t")
test = pd.read_csv(RAW / "drugLibTest_raw.tsv", sep="\t")
joined = lambda frame: frame[review_cols].fillna("").agg(" ".join, axis=1).map(norm)
remove_indices = set(test.index[joined(test).isin(set(joined(train)))])
report = {"purpose": "diagnostic only; headline results remain the unmodified official test split", "exact_combined_review_rows_removed": len(remove_indices), "retained_test_rows": len(test) - len(remove_indices), "models": {}}
for path in RESULTS.glob("predictions_corrected_*.csv"):
    p = pd.read_csv(path)
    retained = p[~p.test_index.isin(remove_indices)]
    report["models"][path.stem.removeprefix("predictions_corrected_")] = {
        "official_test_accuracy": float(accuracy_score(p.true_label, p.predicted_label)),
        "deduplicated_test_accuracy": float(accuracy_score(retained.true_label, retained.predicted_label)),
        "deduplicated_balanced_accuracy": float(balanced_accuracy_score(retained.true_label, retained.predicted_label)),
        "deduplicated_macro_report": classification_report(retained.true_label, retained.predicted_label, labels=labels, target_names=labels, output_dict=True, zero_division=0)["macro avg"],
    }
(RESULTS / "deduplicated_test_diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
