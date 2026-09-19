"""Faithful reconstruction of Thesis Report appendix Listings 1--6.

Only file-format/path fixes and reproducibility plumbing differ from the listings.
Use --corrected to apply the explicitly documented Phase 2 reporting fixes.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import spacy
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, roc_curve)
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
LABELS = ["No Side Effects", "Mild Side Effects", "Severe Side Effects"]
CAT = ["effectiveness", "rating", "condition", "urlDrugName"]
SEED = 42


def bin_side_effects(effect: object) -> str:
    """The appendix's 3-class conversion, retained verbatim in substance."""
    value = str(effect)
    if "No Side Effects" in value:
        return "No Side Effects"
    if "Mild Side Effects" in value or "Moderate Side Effects" in value:
        return "Mild Side Effects"
    return "Severe Side Effects"


def prepare(df: pd.DataFrame, nlp) -> pd.DataFrame:
    df = df.copy()
    df["reviews_for_features"] = (
        df["benefitsReview"].fillna("") + " " + df["commentsReview"].fillna("")
        + " " + df["sideEffectsReview"].fillna("")
    )
    # Equivalent to Listing 1's nlp(str(text).lower()) per row, but batched.
    docs = nlp.pipe(df["reviews_for_features"].astype(str).str.lower(), batch_size=64)
    df["processed_reviews"] = [
        " ".join(token.lemma_ for token in doc if token.is_alpha and not token.is_stop)
        for doc in docs
    ]
    df["side_effects_binned"] = df["sideEffects"].apply(bin_side_effects)
    return df


def write_json(name: str, obj) -> None:
    (RESULTS / name).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def dataset_audit(train: pd.DataFrame, test: pd.DataFrame) -> None:
    audit = {
        "source": "UCI Drug Reviews (Druglib.com), ID 461",
        "train_rows": len(train), "test_rows": len(test), "total_rows": len(train) + len(test),
        "train_columns": list(train.columns), "test_columns": list(test.columns),
        "train_sideEffects_counts": train["sideEffects"].value_counts(dropna=False).to_dict(),
        "test_sideEffects_counts": test["sideEffects"].value_counts(dropna=False).to_dict(),
        "train_null_counts": train.isna().sum().to_dict(),
        "test_null_counts": test.isna().sum().to_dict(),
    }
    write_json("dataset_audit.json", audit)


def integrity_checks(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Exact overlap on raw review fields/full rows; high-Jaccard near text matches."""
    review_cols = ["benefitsReview", "commentsReview", "sideEffectsReview"]
    def norm(x):
        return " ".join(str(x).lower().split())
    def joined(frame):
        return frame[review_cols].fillna("").agg(" ".join, axis=1).map(norm)
    train_combined, test_combined = joined(train), joined(test)
    train_set = set(train_combined)
    exact_combined = int(test_combined.isin(train_set).sum())
    raw_by_field = {c: int(test[c].fillna("").map(norm).isin(set(train[c].fillna("").map(norm))).sum()) for c in review_cols}
    full_cols = list(train.columns)
    train_rows = set(train[full_cols].fillna("<NA>").astype(str).agg("\x1f".join, axis=1))
    test_rows = test[full_cols].fillna("<NA>").astype(str).agg("\x1f".join, axis=1)
    full_overlap = int(test_rows.isin(train_rows).sum())

    # Candidate generation by rare-enough tokens avoids a 3,107 x 1,036 all-pairs scan.
    inverted = defaultdict(set)
    train_tokens = []
    for i, text in enumerate(train_combined):
        tokens = set(text.split())
        train_tokens.append(tokens)
        for token in tokens:
            if len(token) > 3:
                inverted[token].add(i)
    near = []
    for ti, text in enumerate(test_combined):
        tokens = set(text.split())
        candidates = set()
        for token in tokens:
            if len(token) > 3:
                candidates.update(inverted.get(token, ()))
        best_i, best_j = None, 0.0
        for tr_i in candidates:
            union = tokens | train_tokens[tr_i]
            if union:
                score = len(tokens & train_tokens[tr_i]) / len(union)
                if score > best_j:
                    best_i, best_j = tr_i, score
        if best_j >= 0.90:
            near.append({"test_index": int(test.index[ti]), "train_index": int(train.index[best_i]), "jaccard": best_j})
    pd.DataFrame(near).to_csv(RESULTS / "near_duplicate_pairs_jaccard_090.csv", index=False)
    write_json("integrity_checks.json", {
        "exact_combined_review_overlap_test_rows": exact_combined,
        "exact_overlap_by_raw_review_field_test_rows": raw_by_field,
        "exact_full_row_overlap_test_rows": full_overlap,
        "near_duplicate_definition": "token-set Jaccard >= 0.90 on normalized combined review text; candidate search shares a token >3 chars",
        "near_duplicate_test_rows": len(near),
        "note": "This does not make any de-duplicated result a headline result; it is a diagnostic only.",
    })


def save_cm(cm, name, suffix):
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(RESULTS / f"confusion_matrix_{suffix}_{name}.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS, ax=ax)
    ax.set(xlabel="Predicted label", ylabel="True label", title=f"Confusion Matrix: {name}")
    fig.tight_layout(); fig.savefig(FIGURES / f"confusion_matrix_{suffix}_{name}.png", dpi=300); plt.close(fig)


def roc_outputs(models, X_train, y_train, X_test, y_test, corrected: bool):
    summary = {}
    suffix = "corrected" if corrected else "original"
    for name, model in models.items():
        roc_model = model
        if isinstance(model, SVC):
            roc_model = SVC(C=model.C, kernel=model.kernel, class_weight=model.class_weight,
                            random_state=model.random_state, gamma=model.gamma, probability=True)
            roc_model.fit(X_train, y_train)
        scores = roc_model.predict_proba(X_test)
        classes = list(roc_model.classes_)
        # The original report bins in LABELS order but pairs column i, which is wrong when classes sort differently.
        if corrected:
            y_bin = label_binarize(y_test, classes=classes)
            display_labels = classes
        else:
            y_bin = label_binarize(y_test, classes=LABELS)
            display_labels = LABELS
        fpr, tpr, areas = {}, {}, {}
        for i in range(y_bin.shape[1]):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], scores[:, i])
            areas[display_labels[i]] = auc(fpr[i], tpr[i])
        micro_fpr, micro_tpr, _ = roc_curve(y_bin.ravel(), scores.ravel())
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(y_bin.shape[1])]))
        mean_tpr = sum(np.interp(all_fpr, fpr[i], tpr[i]) for i in range(y_bin.shape[1])) / y_bin.shape[1]
        areas["micro"] = auc(micro_fpr, micro_tpr)
        areas["macro"] = auc(all_fpr, mean_tpr)
        summary[name] = {k: float(v) for k, v in areas.items()}
        fig, ax = plt.subplots(figsize=(7, 5))
        for i, label in enumerate(display_labels):
            ax.plot(fpr[i], tpr[i], lw=2, label=f"{label} (AUC={areas[label]:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC: {name}")
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIGURES / f"roc_{suffix}_{name}.png", dpi=300); plt.close(fig)
    write_json(f"roc_auc_{suffix}.json", summary)


def learning_outputs(models, X, y):
    data = {}
    for name, model in models.items():
        sizes, train_scores, cv_scores = learning_curve(model, X, y, cv=3, n_jobs=-1, train_sizes=np.linspace(.1, 1, 5), scoring="accuracy")
        data[name] = {"train_sizes": sizes.tolist(), "train_mean": train_scores.mean(1).tolist(), "cv_mean": cv_scores.mean(1).tolist()}
        fig, ax = plt.subplots(figsize=(7, 5)); ax.plot(sizes, train_scores.mean(1), "o-", label="Training"); ax.plot(sizes, cv_scores.mean(1), "o-", label="CV")
        ax.set(xlabel="Training examples", ylabel="Accuracy", title=f"Learning Curve: {name}"); ax.legend(); ax.grid(); fig.tight_layout(); fig.savefig(FIGURES / f"learning_curve_{name}.png", dpi=300); plt.close(fig)
    write_json("learning_curves.json", data)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--corrected", action="store_true"); parser.add_argument("--skip-learning", action="store_true")
    args = parser.parse_args(); RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); os.environ["PYTHONHASHSEED"] = str(SEED)
    train = pd.read_csv(RAW / "drugLibTrain_raw.tsv", sep="\t")
    test = pd.read_csv(RAW / "drugLibTest_raw.tsv", sep="\t")
    dataset_audit(train, test)
    try: nlp = spacy.load("en_core_web_sm")
    except OSError as exc: raise SystemExit("Missing spaCy model. Install it with: python -m spacy download en_core_web_sm") from exc
    started = time.time(); train, test = prepare(train, nlp), prepare(test, nlp)
    vectorizer = TfidfVectorizer(max_features=7500, ngram_range=(1, 2), min_df=5, max_df=.95)
    X_train_text = vectorizer.fit_transform(train.processed_reviews); X_test_text = vectorizer.transform(test.processed_reviews)
    encoder = OneHotEncoder(handle_unknown="ignore")
    X_train_cat = encoder.fit_transform(train[CAT].astype(str)); X_test_cat = encoder.transform(test[CAT].astype(str))
    X_train, X_test = hstack([X_train_text, X_train_cat]), hstack([X_test_text, X_test_cat])
    y_train, y_test = train.side_effects_binned, test.side_effects_binned
    grids = {
        "Tuned Logistic Regression": GridSearchCV(LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000), {"C":[1,10,20], "solver":["saga"], "penalty":["l1","l2"]}, cv=3, n_jobs=-1, scoring="accuracy"),
        "Tuned RandomForest": GridSearchCV(RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1), {"n_estimators":[200,300], "max_depth":[20,30], "min_samples_split":[2,5], "min_samples_leaf":[1,2]}, cv=3, n_jobs=-1, scoring="accuracy"),
        "Tuned SVC": GridSearchCV(SVC(random_state=42, class_weight="balanced", gamma="scale"), {"C":[1,10,50], "kernel":["linear","rbf"]}, cv=3, n_jobs=-1, scoring="accuracy"),
    }
    suffix = "corrected" if args.corrected else "original"
    models, output = {}, {"corrected": args.corrected, "feature_matrix_shape": list(X_train.shape), "test_feature_matrix_shape": list(X_test.shape), "train_class_counts": y_train.value_counts().to_dict(), "test_class_counts": y_test.value_counts().to_dict()}
    for name, grid in grids.items():
        grid.fit(X_train, y_train); model = grid.best_estimator_; models[name] = model; pred = model.predict(X_test)
        # Listing 3 does not pass labels, so its target_names attach to alphabetical classes.
        report = classification_report(y_test, pred, labels=LABELS if args.corrected else None, target_names=LABELS, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, pred, labels=LABELS)
        output[name] = {"best_params": grid.best_params_, "best_cv_accuracy": float(grid.best_score_), "test_accuracy": float(accuracy_score(y_test, pred)), "classification_report": report, "confusion_matrix": cm.tolist()}
        pd.DataFrame({"test_index": test.index, "true_label": y_test, "predicted_label": pred}).to_csv(
            RESULTS / f"predictions_{suffix}_{name.replace(' ', '_')}.csv", index=False
        )
        save_cm(cm, name.replace(" ", "_"), "corrected" if args.corrected else "original")
    output["runtime_seconds"] = time.time() - started
    output["versions"] = {"python": sys.version, "platform": platform.platform(), "pandas": pd.__version__, "sklearn": sklearn.__version__, "spacy": spacy.__version__}
    write_json(f"pipeline_{suffix}_results.json", output)
    (RESULTS / "env.txt").write_text("\n".join(f"{k}: {v}" for k,v in output["versions"].items()) + "\n", encoding="utf-8")
    roc_outputs(models, X_train, y_train, X_test, y_test, args.corrected)
    if not args.skip_learning:
        learning_outputs(models, X_train, y_train)
    integrity_checks(train, test)

if __name__ == "__main__": main()
