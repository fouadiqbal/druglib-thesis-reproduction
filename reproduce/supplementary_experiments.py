"""Clearly labelled supplementary analyses for the official UCI split.

The headline protocol is unchanged. This script adds fold-contained feature fitting,
additional test metrics, bootstrap intervals, McNemar's test, and a small ablation.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from scipy.stats import binomtest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, matthews_corrcoef, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
LABELS = ["No Side Effects", "Mild Side Effects", "Severe Side Effects"]
CAT = ["effectiveness", "rating", "condition", "urlDrugName"]
SEED = 42


def bin_effect(value: object) -> str:
    value = str(value)
    if "No Side Effects" in value:
        return "No Side Effects"
    if "Mild Side Effects" in value or "Moderate Side Effects" in value:
        return "Mild Side Effects"
    return "Severe Side Effects"


def prepare(df: pd.DataFrame, nlp) -> pd.DataFrame:
    out = df.copy()
    text = out["benefitsReview"].fillna("") + " " + out["commentsReview"].fillna("") + " " + out["sideEffectsReview"].fillna("")
    docs = nlp.pipe(text.astype(str).str.lower(), batch_size=64)
    out["processed_reviews"] = [" ".join(t.lemma_ for t in d if t.is_alpha and not t.is_stop) for d in docs]
    out["target"] = out["sideEffects"].apply(bin_effect)
    return out


def estimator(name: str):
    if name == "Logistic Regression":
        return LogisticRegression(C=1, solver="saga", penalty="l2", random_state=SEED, class_weight="balanced", max_iter=1000)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, max_depth=30, min_samples_split=2, min_samples_leaf=2, random_state=SEED, class_weight="balanced", n_jobs=-1)
    return SVC(C=10, kernel="rbf", gamma="scale", random_state=SEED, class_weight="balanced")


def full_pipeline():
    return ColumnTransformer([
        ("text", TfidfVectorizer(max_features=7500, ngram_range=(1, 2), min_df=5, max_df=.95), "processed_reviews"),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])


def text_pipeline():
    return ColumnTransformer([("text", TfidfVectorizer(max_features=7500, ngram_range=(1, 2), min_df=5, max_df=.95), "processed_reviews")])


def cat_pipeline():
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])


def main() -> None:
    random.seed(SEED); np.random.seed(SEED); RESULTS.mkdir(exist_ok=True)
    nlp = spacy.load("en_core_web_sm")
    train = prepare(pd.read_csv(RAW / "drugLibTrain_raw.tsv", sep="\t"), nlp)
    test = prepare(pd.read_csv(RAW / "drugLibTest_raw.tsv", sep="\t"), nlp)
    y_train, y_test = train["target"], test["target"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Supplementary A: feature fitting is inside each fold.
    cv_rows = []
    for name in ["Logistic Regression", "Random Forest", "SVC"]:
        pipe = Pipeline([("features", full_pipeline()), ("model", estimator(name))])
        scores = cross_validate(pipe, train, y_train, cv=cv, scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"}, n_jobs=1)
        cv_rows.append({"model": name, "accuracy_mean": float(scores["test_accuracy"].mean()), "accuracy_std": float(scores["test_accuracy"].std(ddof=1)), "macro_f1_mean": float(scores["test_macro_f1"].mean()), "macro_f1_std": float(scores["test_macro_f1"].std(ddof=1))})
    pd.DataFrame(cv_rows).to_csv(RESULTS / "supplementary_5fold_pipeline_cv.csv", index=False)

    # Supplementary B: extra metrics from the official test split.
    extra = {}
    predictions = {}
    for name in ["Logistic Regression", "Random Forest", "SVC"]:
        path = RESULTS / f"predictions_corrected_Tuned_{'RandomForest' if name == 'Random Forest' else ('Logistic_Regression' if name == 'Logistic Regression' else 'SVC')}.csv"
        pred = pd.read_csv(path)["predicted_label"]
        predictions[name] = pred.to_numpy()
        extra[name] = {"accuracy": float(accuracy_score(y_test, pred)), "macro_f1": float(f1_score(y_test, pred, labels=LABELS, average="macro")), "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)), "matthews_corrcoef": float(matthews_corrcoef(y_test, pred)), "classification_report": classification_report(y_test, pred, labels=LABELS, target_names=LABELS, output_dict=True, zero_division=0)}
    (RESULTS / "supplementary_test_metrics.json").write_text(json.dumps(extra, indent=2, default=float), encoding="utf-8")

    # Supplementary C: nonparametric bootstrap 95% accuracy intervals.
    rng = np.random.default_rng(SEED); boot = {}
    for name, pred in predictions.items():
        values = np.empty(2000)
        for i in range(len(values)):
            idx = rng.integers(0, len(y_test), len(y_test)); values[i] = accuracy_score(y_test.to_numpy()[idx], pred[idx])
        boot[name] = {"accuracy": float(accuracy_score(y_test, pred)), "bootstrap_replicates": 2000, "ci95_percentile": [float(np.quantile(values, .025)), float(np.quantile(values, .975))]}
    (RESULTS / "supplementary_bootstrap_accuracy.json").write_text(json.dumps(boot, indent=2), encoding="utf-8")

    # Supplementary D: exact McNemar test for SVC versus Random Forest.
    svc, rf = predictions["SVC"], predictions["Random Forest"]; truth = y_test.to_numpy()
    svc_correct, rf_correct = svc == truth, rf == truth
    b = int(np.sum(svc_correct & ~rf_correct)); c = int(np.sum(~svc_correct & rf_correct))
    mcnemar = {"svc_correct_rf_wrong": b, "svc_wrong_rf_correct": c, "exact_binomial_pvalue_two_sided": float(binomtest(min(b, c), n=b + c, p=.5).pvalue) if b + c else 1.0}
    (RESULTS / "supplementary_mcnemar_svc_vs_rf.json").write_text(json.dumps(mcnemar, indent=2), encoding="utf-8")

    # Supplementary E: text-only, categorical-only, and combined ablation.
    ablation = []
    for feature_name, transformer in [("text_only", text_pipeline()), ("categorical_only", cat_pipeline()), ("text_plus_categorical", full_pipeline())]:
        for name in ["Logistic Regression", "Random Forest", "SVC"]:
            pipe = Pipeline([("features", transformer), ("model", estimator(name))]); started = time.time(); pipe.fit(train, y_train); pred = pipe.predict(test)
            ablation.append({"feature_set": feature_name, "model": name, "accuracy": float(accuracy_score(y_test, pred)), "macro_f1": float(f1_score(y_test, pred, labels=LABELS, average="macro")), "runtime_seconds": time.time() - started})
    pd.DataFrame(ablation).to_csv(RESULTS / "supplementary_ablation.csv", index=False)
    print(pd.DataFrame(cv_rows)); print(pd.DataFrame(ablation))


if __name__ == "__main__":
    main()
