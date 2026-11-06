"""Step 5 - train and evaluate the classifiers on the face embeddings.

Compares SVM, Logistic Regression and KNN (hyper-parameters chosen with
cross-validation on the TRAIN part only), reports results on the held-out
TEST part, and saves the best probabilistic model for the app.
"""
from __future__ import annotations

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import config
from src.data_utils import stratified_split


def candidates():
    # (estimator, parameter grid, can be deployed?)  KNN is compared but not deployed:
    # its probabilities are coarse, so an "Unknown" threshold would not work well.
    return {
        "SVM": (SVC(random_state=config.SEED), {"kernel": ["linear", "rbf"], "C": [0.1, 1, 10, 100]}, True),
        "Logistic Regression": (LogisticRegression(max_iter=3000), {"C": [0.1, 1, 10, 100]}, True),
        "KNN": (KNeighborsClassifier(metric="cosine", algorithm="brute"),
                {"n_neighbors": [1, 3, 5], "weights": ["uniform", "distance"]}, False),
    }


def metrics_row(name, y_true, y_pred, extra=None):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    row = {"model": name, "accuracy": accuracy_score(y_true, y_pred),
           "precision_macro": p, "recall_macro": r, "f1_macro": f}
    row.update(extra or {})
    return row


def save_confusion_matrix(y_true, y_pred, classes, path, title):
    size = max(6, 0.9 * len(classes))
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=classes, xticks_rotation=45,
                                            cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_training(X, y, paths, models_dir=config.MODELS_DIR, reports_dir=config.REPORTS_DIR,
                 test_size=config.TEST_SIZE, seed=config.SEED, verbose=True):
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    X, y, paths = np.asarray(X), np.asarray(y), np.asarray(paths)

    counts = pd.Series(y).value_counts()
    if counts.min() < 6:
        raise ValueError(f"Every player needs at least 6 images; too few: {counts[counts < 6].to_dict()}")
    train_idx, test_idx = stratified_split(y, test_size, seed)
    Xtr, Xte, ytr, yte = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    classes = sorted(set(y))
    if verbose:
        print(f"Players: {len(classes)} | train images: {len(ytr)} | test images: {len(yte)}")

    n_splits = int(min(5, pd.Series(ytr).value_counts().min()))
    cv = StratifiedKFold(n_splits=max(n_splits, 2), shuffle=True, random_state=seed)

    results, fitted = [], {}
    for name, (est, grid, deployable) in candidates().items():
        search = GridSearchCV(est, grid, cv=cv, scoring="accuracy", n_jobs=-1)
        search.fit(Xtr, ytr)
        best = clone(est).set_params(**search.best_params_)
        if name == "SVM":
            best.set_params(probability=True)          # needed for confidence scores
        best.fit(Xtr, ytr)
        pred = best.predict(Xte)
        row = metrics_row(name, yte, pred, {"cv_accuracy": search.best_score_,
                                            "best_params": json.dumps(search.best_params_),
                                            "deployable": deployable})
        results.append(row)
        fitted[name] = best
        if verbose:
            print(f"{name:20s} CV acc {search.best_score_:.3f} | test acc {row['accuracy']:.3f} | {search.best_params_}")

    res = pd.DataFrame(results)
    res.to_csv(reports_dir / "embedding_results.csv", index=False)

    # choose the deployed model by cross-validation score (never by the test set)
    pool = res[res["deployable"]]
    best_name = pool.sort_values("cv_accuracy", ascending=False).iloc[0]["model"]
    best = fitted[best_name]
    pred = best.predict(Xte)
    proba = best.predict_proba(Xte)
    conf = proba.max(axis=1)

    (reports_dir / "classification_report.txt").write_text(
        f"Model: {best_name}\n\n" + classification_report(yte, pred, zero_division=0), encoding="utf-8")
    save_confusion_matrix(yte, pred, classes, reports_dir / "confusion_matrix.png",
                          f"{best_name} - confusion matrix (test set)")

    wrong = pred != yte
    pd.DataFrame({"crop_path": paths[test_idx][wrong], "true": yte[wrong], "predicted": pred[wrong],
                  "confidence": conf[wrong].round(3)}).to_csv(reports_dir / "misclassified.csv", index=False)

    rows = []
    for thr in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        accepted = conf >= thr
        rows.append({"threshold": thr, "coverage": accepted.mean(),
                     "accuracy_on_accepted": accuracy_score(yte[accepted], pred[accepted]) if accepted.any() else np.nan})
    pd.DataFrame(rows).to_csv(reports_dir / "threshold_table.csv", index=False)

    pd.Series(y).value_counts().sort_values().plot.barh(figsize=(7, max(3, 0.4 * len(classes))), color="#2b6cb0")
    plt.xlabel("images used"); plt.title("Images per player"); plt.tight_layout()
    plt.savefig(reports_dir / "class_distribution.png", dpi=150); plt.close()

    joblib.dump({"model": best, "classes": classes, "model_name": best_name,
                 "threshold": config.UNKNOWN_THRESHOLD,
                 "embedding_model": "facenet-pytorch InceptionResnetV1 (vggface2)"},
                models_dir / "face_classifier.joblib")

    summary = {"deployed_model": best_name, "players": classes, "n_train": int(len(ytr)), "n_test": int(len(yte)),
               "test_accuracy": float(accuracy_score(yte, pred)),
               "test_f1_macro": float(precision_recall_fscore_support(yte, pred, average="macro", zero_division=0)[2])}
    (reports_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if verbose:
        print(f"\nDeployed model: {best_name} | test accuracy {summary['test_accuracy']:.3f} | "
              f"macro F1 {summary['test_f1_macro']:.3f}")
        print(f"Reports saved in {reports_dir}")
    return summary


def main():
    if not config.EMBEDDINGS_FILE.exists():
        raise SystemExit("Run `python -m src.embeddings` first.")
    d = np.load(config.EMBEDDINGS_FILE)
    run_training(d["X"], d["y"], d["paths"])


if __name__ == "__main__":
    main()
