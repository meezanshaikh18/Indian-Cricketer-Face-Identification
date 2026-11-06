"""Step 6 (optional but recommended) - classic baseline: grayscale pixels -> PCA ("eigenfaces") -> SVM.

Uses exactly the same photos and train/test split as src.train, so the two
approaches can be compared fairly in the report.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import config
from src.data_utils import stratified_split


def load_gray(paths, size=config.BASELINE_IMAGE_SIZE):
    arrs = [np.asarray(Image.open(config.ROOT / p).convert("L").resize((size, size)), dtype="float32").ravel() / 255.0
            for p in paths]
    return np.stack(arrs)


def run_baseline(X, y, size=config.BASELINE_IMAGE_SIZE, reports_dir=config.REPORTS_DIR,
                 test_size=config.TEST_SIZE, seed=config.SEED, verbose=True):
    reports_dir.mkdir(parents=True, exist_ok=True)
    X, y = np.asarray(X), np.asarray(y)
    train_idx, test_idx = stratified_split(y, test_size, seed)
    Xtr, Xte, ytr, yte = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    classes = sorted(set(y))

    n_splits = max(int(min(5, pd.Series(ytr).value_counts().min())), 2)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # PCA can't have more components than samples in one CV training fold or than pixels
    fold_size = int(len(ytr) * (n_splits - 1) / n_splits) - 1
    limit = min(fold_size, X.shape[1])
    n_comp = [n for n in (30, 60, 100) if n < limit] or [max(2, min(limit - 1, 10))]
    pipe = Pipeline([("pca", PCA(whiten=True, random_state=seed)), ("svm", SVC(kernel="rbf"))])
    grid = {"pca__n_components": n_comp, "svm__C": [1, 10, 100], "svm__gamma": ["scale", 0.001, 0.01]}
    search = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=-1).fit(Xtr, ytr)
    pred = search.predict(Xte)

    p, r, f, _ = precision_recall_fscore_support(yte, pred, average="macro", zero_division=0)
    row = {"model": "Baseline: PCA + SVM (pixels)", "accuracy": accuracy_score(yte, pred), "precision_macro": p,
           "recall_macro": r, "f1_macro": f, "cv_accuracy": search.best_score_,
           "best_params": json.dumps({k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in search.best_params_.items()}),
           "deployable": False}

    (reports_dir / "baseline_classification_report.txt").write_text(
        classification_report(yte, pred, zero_division=0), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(classes)),) * 2)
    ConfusionMatrixDisplay.from_predictions(yte, pred, labels=classes, xticks_rotation=45, cmap="Oranges",
                                            ax=ax, colorbar=False)
    ax.set_title("Baseline PCA + SVM - confusion matrix (test set)")
    fig.tight_layout(); fig.savefig(reports_dir / "baseline_confusion_matrix.png", dpi=150); plt.close(fig)

    comps = search.best_estimator_.named_steps["pca"].components_
    n_show = min(12, len(comps))
    fig, axes = plt.subplots(2, 6, figsize=(12, 4.5))
    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i < n_show:
            ax.imshow(comps[i].reshape(size, size), cmap="gray"); ax.set_title(f"PC {i + 1}", fontsize=9)
    fig.suptitle("Eigenfaces (top principal components)"); fig.tight_layout()
    fig.savefig(reports_dir / "eigenfaces.png", dpi=150); plt.close(fig)

    out = pd.DataFrame([row])
    out.to_csv(reports_dir / "baseline_results.csv", index=False)
    emb = reports_dir / "embedding_results.csv"
    if emb.exists():
        pd.concat([pd.read_csv(emb), out], ignore_index=True).to_csv(reports_dir / "model_comparison.csv", index=False)
    if verbose:
        print(f"Baseline PCA+SVM: CV acc {search.best_score_:.3f} | test acc {row['accuracy']:.3f} | {search.best_params_}")
    return row


def main():
    if not config.EMBEDDINGS_FILE.exists():
        raise SystemExit("Run `python -m src.embeddings` first.")
    d = np.load(config.EMBEDDINGS_FILE)
    run_baseline(load_gray(d["paths"]), d["y"])
    print("Comparison table: reports/model_comparison.csv")


if __name__ == "__main__":
    main()
