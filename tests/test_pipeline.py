"""Offline tests on synthetic data (no Kaggle, no torch needed)."""
from pathlib import Path

import numpy as np
from PIL import Image

import config
from src import data_utils as du
from src.train import run_training
from src.train_baseline import run_baseline


def _touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8)).save(p)


def test_label_from_class_folders(tmp_path):
    _touch(tmp_path / "Virat_Kohli" / "a.jpg")
    _touch(tmp_path / "ms dhoni" / "b.png")
    df = du.discover_dataset(tmp_path)
    assert set(df["label"]) == {"Virat Kohli", "Ms Dhoni"}


def test_label_from_split_folders(tmp_path):
    _touch(tmp_path / "train" / "Rohit Sharma" / "1.jpg")
    _touch(tmp_path / "test" / "Rohit Sharma" / "2.jpg")
    _touch(tmp_path / "dataset" / "images" / "Rohit Sharma" / "3.jpg")
    assert set(du.discover_dataset(tmp_path)["label"]) == {"Rohit Sharma"}


def test_label_from_flat_filenames(tmp_path):
    _touch(tmp_path / "virat_kohli_12.jpg")
    _touch(tmp_path / "Jasprit Bumrah (3).jpg")
    _touch(tmp_path / "hardik-pandya-7.png")
    _touch(tmp_path / "12345.jpg")                      # no name -> ignored
    assert set(du.discover_dataset(tmp_path)["label"]) == {"Virat Kohli", "Jasprit Bumrah", "Hardik Pandya"}


def test_select_classes_and_split():
    y = np.array(["a"] * 30 + ["b"] * 25 + ["c"] * 5)
    assert du.select_classes(y, 20) == ["a", "b"]
    assert du.select_classes(y, 1, max_classes=1) == ["a"]
    tr, te = du.stratified_split(y[:55][:])
    assert len(set(tr) & set(te)) == 0


def test_drop_near_duplicates():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(5, 64))
    X = np.vstack([base, base + 1e-4 * rng.normal(size=base.shape)])   # every vector duplicated
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = np.array(["a"] * 10)
    assert du.drop_near_duplicates(X, y, 0.98).sum() == 5


def _fake_embeddings(n_per=40, classes=("A", "B", "C", "D"), dim=512, noise=0.6, seed=0):
    rng = np.random.default_rng(seed)
    X, y, paths = [], [], []
    for ci, c in enumerate(classes):
        centre = rng.normal(size=dim)
        v = centre + noise * rng.normal(size=(n_per, dim)) * np.sqrt(dim) / np.sqrt(dim)
        X.append(v); y += [c] * n_per; paths += [f"x/{c}/{i}.jpg" for i in range(n_per)]
    X = np.vstack(X)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X, np.array(y), np.array(paths)


def test_training_end_to_end(tmp_path):
    X, y, paths = _fake_embeddings()
    summary = run_training(X, y, paths, models_dir=tmp_path / "m", reports_dir=tmp_path / "r", verbose=False)
    assert summary["test_accuracy"] > 0.9
    assert (tmp_path / "m" / "face_classifier.joblib").exists()
    for name in ["confusion_matrix.png", "classification_report.txt", "threshold_table.csv", "metrics.json"]:
        assert (tmp_path / "r" / name).exists()
    import joblib
    bundle = joblib.load(tmp_path / "m" / "face_classifier.joblib")
    assert bundle["model_name"] in {"SVM", "Logistic Regression"}
    assert bundle["model"].predict_proba(X[:3]).shape == (3, 4)


def test_training_rejects_tiny_class(tmp_path):
    X, y, paths = _fake_embeddings(n_per=4)
    try:
        run_training(X, y, paths, models_dir=tmp_path / "m", reports_dir=tmp_path / "r", verbose=False)
    except ValueError as e:
        assert "at least 6" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_baseline_end_to_end(tmp_path):
    rng = np.random.default_rng(1)
    size, imgs, y = 16, [], []
    for ci, c in enumerate("ABC"):
        pattern = rng.random((size, size))
        for _ in range(30):
            imgs.append((pattern + 0.15 * rng.random((size, size))).ravel()); y.append(c)
    row = run_baseline(np.array(imgs), np.array(y), size=size, reports_dir=tmp_path, verbose=False)
    assert row["accuracy"] > 0.9
    assert (tmp_path / "eigenfaces.png").exists() and (tmp_path / "baseline_confusion_matrix.png").exists()


def test_trailing_digit_in_folder_name_and_player_filter(tmp_path):
    _touch(tmp_path / "indian cricketer" / "M S Dhoni1" / "a.jpg")
    _touch(tmp_path / "indian cricketer" / "Rahul Dravid2" / "b.jpg")
    _touch(tmp_path / "indian cricketer" / "Kl Rahul" / "c.jpg")
    _touch(tmp_path / "indian cricketer" / "Someone Else" / "d.jpg")
    df = du.discover_dataset(tmp_path)
    assert set(df["label"]) == {"M S Dhoni", "Rahul Dravid", "Kl Rahul", "Someone Else"}
    kept, missing = du.filter_players(df, ["m s dhoni", "Rahul  Dravid", "Virat Kohli"])
    assert set(kept["label"]) == {"M S Dhoni", "Rahul Dravid"}
    assert missing == ["Virat Kohli"]
    assert "Kl Rahul" in du.suggest_names("KL Rahul", df["label"])
