"""Dataset discovery, label inference, class selection, splitting, de-duplication."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config


def list_images(root) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in config.IMAGE_EXTS)


def normalize_label(name: str) -> str:
    """'virat_kohli' / 'VIRAT-KOHLI' / 'Virat  Kohli' -> 'Virat Kohli'."""
    name = re.sub(r"[_\-\.]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name.title()


def slugify(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


_TRAILING_NUMBER = re.compile(r"^(.*?)[\s_\-\.]*[\(\[]?\d+[\)\]]?$")


def label_from_filename(stem: str) -> str | None:
    """'virat_kohli_12' -> 'Virat Kohli'; '12345' -> None."""
    match = _TRAILING_NUMBER.match(stem)
    name = match.group(1) if match else stem
    name = normalize_label(name)
    return name or None


def label_from_path(path: Path, root: Path) -> str | None:
    """Use the nearest non-generic parent folder; fall back to the file name.
    A trailing digit in a folder name ("M S Dhoni1") is a dataset quirk and is removed."""
    folders = path.relative_to(root).parts[:-1]
    for part in reversed(folders):
        if part.lower() not in config.GENERIC_FOLDER_NAMES:
            cleaned = re.sub(r"[\s_\-]*\d+$", "", part).strip()
            return normalize_label(cleaned or part)
    return label_from_filename(path.stem)


def _key(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+$", "", name.strip())).lower()


def filter_players(df: pd.DataFrame, selected) -> tuple[pd.DataFrame, list[str]]:
    """Keep only the selected players (case-insensitive). Returns (filtered_df, missing_names)."""
    wanted = {_key(n): n for n in selected}
    keys = df["label"].map(_key)
    mask = keys.isin(wanted)
    missing = [n for k, n in wanted.items() if k not in set(keys[mask])]
    return df[mask].copy(), missing


def suggest_names(name: str, labels, n: int = 3) -> list[str]:
    import difflib
    return difflib.get_close_matches(name, list(labels), n=n, cutoff=0.6)


def discover_dataset(root) -> pd.DataFrame:
    """Return a DataFrame(path, label) for every image under `root`."""
    root = Path(root)
    rows = []
    for p in list_images(root):
        label = label_from_path(p, root)
        if label:
            rows.append({"path": str(p), "label": label})
    return pd.DataFrame(rows, columns=["path", "label"])


def select_classes(labels, min_images: int, max_classes: int | None = None) -> list[str]:
    """Players with >= min_images, optionally only the `max_classes` largest."""
    counts = pd.Series(list(labels)).value_counts()
    counts = counts[counts >= min_images]
    if max_classes:
        counts = counts.head(max_classes)
    return sorted(counts.index.tolist())


def stratified_split(y, test_size: float = config.TEST_SIZE, seed: int = config.SEED):
    """Deterministic stratified split -> (train_idx, test_idx)."""
    idx = np.arange(len(y))
    return train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)


def drop_near_duplicates(X: np.ndarray, y: np.ndarray, threshold: float = config.DEDUP_COSINE) -> np.ndarray:
    """Boolean keep-mask. Within each player, drop photos whose (L2-normalised)
    embedding is almost identical to one already kept. This stops the same
    photo from landing in both the train and test sets."""
    keep = np.zeros(len(X), dtype=bool)
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        sims = X[idx] @ X[idx].T
        kept: list[int] = []
        for i in range(len(idx)):
            if not kept or np.all(sims[i, kept] < threshold):
                kept.append(i)
        keep[idx[kept]] = True
    return keep
