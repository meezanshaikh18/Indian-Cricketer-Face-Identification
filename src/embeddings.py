"""Step 4 - turn every face crop into a 512-number FaceNet embedding.

Output: data/embeddings.npz  (X = embeddings, y = player names, paths = crop files)
"""
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import config
from src.data_utils import drop_near_duplicates, select_classes
from src.face_embedder import FaceEmbedder


def main():
    if not config.MANIFEST_FILE.exists():
        raise SystemExit("Run `python -m src.preprocess` first.")
    manifest = pd.read_csv(config.MANIFEST_FILE)

    keep = select_classes(manifest["label"], config.MIN_IMAGES_PER_CLASS, config.MAX_CLASSES)
    dropped = sorted(set(manifest["label"]) - set(keep))
    if dropped:
        print(f"Dropping players with < {config.MIN_IMAGES_PER_CLASS} usable faces: {dropped}")
    manifest = manifest[manifest["label"].isin(keep)].reset_index(drop=True)
    if manifest.empty:
        raise SystemExit("No player has enough images. Lower MIN_IMAGES_PER_CLASS in config.py.")

    embedder = FaceEmbedder()
    faces = [Image.open(config.ROOT / p).convert("RGB") for p in tqdm(manifest["crop_path"], desc="Loading crops")]
    X = embedder.embed(faces)
    y = manifest["label"].to_numpy().astype(str)
    paths = manifest["crop_path"].to_numpy().astype(str)

    mask = drop_near_duplicates(X, y, config.DEDUP_COSINE)
    print(f"Removed {int((~mask).sum())} near-duplicate photos (same player, almost identical face).")
    for label in np.unique(y):
        total, removed = int((y == label).sum()), int(((y == label) & ~mask).sum())
        if removed / total > 0.3:
            print(f"  WARNING: {removed}/{total} photos of {label} were treated as duplicates. "
                  f"If that looks too aggressive, raise DEDUP_COSINE in config.py (e.g. 0.99).")
    X, y, paths = X[mask], y[mask], paths[mask]

    # a player may have fallen below the minimum after de-duplication
    keep2 = select_classes(y, config.MIN_IMAGES_PER_CLASS // 2 or 1)
    m2 = np.isin(y, keep2)
    X, y, paths = X[m2], y[m2], paths[m2]
    if len(y) == 0:
        raise SystemExit("No images left after cleaning. Lower MIN_IMAGES_PER_CLASS or raise DEDUP_COSINE in config.py.")

    np.savez_compressed(config.EMBEDDINGS_FILE, X=X, y=y, paths=paths)
    print("\nImages per player after cleaning:")
    print(pd.Series(y).value_counts().to_string())
    print(f"\nSaved {X.shape[0]} embeddings of size {X.shape[1]} -> {config.EMBEDDINGS_FILE}")


if __name__ == "__main__":
    main()
