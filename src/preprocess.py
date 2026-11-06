"""Step 3 - find the face in every photo, crop it, and save the crops.

Output:  data/processed/<player>/00001.jpg ...   and   data/processed_manifest.csv
Usage:   python -m src.preprocess [--raw PATH] [--limit N]
"""
import argparse
import hashlib
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

import config
from src.data_utils import discover_dataset, filter_players, slugify, suggest_names
from src.face_embedder import FaceEmbedder


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", help="dataset folder (default: path saved by download_data)")
    parser.add_argument("--limit", type=int, help="only process N images per player (quick test)")
    parser.add_argument("--all-players", action="store_true",
                        help="ignore SELECTED_PLAYERS in config.py and process every player (slow)")
    args = parser.parse_args()

    if args.raw:
        raw = Path(args.raw)
    elif config.RAW_PATH_FILE.exists():
        raw = Path(config.RAW_PATH_FILE.read_text(encoding="utf-8").strip())
    else:
        raise SystemExit("Dataset path unknown. Run `python -m src.download_data` first.")

    df = discover_dataset(raw)
    if df.empty:
        raise SystemExit("No labelled images found. Run `python -m src.inspect_dataset` to see the layout.")
    if config.SELECTED_PLAYERS and not args.all_players:
        all_labels = df["label"].unique()
        df, missing = filter_players(df, config.SELECTED_PLAYERS)
        for name in missing:
            print(f"WARNING: '{name}' not found in the dataset. Did you mean: {suggest_names(name, all_labels)} ?")
        if df.empty:
            raise SystemExit("None of SELECTED_PLAYERS were found. Check the names in config.py.")
        print(f"Using {df['label'].nunique()} selected players ({len(df)} images). "
              f"Edit SELECTED_PLAYERS in config.py to change, or use --all-players (slow).")
    if args.limit:
        df = df.groupby("label", group_keys=False).head(args.limit)

    # remove byte-identical files (exact duplicates)
    df["hash"] = [md5(p) for p in tqdm(df["path"], desc="Checking duplicates")]
    before = len(df)
    df = df.drop_duplicates(subset=["label", "hash"]).reset_index(drop=True)
    print(f"Removed {before - len(df)} exact duplicate files.")

    if config.PROCESSED_DIR.exists():
        shutil.rmtree(config.PROCESSED_DIR)       # derived data - safe to rebuild
    config.PROCESSED_DIR.mkdir(parents=True)

    embedder = FaceEmbedder(load_recognizer=False)
    rows, stats = [], {}
    counters = {}
    for rec in tqdm(df.itertuples(index=False), total=len(df), desc="Detecting faces"):
        s = stats.setdefault(rec.label, {"player": rec.label, "raw_images": 0, "face_found": 0, "no_face": 0, "unreadable": 0})
        s["raw_images"] += 1
        try:
            img = Image.open(rec.path)
            face, prob = embedder.detect_and_crop(img)
        except Exception as exc:                     # corrupt image etc.
            s["unreadable"] += 1
            continue
        if face is None:
            s["no_face"] += 1
            continue
        slug = slugify(rec.label)
        counters[slug] = counters.get(slug, 0) + 1
        out = config.PROCESSED_DIR / slug / f"{counters[slug]:05d}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        face.save(out, quality=95)
        s["face_found"] += 1
        rows.append({
            "crop_path": out.relative_to(config.ROOT).as_posix(),
            "label": rec.label,
            "source_path": rec.path,
            "det_prob": round(prob, 4),
        })

    pd.DataFrame(rows).to_csv(config.MANIFEST_FILE, index=False)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    summary = pd.DataFrame(stats.values()).sort_values("face_found", ascending=False)
    summary.to_csv(config.REPORTS_DIR / "preprocess_summary.csv", index=False)
    print("\n", summary.to_string(index=False))
    print(f"\nSaved {len(rows)} face crops -> {config.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
