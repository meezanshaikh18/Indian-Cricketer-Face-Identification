"""Step 2 - look at the raw dataset: folder layout and images per player.

Usage:  python -m src.inspect_dataset [optional/path/to/dataset]
"""
import sys
from pathlib import Path

import config
from src.data_utils import discover_dataset, filter_players, list_images, suggest_names


def raw_root() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    if config.RAW_PATH_FILE.exists():
        return Path(config.RAW_PATH_FILE.read_text(encoding="utf-8").strip())
    sys.exit("Dataset path unknown. Run `python -m src.download_data` first.")


def print_tree(root: Path, max_depth: int = 2):
    print(f"\nFolder layout of {root}")
    def walk(folder: Path, depth: int):
        subs = sorted(p for p in folder.iterdir() if p.is_dir())
        files = [p for p in folder.iterdir() if p.is_file()]
        for s in subs[:15]:
            n = len(list_images(s))
            print("  " * depth + f"[{s.name}]  ({n} images)")
            if depth < max_depth:
                walk(s, depth + 1)
        if len(subs) > 15:
            print("  " * depth + f"... and {len(subs) - 15} more folders")
        if files and depth == 0:
            print(f"  ({len(files)} loose files at top level, e.g. {[f.name for f in files[:3]]})")
    walk(root, 0)


def main():
    root = raw_root()
    print_tree(root)
    df = discover_dataset(root)
    print(f"\nTotal images found: {len(list_images(root))}   |   with a player label: {len(df)}")
    counts = df["label"].value_counts()
    print(f"Players found: {len(counts)}\n")
    print(counts.to_string())
    small = counts[counts < config.MIN_IMAGES_PER_CLASS]
    if len(small):
        print(f"\n{len(small)} player(s) have fewer than {config.MIN_IMAGES_PER_CLASS} images and will be dropped "
              f"(change MIN_IMAGES_PER_CLASS in config.py).")
    if config.SELECTED_PLAYERS:
        found, missing = filter_players(df, config.SELECTED_PLAYERS)
        print(f"\nSELECTED_PLAYERS in config.py: {found['label'].nunique()} of {len(config.SELECTED_PLAYERS)} found "
              f"({len(found)} images).")
        for name in missing:
            print(f"  NOT FOUND: '{name}'  -> did you mean {suggest_names(name, counts.index)} ?")
    print("\nIf these player names look wrong, copy this output and ask for help adjusting src/data_utils.py.")


if __name__ == "__main__":
    main()
