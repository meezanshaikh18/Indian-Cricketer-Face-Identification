"""Run the whole pipeline:  python run_all.py   (add --skip-download if data is already downloaded)"""
import subprocess
import sys

steps = [
    ("Download dataset", "src.download_data"),
    ("Inspect dataset", "src.inspect_dataset"),
    ("Detect + crop faces", "src.preprocess"),
    ("Compute face embeddings", "src.embeddings"),
    ("Train + evaluate classifiers", "src.train"),
    ("Train PCA + SVM baseline", "src.train_baseline"),
]
skip = "--skip-download" in sys.argv
for title, module in steps:
    if skip and module == "src.download_data":
        continue
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    code = subprocess.call([sys.executable, "-m", module])
    if code != 0:
        sys.exit(f"Step failed: {module}")
print("\nAll done. Launch the app with:  streamlit run app.py")
