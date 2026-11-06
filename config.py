"""Central settings for the Indian Cricketer Face Identification project.

Change values here instead of editing the scripts.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

KAGGLE_DATASET = "aviratgupta/indian-cricketers-dataset"
RAW_PATH_FILE = DATA_DIR / "raw_dataset_path.txt"   # written by src.download_data
PROCESSED_DIR = DATA_DIR / "processed"              # cropped faces, one folder per player
MANIFEST_FILE = DATA_DIR / "processed_manifest.csv"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npz"
MODEL_FILE = MODELS_DIR / "face_classifier.joblib"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Folder names that describe a split / container, not a player.
GENERIC_FOLDER_NAMES = {
    "train", "test", "val", "valid", "validation", "training", "testing",
    "images", "image", "imgs", "img", "photos", "faces", "data", "dataset", "raw",
}

# ---- Face detection / embedding -------------------------------------------
FACE_SIZE = 160            # crop size expected by the FaceNet model
FACE_MARGIN = 20           # extra pixels around the detected face
MIN_FACE_SIZE = 40         # ignore faces smaller than this (pixels)
MIN_DETECTION_PROB = 0.90  # ignore low-confidence detections
MAX_IMAGE_SIDE = 1280      # large photos are shrunk before detection (speed)

# ---- Which players to use --------------------------------------------------
# The dataset has 324 players (~25 photos each). Training on all of them is slow and very
# hard, so by default we use these 20 well-known players. Edit this list to choose others
# (names are matched ignoring case and any trailing digit in the folder name), or set it to
# None to use every player (very slow: run `python -m src.preprocess --all-players`).
SELECTED_PLAYERS = [
    "Virat Kohli", "Rohit Sharma", "M S Dhoni", "Sachin Tendulkar", "Jasprit Bumrah",
    "Hardik Pandya", "Rishabh Pant", "Kl Rahul", "Shubman Gill", "Ravindra Jadeja",
    "Ravichandran Ashwin", "Suryakumar Yadav", "Shikhar Dhawan", "Yuvraj Singh",
    "Sourav Ganguly", "Rahul Dravid", "Virender Sehwag", "Mohammed Shami",
    "Mohammed Siraj", "Yuzvendra Chahal",
]
MIN_IMAGES_PER_CLASS = 15  # players with fewer usable face crops are dropped
MAX_CLASSES = None         # e.g. 8 = keep only the 8 players with most images; None = all

# ---- Training ---------------------------------------------------------------
TEST_SIZE = 0.2
SEED = 42
DEDUP_COSINE = 0.98        # near-identical photos of the same player are removed
UNKNOWN_THRESHOLD = 0.50   # below this confidence the app answers "Unknown"

# ---- PCA + SVM baseline ------------------------------------------------------
BASELINE_IMAGE_SIZE = 64
