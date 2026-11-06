"""Identify the cricketer in one image.

Usage:  python -m src.predict photo.jpg [--threshold 0.5]
"""
import argparse

from PIL import Image

from src.face_embedder import FaceEmbedder
from src.inference import load_bundle, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--threshold", type=float, default=None, help="below this confidence -> Unknown")
    args = ap.parse_args()

    bundle = load_bundle()
    result = predict(Image.open(args.image), FaceEmbedder(), bundle, args.threshold)
    if result["status"] == "no_face":
        print("No clear face detected in this image.")
        return
    print(f"\nPrediction : {result['label']}")
    print(f"Confidence : {result['confidence']:.1%}  (threshold {result['threshold']:.0%})")
    if result["unknown"]:
        print(f"Closest match was {result['best_guess']}, but confidence is too low to be sure.")
    print("\nTop matches:")
    for name, p in result["top"][:3]:
        print(f"  {name:25s} {p:.1%}")


if __name__ == "__main__":
    main()
