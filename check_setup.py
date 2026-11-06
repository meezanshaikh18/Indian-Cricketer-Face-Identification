"""Run this once after installing requirements:  python check_setup.py"""
import sys

print("Python", sys.version.split()[0])
for mod in ["numpy", "pandas", "sklearn", "matplotlib", "PIL", "torch", "facenet_pytorch", "streamlit", "kagglehub"]:
    try:
        m = __import__(mod)
        print(f"  OK   {mod:16s} {getattr(m, '__version__', '')}")
    except Exception as exc:
        print(f"  FAIL {mod:16s} {exc}")

print("\nLoading face models (the first run downloads ~100 MB of weights)...")
import numpy as np
from PIL import Image
from src.face_embedder import FaceEmbedder

emb = FaceEmbedder()
noise = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype="uint8"))
face, _ = emb.detect_and_crop(noise)
vec = emb.embed([noise.resize((160, 160))])
print("Face detector OK (no face in random noise:", face is None, ")")
print("Embedding shape:", vec.shape, "-> expected (1, 512)")
print("\nSetup looks good.")
