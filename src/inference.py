"""Shared prediction logic used by the command-line tool and the Streamlit app."""
from __future__ import annotations

import joblib
import numpy as np

import config


def load_bundle(path=config.MODEL_FILE) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Train the model first (python -m src.train).")
    return joblib.load(path)


def predict(img, embedder, bundle: dict, threshold: float | None = None, top_k: int = 5) -> dict:
    """Identify the player in a PIL image.

    Returns {"status": "no_face"} or a dict with label ("Unknown" when the
    confidence is below the threshold), best_guess, confidence and the top-k list.
    """
    face, det_prob = embedder.detect_and_crop(img)
    if face is None:
        return {"status": "no_face", "det_prob": det_prob}
    emb = embedder.embed([face])
    model = bundle["model"]
    proba = model.predict_proba(emb)[0]
    order = np.argsort(proba)[::-1][:top_k]
    top = [(str(model.classes_[i]), float(proba[i])) for i in order]
    thr = bundle["threshold"] if threshold is None else threshold
    best_label, best_p = top[0]
    return {
        "status": "ok",
        "face": face,
        "det_prob": det_prob,
        "best_guess": best_label,
        "confidence": best_p,
        "unknown": best_p < thr,
        "label": "Unknown" if best_p < thr else best_label,
        "top": top,
        "threshold": thr,
    }
