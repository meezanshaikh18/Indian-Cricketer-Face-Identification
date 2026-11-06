"""Face detection (MTCNN) + face embeddings (FaceNet / InceptionResnetV1, VGGFace2).

torch and facenet-pytorch are imported lazily so the rest of the project
(training, tests) can be imported without them.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

import config


def _to_float(prob):
    """MTCNN may return None, a float or a 1-element array/list; normalise it."""
    if prob is None:
        return None
    try:
        arr = np.asarray(prob, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    if arr.size == 0 or np.all(np.isnan(arr)):
        return None
    return float(np.nanmax(arr))


def prepare_image(img: Image.Image) -> Image.Image:
    """Fix phone-photo rotation, force RGB, shrink very large photos."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    longest = max(img.size)
    if longest > config.MAX_IMAGE_SIDE:
        scale = config.MAX_IMAGE_SIDE / longest
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    return img


class FaceEmbedder:
    def __init__(self, device: str | None = None, load_recognizer: bool = True):
        import torch
        from facenet_pytorch import MTCNN, InceptionResnetV1

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.mtcnn = MTCNN(
            image_size=config.FACE_SIZE,
            margin=config.FACE_MARGIN,
            min_face_size=config.MIN_FACE_SIZE,
            post_process=False,      # keep raw 0-255 pixels; we normalise ourselves
            select_largest=True,     # if several faces, use the biggest one
            keep_all=False,
            device=self.device,
        )
        self.recognizer = None
        if load_recognizer:
            self.recognizer = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    # ------------------------------------------------------------------
    def detect_and_crop(self, img: Image.Image):
        """Return (face_crop_PIL or None, detection_probability or None)."""
        img = prepare_image(img)
        face, prob = self.mtcnn(img, return_prob=True)
        p = _to_float(prob)
        if face is None or p is None or p < config.MIN_DETECTION_PROB:
            return None, p
        arr = face.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
        return Image.fromarray(arr), p

    # ------------------------------------------------------------------
    def embed(self, faces: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        """512-d L2-normalised embeddings. Each face is embedded together with
        its mirror image (averaged), which makes the vectors slightly more robust."""
        torch = self.torch
        size = config.FACE_SIZE
        out = []
        with torch.no_grad():
            for i in range(0, len(faces), batch_size):
                batch = faces[i:i + batch_size]
                arr = np.stack(
                    [np.asarray(f.convert("RGB").resize((size, size))) for f in batch]
                ).astype("float32")
                x = torch.from_numpy(arr).permute(0, 3, 1, 2)
                x = ((x - 127.5) / 128.0).to(self.device)
                e = self.recognizer(x) + self.recognizer(torch.flip(x, dims=[3]))
                out.append(e.cpu().numpy())
        emb = np.concatenate(out, axis=0)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10
        return emb
