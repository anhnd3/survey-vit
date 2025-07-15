"""
Application.emotion_services.app.inference
-----------------------------------------

Light-weight orchestration layer that wires together the *separate* backbone
loaders (vit.py, resnet.py, transnext.py, shufflevit.py).  Nothing model-
specific lives here any more; each loader file owns its own preprocessing and
classifier-head tweaks.

Run the service from the repo root with either:

    uvicorn Application.emotion_services.app.main:app --reload
or
    python  -m Application.emotion_services.app.main
"""

from __future__ import annotations

import pathlib
from typing import Dict, Tuple

import torch
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# ------------------------------------------------------------
# 1. Constants
# ------------------------------------------------------------
NUM_CLASSES = 7
EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"]

# ------------------------------------------------------------
# 2. Portable image → tensor transform (224 × 224, RGB)
#    All backbones here expect this, so keep it centralised.
# ------------------------------------------------------------
_IMG_TFM = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# ------------------------------------------------------------
# 3. Generic prediction wrapper
# ------------------------------------------------------------
class EmotionModel:
    """
    Thin wrapper that turns a raw PIL image into a softmax vector.
    Each backbone loader returns an nn.Module already moved to the given device.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.model.eval()  # just to be safe

    @torch.inference_mode()
    def predict(self, img: Image.Image) -> Dict[str, float]:
        x = _IMG_TFM(img.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(x)  # shape [1, NUM_CLASSES]
        probs = logits.softmax(dim=-1).squeeze(0).cpu().numpy()
        return {cls: float(prob) for cls, prob in zip(EMOTION_CLASSES, probs)}


# ------------------------------------------------------------
# 4. Import backbone loaders (each in its own file)
# ------------------------------------------------------------
import vit_loader, transnext_loader, shufflevit_loader


_LOADER_REGISTRY = {
    "vit_b16.pth": vit_loader.load_vit_b16,
    "TransNeXt-B.pth": transnext_loader.load_transnext_b,
    "ShuffleViT-DFER.t7": shufflevit_loader.load_shufflevit_dfer,
}

# ------------------------------------------------------------
# 5. Public dict holding ready-to-serve models
# ------------------------------------------------------------
emotion_models: Dict[str, EmotionModel] = {}

# ------------------------------------------------------------
# 6. Initialise / warm-up
# ------------------------------------------------------------
def _ckpt_path(name: str) -> pathlib.Path:
    """
    Return checkpoint path for a given backbone name, e.g.  'vit_b16' →
    Application/emotion-services/app/models/vit_b16.pth
    """
    here = pathlib.Path(__file__).resolve().parent
    return here / "models" / f"{name}"


def init_inference(device: torch.device | None = None) -> None:
    """
    Load every backbone found in `_LOADER_REGISTRY`.  For each key `k` we look
    for `models/{k}.pth`; if the file is missing we fall back to ImageNet
    weights provided by the loader.
    """
    global emotion_models
    if emotion_models:
        return  # already initialised

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[inference] Using device: {device}")

    for name, loader in _LOADER_REGISTRY.items():
        ckpt = _ckpt_path(name)
        try:
            model = loader(ckpt if ckpt.exists() else None, device)
            emotion_models[name] = EmotionModel(model, device)
            print(f"[✓] {name} loaded")
        except Exception as exc:  # pragma: no cover
            print(f"[✗] {name} failed: {exc}")

    if not emotion_models:
        raise RuntimeError("No emotion models could be loaded — check your checkpoints")

    # -------- warm-up (single dummy inference) -------------------------------
    dummy = Image.new("RGB", (224, 224), "white")
    for name in tqdm(emotion_models, desc="Warming-up models"):
        _ = emotion_models[name].predict(dummy)
    print("[inference] Ready models:", ", ".join(emotion_models))


# ------------------------------------------------------------
# 7. Face detection + annotation utility (FastAPI depends on it)
# ------------------------------------------------------------
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_and_annotate(
    image: Image.Image, model_key: str
) -> Tuple[Image.Image, Dict[str, float]]:
    """
    • Detect the largest face (simple Haar cascade).
    • Draw the bounding box.
    • Run the selected emotion model.
    • Return (annotated PIL image, {class: prob})
    """
    if model_key not in emotion_models:
        raise KeyError(f"Model '{model_key}' is not initialised")

    cv_img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, 1.1, 5)

    # only proceed if at least one face was found
    if len(faces) > 0:
        # pick the largest face -> (x, y, w, h)
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        cv2.rectangle(cv_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    annotated = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    probs = emotion_models[model_key].predict(image)
    return annotated, probs

