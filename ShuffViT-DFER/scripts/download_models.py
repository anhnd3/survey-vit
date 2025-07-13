#!/usr/bin/env python3
"""
download_models.py  –  robust to TorchVision version drift
-----------------------------------------------------------
• Saves ImageNet-pretrained ShuffleNet V2 and EfficientViT-M2
  state-dicts under ./models/ for offline use.
"""

import torch, argparse
from pathlib import Path

# ---------- ShuffleNet V2 -----------------------------------------------------
def load_shufflenet():
    """
    Returns an ImageNet-pretrained ShuffleNetV2 (1.0x) regardless
    of whether the current TorchVision uses the new Weights API
    or the older `pretrained=True` flag.
    """
    from torchvision.models import shufflenet_v2_x1_0
    try:  # TorchVision ≥0.13
        from torchvision.models import Shufflenet_V2_X1_0_Weights as W
        return shufflenet_v2_x1_0(weights=W.IMAGENET1K_V1)
    except Exception:
        # Fallback for older or newer API variants
        return shufflenet_v2_x1_0(pretrained=True)

# ---------- EfficientViT M2 ---------------------------------------------------
def load_efficientvit():
    """
    Uses timm (>=0.6.13) which fetches weights from the HF hub.
    If timm doesn't have EfficientViT, prompt the user.
    """
    import timm
    try:
        return timm.create_model("efficientvit_m2.r224_in1k",
                                 pretrained=True, num_classes=0)
    except Exception as e:
        raise RuntimeError(
            "✖ timm couldn't find EfficientViT-M2 "
            "(need timm≥0.6.13).  pip install --upgrade timm") from e

# -----------------------------------------------------------------------------


def save_state_dict(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"💾  Saved → {path.relative_to(Path.cwd())}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models", help="output folder")
    args = ap.parse_args()
    out = Path(args.out).resolve()

    print("⏬  Downloading ShuffleNet V2 weights …")
    snet = load_shufflenet()
    save_state_dict(snet, out / "shufflenet_v2_x1_0_imagenet.pth")

    print("⏬  Downloading EfficientViT-M2 weights …")
    evit = load_efficientvit()
    save_state_dict(evit, out / "efficientvit_m2_r224_in1k.pth")

    params = sum(p.numel() for p in snet.parameters()) / 1e6
    print(f"✅  Done.  ShuffleNetV2 params: {params:.2f} M")

if __name__ == "__main__":
    main()
