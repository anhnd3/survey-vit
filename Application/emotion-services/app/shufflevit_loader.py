# shufflevit_loader.py
import torch
import torch.nn as nn
from typing import Optional

# Re-use the global constant you already define in inference.py
NUM_CLASSES     = 7  # or import from a shared constants file

def load_shufflevit_dfer(
    model_path: Optional[str],
    device: torch.device,
) -> nn.Module:
    """
    Load the **ShuffleViT-DFER hybrid** that lives in
    `models/combmodel1.py` (which itself stitches EfficientViT +
    ShuffleNet-V2 + attention fusion).

    • Creates `CombinedModel(num_classes=NUM_CLASSES)`.
    • If `model_path` is given, torch-loads the checkpoint
      (works with both plain .pth and "model": {...} dicts).
    • Moves to the requested device and `.eval()` mode.
    """
    # ── local import so that tests w/out the file still import the module
    from models.combmodel1 import CombinedModel     # same hybrid arch

    model = CombinedModel(num_classes=NUM_CLASSES).to(device)

    if model_path:
        print(f"[ShuffleViT-DFER] Loading weights → {model_path}")
        state = torch.load(model_path, map_location=device)
        state = state.get("model", state)            # handle wrapped ckpts
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f" ⚠️  Missing keys: {len(missing)}")
        if unexpected:
            print(f" ⚠️  Unexpected keys: {len(unexpected)}")

    model.eval()
    return model
