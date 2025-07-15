from typing import Optional
import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 7  # or import from a shared constants file


def load_vit_b16(
    model_path: Optional[str],
    device: torch.device,
    optimized: bool = False,
) -> nn.Module:
    model = models.vit_b_16(weights=None)
    model.heads = (
        nn.Sequential(nn.Dropout(0.1), nn.Linear(768, NUM_CLASSES))
        if optimized
        else nn.Linear(768, NUM_CLASSES)
    )
    model.to(device)

    if model_path:
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state, strict=True)

    model.eval()
    return model
