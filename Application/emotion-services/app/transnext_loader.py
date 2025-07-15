from typing import Optional
import torch
import torch.nn as nn
from transnext import transnext_base                         # <-- now recognised

NUM_CLASSES = 7  # or import from a shared constants file

def load_transnext_b(
    model_path: str | None,
    device: torch.device,
    drop_rate: float = 0.0,
) -> nn.Module:
    model = transnext_base(num_classes=NUM_CLASSES, drop_rate=drop_rate).to(device)

    if model_path:
        print(f"[TransNeXt-B] loading weights from {model_path}")
        # <-- pass weights_only=False so that argparse.Namespace etc. get unpickled
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        # many checkpoints wrap the state_dict under "model"
        state = ckpt.get("model", ckpt)  
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  ⚠️ missing keys: {missing}")
        if unexpected:
            print(f"  ⚠️ unexpected keys: {unexpected}")

    model.eval()
    return model