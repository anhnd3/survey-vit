"""
Minimal stub so `models.build` can import EfficientViT
and CombinedModel can access .head.l.in_features.
"""

import torch.nn as nn
import timm


class EfficientViT(nn.Module):
    def __init__(self, num_classes=1000, distillation=False, **kwargs):
        super().__init__()
        # timm EfficientViT-M2 backbone
        self.net = timm.create_model(
            "efficientvit_m2.r224_in1k",
            pretrained=False,      # we'll load .pth externally
            num_classes=num_classes
        )

        # --- expose .head.l to mimic authors' original API -------------
        cls_layer = (
            self.net.get_classifier()
            if hasattr(self.net, "get_classifier")
            else getattr(self.net, "head", None)
        )

        class _HeadWrapper(nn.Module):
            def __init__(self, layer):
                super().__init__()
                self.l = layer      # <-- this is what combmodel1 looks for
            def forward(self, x):
                return self.l(x)

        self.head = _HeadWrapper(cls_layer)
        self.num_features = getattr(self.net, "num_features", 0)

    def forward(self, x):
        feats = self.net.forward_features(x)          # (B, C, H, W)
        feats = nn.functional.adaptive_avg_pool2d(feats, 1).flatten(1)
        return feats
