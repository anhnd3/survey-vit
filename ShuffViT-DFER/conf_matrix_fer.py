#!/usr/bin/env python3
"""
confmatrixfer.py
----------------
Draws a confusion matrix for the best ShuffleViT-DFER model
trained on FER-2013 (see combinemodelfer.py).

Usage
-----
$ python confmatrixfer.py \
    --weights fer2013_Ourmodel/1/Test_model.t7 \
    --data    datasets/FER2013

If you omit --weights, the default path above is used.
"""

import argparse, itertools
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from torchvision import transforms

from FER2013 import FER2013
from models.combmodel1 import CombinedModel     # same hybrid arch

# ---------------------------------------------------------------------------
CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
NUM_CLASSES = len(CLASS_NAMES)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def get_predictions(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        out  = model(imgs).argmax(1).cpu().numpy()
        y_pred.extend(out)
        y_true.extend(labels.numpy())
    return np.array(y_true), np.array(y_pred)


def plot_confusion(cm, classes, save_path):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('ShuffleViT-DFER x FER-2013')
    # plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha='center', va='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"💾  Confusion matrix saved → {save_path}")
    plt.show()
    
import time

def benchmark(model, dummy, device='cuda', n_warm=10, n_run=100):
    """
    Returns (lat_ms, fps, mem_mb).
    """
    model.eval()
    dummy = dummy.to(device)
    torch.cuda.reset_peak_memory_stats(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(n_warm):
            _ = model(dummy)

    # Timed runs
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_run):
            _ = model(dummy)
    torch.cuda.synchronize(device)
    t1 = time.perf_counter()

    lat_ms = (t1 - t0) / n_run * 1e3          # ms / image
    fps    = 1e3 / lat_ms
    mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return lat_ms, fps, mem_mb
# ------------------------------------------------------------------------------


def main():
    # ------------------------------------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='models_result/fer2013/Test_model.t7')
    ap.add_argument('--data',    default='datasets/FER2013')
    ap.add_argument('--imgsz',   type=int, default=224)
    ap.add_argument('--batch',   type=int, default=64)
    args = ap.parse_args()

    # ---------------- Test transforms (match training) ----------------
    tfm_test = transforms.Compose([
        transforms.Resize((args.imgsz, args.imgsz)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # ---------------- Dataset & DataLoader ----------------------------
    test_set = FER2013(split='test', transform=tfm_test, root=args.data)
    loader   = DataLoader(test_set, batch_size=args.batch,
                          shuffle=False, num_workers=4, pin_memory=True)

    # ---------------- Model -------------------------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = CombinedModel(num_classes=NUM_CLASSES).to(device)
    state  = torch.load(args.weights, map_location=device)
    model.load_state_dict(state, strict=False)

    # ---------------- Predictions -------------------------------------
    y_true, y_pred = get_predictions(model, loader, device)
    cm   = confusion_matrix(y_true, y_pred)
    acc  = (cm.diagonal().sum() / cm.sum())  # overall accuracy

    # ---------------- Plot & Save -------------------------------------
    save_path = Path(args.weights).with_suffix('.png')
    plot_confusion(cm, CLASS_NAMES, save_path)
    print(f"Overall accuracy: {acc*100:.2f}%  (check {save_path})")
    
    # ==== BENCHMARK ==========================================================
    total_params = sum(p.numel() for p in model.parameters()) / 1e6  # triệu

    dummy = torch.randn(1, 3, 224, 224, device=device)
    lat, fps, mem = benchmark(model, dummy, device)

    print(f"Params: {total_params:.2f} M  |  Latency: {lat:.2f} ms  |"f"  FPS: {fps:.1f}  |  VRAM peak: {mem:.0f} MB")


if __name__ == "__main__":
    main()
# ------------------------------------------------------------------------------