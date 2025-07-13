#!/usr/bin/env python3
"""
plot_cm_and_benchmark.py
------------------------
Benchmark *one* FER-2013 model and write a confusion matrix.

Usage
-----
python plot_cm_and_benchmark.py \
    --ckpt  checkpoints/vit_best.pth \
    --arch  vit_b16            # or shufflevit_dfer | transnext_b
    --data  /path/to/FER2013   # must contain test/0 … 6
    --imgsz 224                # 224 for ViT/ShuffleViT, 384 for SWAG ViT
    --tag   vit_ft
"""

import argparse, time, numpy as np, torch, torch.nn as nn
from pathlib import Path
from torchvision import datasets, transforms as T, models
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH  = 64                # batch size for timing + CM
CLASSES = ['0','1','2','3','4','5','6']   # numeric folder names
CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# --------------------------- model factory ------------------------------------
def build_model(arch: str, num_classes=7):
    if arch == "vit_b16":
        m = models.vit_b_16(weights=None)
        m.heads = nn.Linear(768, num_classes)
    elif arch == "shufflevit_dfer":
        from timm import create_model
        m = create_model("shufflevit_b16_224", pretrained=False,
                         num_classes=num_classes)
    elif arch == "transnext_b":
        from timm import create_model
        m = create_model("transnext_base", pretrained=False,
                         num_classes=num_classes)
    else:
        raise ValueError(f"unknown arch {arch}")
    return m.to(DEVICE)

# --------------------------- timing helper ------------------------------------
@torch.no_grad()
def benchmark(model, example):
    torch.cuda.reset_peak_memory_stats()
    times = []
    starter = torch.cuda.Event(True); ender = torch.cuda.Event(True)
    for _ in range(2): model(example)   # warm-up
    loops = 10
    for _ in range(loops):
        starter.record(); model(example); ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))  # ms
    per_img = np.mean(times) / example.size(0)
    fps     = 1000. / per_img
    vram_mb = torch.cuda.max_memory_allocated() / 1024**2
    return per_img, fps, vram_mb

# --------------------------- CM plot ------------------------------------------
def save_cm(cm, tag):
    """Plot confusion matrix with ABSOLUTE counts (no colour-bar)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap='Blues')

    ax.set(
        xticks=range(7), yticks=range(7),
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ylabel='True', xlabel='Predicted',
        title=f'{tag} x FER-2013'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    fmt, thresh = 'd', cm.max() / 2
    for i in range(7):
        for j in range(7):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')

    fig.tight_layout()
    out = f'{tag}_cm.png'
    fig.savefig(out, dpi=200)
    print(f'✓ saved {out}')
# ------------------------------------------------------------------------------

# --------------------------- main ---------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, type=Path)
    ap.add_argument('--arch', required=True,
                    choices=['vit_b16','shufflevit_dfer','transnext_b'])
    ap.add_argument('--data', required=True, type=Path)
    ap.add_argument('--imgsz', type=int, default=224)
    ap.add_argument('--tag',   default='run')
    args = ap.parse_args()

    # data – test split -------------------------------------------------------
    tf = T.Compose([
        T.Grayscale(3),
        T.Resize(int(args.imgsz*1.15)),
        T.CenterCrop(args.imgsz),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),
    ])
    test_ds = datasets.ImageFolder(args.data / 'test', transform=tf)
    test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH,
                                          shuffle=False, num_workers=4,
                                          pin_memory=True)

    # model -------------------------------------------------------------------
    model = build_model(args.arch)
    state = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(state['model'] if 'model' in state else state,
                          strict=False)
    model.eval()

    # params ------------------------------------------------------------------
    params_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    # benchmark on first batch ------------------------------------------------
    xb, _ = next(iter(test_ld)); xb = xb.to(DEVICE)
    latency_ms, fps, vram = benchmark(model, xb)

    # full CM & accuracy ------------------------------------------------------
    y_t, y_p = [], []
    with torch.no_grad():
        for x, y in test_ld:
            p = model(x.to(DEVICE)).argmax(1).cpu()
            y_t.append(y); y_p.append(p)
    y_t = torch.cat(y_t).numpy(); y_p = torch.cat(y_p).numpy()
    acc = accuracy_score(y_t, y_p)
    cm  = confusion_matrix(y_t, y_p)
    print(f"Top-1 accuracy: {acc*100:.2f}%") 
    save_cm(cm, args.tag)

    # CSV output --------------------------------------------------------------
    print("tag,params_m,latency_ms,fps,vram_mb")
    print(f"{args.tag},{params_m:.1f},{latency_ms:.2f},{fps:.1f},{vram:.0f}")

if __name__ == '__main__':
    main()
