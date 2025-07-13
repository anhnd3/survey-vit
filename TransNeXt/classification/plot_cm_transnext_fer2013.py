import sys, pathlib, torch, matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

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

# --- point Python at classification/ so the import works ---
ROOT = pathlib.Path(__file__).resolve().parents[0]          # folder that holds this script
sys.path.append(str(ROOT / "classification"))               # add repo module path

from transnext import transnext_base                         # <-- now recognised

# ---------- paths ----------
CKPT_PATH  = "checkpoints/transnext_base/best.pth"
DATA_ROOT  = "datasets/FER2013/test"
IMG_SIZE   = 224
BATCH_SIZE = 64
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- build model ----------
model = transnext_base(num_classes=7)                        # use the constructor
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"])
model.to(DEVICE).eval()                                     # set to eval mode

# ---------- test loader ----------
tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
test_ds = ImageFolder(DATA_ROOT, transform=tfms)
test_ld = DataLoader(test_ds, batch_size=BATCH_SIZE,
                     shuffle=False, num_workers=4, pin_memory=True)

# ---------- inference ----------
y_true, y_pred = [], []
with torch.no_grad():
    for x, y in test_ld:
        p = model(x.to(DEVICE, non_blocking=True)).argmax(1).cpu()
        y_true.extend(y.numpy())
        y_pred.extend(p.numpy())

# ---------- confusion matrix ----------
cm = confusion_matrix(y_true, y_pred, labels=range(7))
disp = ConfusionMatrixDisplay(cm,
        display_labels=['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral'])
disp.plot(cmap="Blues", xticks_rotation=45, colorbar=False)
plt.title("TransNeXt-B x FER-2013")
plt.tight_layout()

out_dir = ROOT / "results" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / "cm_transnextB.png", dpi=200)
print(f"Saved to {out_dir/'cm_transnextB.png'}")

# ==== BENCHMARK ==========================================================
total_params = sum(p.numel() for p in model.parameters()) / 1e6  # triệu

dummy = torch.randn(1, 3, 224, 224, device=DEVICE)
lat, fps, mem = benchmark(model, dummy, DEVICE)

print(f"Params: {total_params:.2f} M  |  Latency: {lat:.2f} ms  |" f"  FPS: {fps:.1f}  |  VRAM peak: {mem:.0f} MB")