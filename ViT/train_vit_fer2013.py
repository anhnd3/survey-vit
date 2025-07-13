review this code snippet again before running
```
#!/usr/bin/env python
"""
Train ViT-B/16 on FER-2013 with clean CLI, resume support, and per-epoch validation.
"""

import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T, models, datasets
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from tqdm import tqdm

NUM_CLASSES         = 7
FER_CLASSES         = ['0', '1', '2', '3', '4', '5', '6']
FER_CLASSES_LABEL   = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data(root: Path, batch_size: int = 64, num_workers: int = 4):
    """
    Load FER-2013 from an *image-folder* layout:

        root/
          train/
            0/
            1/
            ...
            6/
          (optionally test/ …)

    We create an 80 / 20 train-val split *from the train folder only*.
    """

    # Paths --------------------------------------------------------------------
    train_dir = root / "train"              # e.g. .../FER2013/train/0
    if not train_dir.exists():
        raise FileNotFoundError(f"Expected image folders under {train_dir}")

    # Transforms ---------------------------------------------------------------
    tf_train = T.Compose([
        T.Grayscale(3),
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2, 0.1),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])
    tf_val = T.Compose([
        T.Grayscale(3),
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])

    # Dataset & split ----------------------------------------------------------
    full_ds  = datasets.ImageFolder(train_dir, transform=tf_train)
    val_len  = int(0.2 * len(full_ds))
    train_len = len(full_ds) - val_len
    train_ds, val_ds = random_split(
        full_ds,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(42)  # reproducible split
    )
    # Put eval transform on the *underlying* dataset for val split
    val_ds.dataset.transform = tf_val

    # DataLoaders --------------------------------------------------------------
    train_ld = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_ld = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    print(f"Loaded {train_len} train + {val_len} val images "
          f"from {train_dir.relative_to(root)}")
    return train_ld, val_ld


def build_model(pretrained=True):
    if pretrained:
        model = models.vit_b_16(weights="IMAGENET1K_V1")
    else:
        model = models.vit_b_16(weights=None)
    model.heads = nn.Linear(768, NUM_CLASSES)
    return model.to(DEVICE)

def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running_loss = 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * imgs.size(0)
        pbar.set_postfix(loss=loss.item())
    return running_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    losses = []
    y_true, y_pred = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        losses.append(loss.item() * imgs.size(0))
        preds = outputs.argmax(1)
        y_true.append(labels.cpu())
        y_pred.append(preds.cpu())
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return np.sum(losses) / len(loader.dataset), acc, cm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="FER2013 dataset root")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", type=Path, help="Checkpoint to resume from")
    parser.add_argument("--out", type=Path, default=Path("./checkpoints"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = get_data(args.data, args.batch_size)

    model = build_model(pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scaler = torch.amp.GradScaler('cuda')

    start_epoch = 0
    best_acc   = 0.0           # highest val accuracy seen so far
    best_ckpt  = args.out / "vit_best.pth"
    last_ckpt  = args.out / "vit_last.pth"

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc, val_cm = evaluate(model, val_loader, criterion)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc*100:.2f}%")
        print("Val Confusion Matrix:")
        print(val_cm)

        # ── after printing val metrics ───────────────────────────
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_acc": best_acc,
                },
                best_ckpt,
            )
            print(f"✓ New best ({best_acc*100:.2f}%)  ⇒  {best_ckpt.name}")

        # always (over-)write the "last" checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "best_acc": best_acc,
            },
            last_ckpt,
        )

if __name__ == "__main__":
    main()
    