#!/usr/bin/env python3
"""
combinemodelfer.py
------------------
Train & evaluate ShuffleViT-DFER on the FER-2013 dataset.

 • Expects:
     datasets/FER2013/train/*/*.png   (part 1 output)
     datasets/FER2013/test/*/*.png
     models/shufflenet_v2_x1_0_imagenet.pth
     models/efficientvit_m2_r224_in1k.pth

 • Saves best checkpoint to:
     fer2013_Ourmodel/1/Test_model.t7
"""

import os, argparse, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

# ─── Local repo imports ────────────────────────────────────────────────────────
from models.combmodel1 import CombinedModel                         # unchanged
from FER2013 import FER2013                                         # new file
# ───────────────────────────────────────────────────────────────────────────────

# ---------------------------- Hyper-parameters --------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=90)
    ap.add_argument('--bs',     type=int, default=64,  help='batch size')
    ap.add_argument('--lr',     type=float, default=5e-3)
    ap.add_argument('--imgsz',  type=int, default=224)
    ap.add_argument('--data',   default='datasets/FER2013',
                    help='root folder produced in Part 1')
    ap.add_argument('--model_dir', default='models')
    ap.add_argument('--save_dir',  default='models_result/fer2013')
    return ap.parse_args()

# ---------------------------- Data transforms ---------------------------------
def get_transforms(img_sz):
    train_tf = transforms.Compose([
        transforms.Resize((img_sz, img_sz)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((img_sz, img_sz)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])
    return train_tf, test_tf

# ---------------------------- Train / Eval loops ------------------------------
@torch.inference_mode()
def evaluate(model, loader, crit, device):
    model.eval()
    total_loss, total_acc, n = 0., 0., 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        total_loss += crit(out, labels).item() * imgs.size(0)
        total_acc  += (out.argmax(1) == labels).sum().item()
        n += imgs.size(0)
    return total_loss / n, total_acc / n

def train_one_epoch(model, loader, crit, optim, device):
    model.train()
    total_loss, total_acc, n = 0., 0., 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optim.zero_grad()
        out = model(imgs)
        loss = crit(out, labels)
        loss.backward()
        optim.step()
        total_loss += loss.item() * imgs.size(0)
        total_acc  += (out.argmax(1) == labels).sum().item()
        n += imgs.size(0)
    return total_loss / n, total_acc / n

# ------------------------------------------------------------------------------
def main():
    opt = parse_args()
    Path(opt.save_dir).mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_tf, test_tf = get_transforms(opt.imgsz)

    train_set = FER2013(split='train', transform=train_tf, root=opt.data)
    test_set  = FER2013(split='test',  transform=test_tf,  root=opt.data)

    train_loader = DataLoader(train_set, batch_size=opt.bs, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=opt.bs, shuffle=False,
                              num_workers=4, pin_memory=True)

    # -------------------- Model ------------------------------------------------
    model = CombinedModel(num_classes=7).to(device)

    # Manually load pretrained backbones
    # model.feature_extractor_shufflenet.load_state_dict(
    #     torch.load(f"{opt.model_dir}/shufflenet_v2_x1_0_imagenet.pth"))
    # model.feature_extractor_efficientvit.load_state_dict(
    #     torch.load(f"{opt.model_dir}/efficientvit_m2_r224_in1k.pth"))
    
    state = torch.load(f"{opt.model_dir}/shufflenet_v2_x1_0_imagenet.pth")
    model.feature_extractor_shufflenet.load_state_dict(state, strict=False)
    
    state1 = torch.load(f"{opt.model_dir}/efficientvit_m2_r224_in1k.pth")
    model.feature_extractor_efficientvit.load_state_dict(state1, strict=False)

    # -------------------- Optim / Loss ----------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=opt.lr,
                          momentum=0.9, weight_decay=5e-4)

    best_acc = 0.0
    for epoch in range(1, opt.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion,
                                          optimizer, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        dt = time.time() - t0

        print(f"[{epoch}/{opt.epochs}] "
              f"train {tr_acc:.4f}/{tr_loss:.4f} | "
              f"test {te_acc:.4f}/{te_loss:.4f} | "
              f"{dt:.1f}s")

        if te_acc > best_acc:
            best_acc = te_acc
            ckpt = Path(opt.save_dir) / "Test_model.t7"
            torch.save(model.state_dict(), ckpt)
            print(f"  ↳ 🏅  New best saved: {ckpt}")

    print(f"✅ Finished. Best test acc: {best_acc:.4f}")

if __name__ == "__main__":
    main()
# ------------------------------------------------------------------------------