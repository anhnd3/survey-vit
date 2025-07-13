#!/usr/bin/env python3
"""
prepare_fer2013.py  –  Hugging Face edition
Converts the HF dataset Jeneral/fer-2013 into an ImageFolder tree.

Output structure
datasets/FER2013/
    train/0/*.png  …  train/6/*.png
    test/0/*.png   …  test/6/*.png
"""

import argparse, os
from pathlib import Path
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset


def save_png(img: Image.Image, label: int, split: str, root: Path, idx: int):
    img = img.convert("RGB")              # ensure 3-channel
    tgt_dir = root / split / str(label)
    tgt_dir.mkdir(parents=True, exist_ok=True)
    img.save(tgt_dir / f"{idx}.png", compress_level=1)


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("--dataset", default="Jeneral/fer-2013",
                      help="HF dataset repo or local path")
    argp.add_argument("--out", default="datasets/FER2013",
                      help="Destination root folder")
    args = argp.parse_args()
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)

    print(f"⏬  Loading {args.dataset} …")
    ds = load_dataset(args.dataset)

    for split in ds.keys():                      # 'train', 'test'
        print(f"🖼️  Converting {split} split")
        for i, ex in enumerate(tqdm(ds[split], total=len(ds[split]))):
            # HF stores bytes; convert to PIL.Image
            img = Image.open(BytesIO(ex["img_bytes"]))
            label = int(ex["labels"])
            save_png(img, label, split, root, i)

    print(f"✅  FER-2013 ready in ImageFolder layout at {root}")


if __name__ == "__main__":
    main()

