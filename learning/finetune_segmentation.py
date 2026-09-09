# -*- coding: utf-8 -*-
"""
learning/finetune_segmentation.py
===================================
Fine-tune SegFormer-B0 trên dataset lũ lụt (FloodNet / RescueNet).

Dataset miễn phí:
    ─ FloodNet  — https://github.com/tallyuzhe/FloodNet
      Kaggle: https://www.kaggle.com/datasets/handlungjern/floodnet
      2337 ảnh train/val, label: 10 classes (flood, road, car, building...)
      Chuyển thành binary: flood/not-flood trước khi train.

    ─ RescueNet — https://github.com/jpcancela/RescueNet
      Kaggle:   https://www.kaggle.com/datasets/handlungjern/rescuenet
      ~15k ảnh + masks, binary flood.

Cấu trúc folder trước khi chạy:
    datasets/
        FloodNet/
            train/
                images/      ← *.jpg
                labels/      ← *.png (0=no flood, 1=flood)
            val/
                images/
                labels/
    HOẶC theo RescueNet:
        datasets/
            RescueNet/
                train/
                    images/
                    labels/
                val/
                    images/
                    labels/

Cách chạy:
    python -m learning.finetune_segmentation \\
        --dataset_dir datasets/FloodNet \\
        --output_dir models/flood_segnet \\
        --epochs 30 --lr 3e-4 --batch_size 8

    Sau khi train: copy models/flood_segnet/ rồi dùng trong config.yaml:
        ground_detection.model = "local:models/flood_segnet"

Lưu ý:
    - Cần GPU (≥8GB VRAM); nếu CPU → epochs=5, batch=2 để test nhanh
    - pip install torch torchvision transformers datasets albumentations
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger("finetune_seg")

# Ánh xạ labels → binary: FloodNet/RescueNet có nhiều class,
# nhưng ta chỉ cần binary: flood=1, mọi thứ khác=0
# FloodNet label IDs: 0=background, 1=road, 2=car, 3=building,
#   4=tree, 5=pool, 6=flood, 7=pole, 8=sidewalk, 9=person/bike
# Cách đơn giản nhất: pixel != 0 và KHÔNG phải road/building/car = flood?
# Cách tốt hơn cho FloodNet: class 6 ("flood") = 1, còn lại = 0
FLOODNET_FLOOD_ID = 6
RESCUENET_FLOOD_ID = 1


def load_dataset(
    dataset_dir: str,
    split: str = "train",
    flood_id: int = FLOODNET_FLOOD_ID,
    img_size: int = 512,
):
    """
    Load dataset từ folder. Trả về list (PIL.Image mask, PIL.Image image)
    """
    import cv2

    ds_path = Path(dataset_dir) / split
    img_dir = ds_path / "images"
    msk_dir = ds_path / "labels"
    img_exts = {".jpg", ".jpeg", ".png", ".webp"}

    items = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in img_exts:
            continue
        # Tìm mask tương ứng (có thể là .png hoặc .jpg)
        stem = img_path.stem
        msk_path = None
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            cand = msk_dir / f"{stem}{ext}"
            if cand.exists():
                msk_path = cand
                break
        if msk_path is None:
            continue
        items.append((img_path, msk_path))

    log.info(f"  Loaded {len(items)} {split} images from {ds_path}")

    # Lazy loading trong DataLoader-style — trả về paths, load on demand
    return items, flood_id, img_size


def _read_pair(img_path: Path, msk_path: Path, flood_id: int, img_size: int,
               is_train: bool = True):
    """
    Đọc + augment 1 cặp image/mask. Trả về (img_tensor, mask_tensor).
    """
    import cv2
    import torch

    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Không đọc được: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Không đọc được mask: {msk_path}")

    # Resize
    img  = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

    # Convert mask sang binary: flood_id → 1, còn lại → 0
    binary = (mask == flood_id).astype(np.uint8)

    # Random augment (train only)
    if is_train:
        if random.random() < 0.5:
            img  = np.flip(img,  axis=1).copy()
            binary = np.flip(binary, axis=1).copy()
        if random.random() < 0.3:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)

    # Normalize image (ImageNet standards)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img  = (img - mean) / std

    # To tensor: CHW
    img_t  = torch.from_numpy(img.transpose(2, 0, 1)).float()
    msk_t  = torch.from_numpy(binary).long().unsqueeze(0)  # (1,H,W)
    return img_t, msk_t


def train(args):
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    class FloodSegDataset(Dataset):
        def __init__(self, items, flood_id, img_size, is_train=True):
            self.items = items
            self.flood_id = flood_id
            self.img_size = img_size
            self.is_train = is_train
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            img_path, msk_path = self.items[idx]
            return _read_pair(img_path, msk_path, self.flood_id, self.img_size, self.is_train)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"  Device: {device}")

    # Load datasets
    train_items, flood_id, img_size = load_dataset(args.dataset_dir, "train", img_size=args.img_size)
    val_items,   _,           _     = load_dataset(args.dataset_dir, "val",   img_size=args.img_size)

    if not train_items:
        log.error("Không tìm thấy ảnh train trong dataset_dir")
        return

    # Limit dataset nếu cần test nhanh
    if args.max_samples and args.max_samples > 0:
        train_items = train_items[:args.max_samples]
        val_items   = val_items[:min(args.max_samples, len(val_items))]

    train_ds = FloodSegDataset(train_items, flood_id, img_size, is_train=True)
    val_ds   = FloodSegDataset(val_items,   flood_id, img_size, is_train=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model: SegFormer-B0 pretrained → thay decoder head thành 2 classes (binary)
    from transformers import SegformerForSemanticSegmentation, SegformerConfig
    model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
    config = SegformerConfig.from_pretrained(model_name)
    config.num_labels = 2  # binary: flood / not-flood
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name, num_labels=2, ignore_mismatched_sizes=True,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_iou = 0.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"  Bắt đầu train: {args.epochs} epochs, lr={args.lr}, batch={args.batch_size}")

    for epoch in range(args.epochs):
        # ── Train ──────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        n_batch = 0
        for imgs, msks in train_dl:
            imgs, msks = imgs.to(device), msks.to(device)
            outputs = model(imgs)
            # outputs.logits: (B, 2, H/4, W/4) → interpolate về (B,2,H,W)
            logits = torch.nn.functional.interpolate(
                outputs.logits, size=msks.shape[2:], mode="bilinear", align_corners=False,
            )
            loss = loss_fn(logits, msks.squeeze(1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batch += 1
        scheduler.step()
        avg_loss = train_loss / max(n_batch, 1)

        # ── Val (IoU flood class) ──────────────────────────────────────
        model.eval()
        tp = fp = fn = 0
        with torch.no_grad():
            for imgs, msks in val_dl:
                imgs, msks = imgs.to(device), msks.to(device)
                outs = model(imgs)
                logits = torch.nn.functional.interpolate(
                    outs.logits, size=msks.shape[2:], mode="bilinear", align_corners=False,
                )
                pred = logits.argmax(dim=1)
                tgt  = msks.squeeze(1)
                tp += ((pred == 1) & (tgt == 1)).sum().item()
                fp += ((pred == 1) & (tgt == 0)).sum().item()
                fn += ((pred == 0) & (tgt == 1)).sum().item()

        iou = tp / (tp + fp + fn + 1e-7)
        log.info(f"  Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f}, flood_IoU={iou:.4f}")

        if iou > best_val_iou:
            best_val_iou = iou
            # Save best
            model.save_pretrained(str(out_dir))
            from transformers import SegformerFeatureExtractor
            feat = SegformerFeatureExtractor.from_pretrained(model_name)
            feat.save_pretrained(str(out_dir))
            log.info(f"    → Saved best (IoU={iou:.4f})")

            if iou >= args.target_iou:
                log.info(f"  Đạt target IoU={args.target_iou} → dừng sớm")
                break

    # Save metadata
    meta = {"flood_id": flood_id, "img_size": img_size,
            "best_val_iou": best_val_iou, "epochs_run": epoch + 1}
    (out_dir / "finetune_meta.json").write_text(json.dumps(meta, indent=2))
    log.info(f"  Done. Best IoU={best_val_iou:.4f}. Model → {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SegFormer on flood segmentation")
    parser.add_argument("--dataset_dir", required=True, help="Folder chứa train/val splits")
    parser.add_argument("--output_dir",   default="models/flood_segnet")
    parser.add_argument("--epochs",       type=int, default=30)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--batch_size",   type=int, default=8)
    parser.add_argument("--img_size",     type=int, default=512)
    parser.add_argument("--max_samples",  type=int, default=0, help="Limit dataset size (0=full)")
    parser.add_argument("--target_iou",   type=float, default=0.70, help="Stop early khi đạt IoU")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train(args)


if __name__ == "__main__":
    main()
