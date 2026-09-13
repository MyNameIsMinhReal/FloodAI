# -*- coding: utf-8 -*-
"""
learning/finetune_segmentation.py
==================================
Fine-tune SegFormer trên dataset lũ lụt cho WATER SEGMENTATION.
Phiên bản cải tiến theo spec:

  2. Không resize méo ảnh
     - Giữ aspect ratio (letterbox pad, không kéo giãn).
     - Dataset prep: chuẩn hóa cạnh dài ~1536-2048px.
     - Train: resize xuống 768×768 hoặc 1024×1024.
     - Mask transform y hệt ảnh (cùng scale + pad).

  3. Train Water Segmentation trước
     - SegFormer-B0 = baseline, SegFormer-B2 = model chính.
     - A4000 16GB: B2 + mixed precision + batch nhỏ.
     - batch_size=2, image_size=768, fp16/bf16, grad_accum=4.
     - Có thể thử 1024×1024 nếu VRAM ổn.

  4. Loss
     - Dice + CE/BCE + Boundary Loss (bắt đúng mép nước).

  5. Đánh giá
     - IoU, Dice, Precision, Recall, Boundary IoU.

Dataset (binary flood mask):
    FloodNet  — https://github.com/tallyuzhe/FloodNet  (class flood id=6)
    RescueNet — https://github.com/jpcancela/RescueNet (class flood id=1)
    Hoặc folder chuẩn: train/images + train/labels (ảnh + mask png)

Cách dùng:
  # 0) Prep dataset: chuẩn hóa cạnh dài, không méo
  python -m learning.finetune_segmentation --stage prepare \
      --dataset_dir datasets/raw --output_dir datasets/FloodNet_proc \
      --long_side 1536

  # 1) Train SegFormer-B2 (model chính, A4000 16GB)
  python -m learning.finetune_segmentation --stage train \
      --dataset_dir datasets/FloodNet_proc --output_dir models/flood_segnet_b2 \
      --model b2 --img_size 768 --batch_size 2 --grad_accum 4 \
      --epochs 60 --target_iou 0.85

  # 2a) hoặc baseline B0 (test nhanh)
  python -m learning.finetune_segmentation --stage train \
      --dataset_dir datasets/FloodNet_proc --output_dir models/flood_segnet_b0 \
      --model b0 --img_size 512 --batch_size 4 --target_iou 0.75

  Sau khi train: copy model ra rồi dùng trong config.yaml:
      ground_detection.model = "local:models/flood_segnet_b2"

  Cần GPU ≥8GB; thiếu GPU thì batch=1, b0, 512px để test.
  pip install torch torchvision transformers datasets albumentations
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

log = logging.getLogger("finetune_seg")

# FloodNet label IDs: 6="flood". RescueNet binary: 1="flood".
FLOODNET_FLOOD_ID = 6
RESCUENET_FLOOD_ID = 1

SEGFORMER_MODELS = {
    "b0": "nvidia/segformer-b0-finetuned-ade-512-512",
    "b2": "nvidia/segformer-b2-finetuned-ade-512-512",
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA — aspect-ratio preserving (letterbox), mask transform y hệt ảnh
# ─────────────────────────────────────────────────────────────────────────────

def _letterbox(img: np.ndarray, mask: np.ndarray, target: int,
               val: int = 0, mask_val: int = 255):
    """Resize giữ aspect ratio → không méo. Pad đều 2 cạnh về square.
    Trả về (img,(h_scale,w_scale,top,left),mask) và metadata để transform y hệt."""
    h, w = img.shape[:2]
    scale = target / max(h, w)
    nh, nw = round(h * scale), round(w * scale)
    img = cv_resize(img, (nw, nh), interp="linear")
    if mask is not None:
        mask = cv_resize(mask, (nw, nh), interp="nearest")

    top = (target - nh) // 2
    left = (target - nw) // 2
    canvas = np.full((target, target, img.shape[2]), val, dtype=img.dtype) \
        if img.ndim == 3 else np.full((target, target), val, dtype=img.dtype)
    canvas[top:top + nh, left:left + nw] = img

    if mask is None:
        return canvas, {"scale": scale, "top": top, "left": left}
    mcanvas = np.full((target, target), mask_val, dtype=mask.dtype)
    mcanvas[top:top + nh, left:left + nw] = mask
    return canvas, mcanvas, {"scale": scale, "top": top, "left": left}


def cv_resize(arr: np.ndarray, size: Tuple[int, int], interp: str = "linear"):
    import cv2
    if interp == "linear":
        interp_cv = cv2.INTER_LINEAR
    elif interp == "cubic":
        interp_cv = cv2.INTER_CUBIC
    else:
        interp_cv = cv2.INTER_NEAREST
    if arr.ndim == 2:
        return cv2.resize(arr, size, interpolation=interp_cv)
    return cv2.resize(arr, size, interpolation=interp_cv)


def load_dataset(dataset_dir: str, split: str = "train"):
    """Trả về list (img_path, mask_path) cho train/val đã proc (aspect-ratio giữ)."""
    ds_path = Path(dataset_dir) / split
    img_dir = ds_path / "images"
    msk_dir = ds_path / "labels"
    img_exts = {".jpg", ".jpeg", ".png", ".webp"}
    items = []
    if not img_dir.exists():
        return items
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in img_exts:
            continue
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
    return items


def _read_pair(img_path: Path, msk_path: Path, flood_id: int, img_size: int, is_train: bool):
    """Đọc + augment 1 cặp. Giữ aspect ratio (letterbox), mask transform y hệt."""
    import cv2
    import torch

    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Không đọc được: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Không đọc được mask: {msk_path}")

    # ── Augment trước khi resize (hệ quy chiếu gốc, không méo) ──────────
    if is_train:
        if np.random.rand() < 0.5:
            img = np.flip(img, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        if np.random.rand() < 0.4:
            ang = np.random.uniform(-10, 10)
            (hh, ww) = img.shape[:2]
            M = cv2.getRotationMatrix2D((ww / 2, hh / 2), ang, 1.0)
            img = cv2.warpAffine(img, M, (ww, hh), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask, M, (ww, hh), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        # Scale/zoom ngẫu nhiên (vẫn giữ aspect)
        if np.random.rand() < 0.4:
            s = np.random.uniform(0.9, 1.2)
            hh, ww = img.shape[:2]
            nh, nw = int(hh * s), int(ww * s)
            img = cv_resize(img, (nw, nh), "cubic")
            mask = cv_resize(mask, (nw, nh), "nearest")
            # crop về kích thước gốc (center)
            y0, x0 = max(0, (nh - hh) // 2), max(0, (nw - ww) // 2)
            img = img[y0:y0 + hh, x0:x0 + ww]
            mask = mask[y0:y0 + hh, x0:x0 + ww]
        if np.random.rand() < 0.3:
            k = int(np.random.choice([3, 5, 7]))
            img = cv2.GaussianBlur(img, (k, k), 0)
        if np.random.rand() < 0.2:
            img = img.astype(np.float32) * np.random.uniform(0.8, 1.2)
            img = np.clip(img, 0, 255).astype(np.uint8)

    # ── Letterbox (giữ aspect ratio, không méo) ──────────────────────────
    img, mask, meta = _letterbox(img, mask, img_size)

    # Binary mask: flood_id → 1, ignore 255 (padding)
    binary = np.where(mask == flood_id, 1, 0).astype(np.int64)
    ignore = (mask == 255)

    # Normalize (ImageNet)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    img_t = torch.from_numpy(img.transpose(2, 0, 1)).float()
    msk_t = torch.from_numpy(binary).long()           # (H,W)
    ign_t = torch.from_numpy(ignore).bool()
    return img_t, msk_t, ign_t


# ─────────────────────────────────────────────────────────────────────────────
# LOSS — Dice + CE/BCE + Boundary
# ─────────────────────────────────────────────────────────────────────────────

def boundary_mask(mask: np.ndarray, radius: int = 1):
    """Boundary (mép) của binary mask bằng morphological erosion."""
    import cv2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    eroded = cv2.erode(mask.astype(np.uint8), k)
    return (mask > eroded).astype(np.uint8)


def combined_seg_loss(logits, targets, ignore, dice_weight=0.5, ce_weight=0.3,
                      boundary_weight=0.2, smooth=1.0):
    """logits: (B,2,H,W)  targets: (B,H,W) long  ignore: (B,H,W) bool (padding)."""
    import torch
    import torch.nn.functional as F

    B, C, H, W = logits.shape
    target = targets
    valid = ~ignore
    # Bỏ padding khỏi loss
    logits_v = logits[valid]                     # (N,2)
    target_v = target[valid]                     # (N,)
    if logits_v.numel() == 0:
        return logits.sum() * 0.0

    # ── CE ──────────────────────────────────────────────────────────────
    ce = F.cross_entropy(logits_v, target_v)

    # ── Dice (soft) trên flood class ─────────────────────────────────────
    probs = F.softmax(logits, dim=1)[:, 1]       # (B,H,W) prob flood
    t1 = target.float()
    p = probs[valid]
    t = t1[valid].clamp(0, 1)
    inter = (p * t).sum()
    union = p.sum() + t.sum()
    dice = 1.0 - (2.0 * inter + smooth) / (union + smooth)

    # ── Boundary loss: chỉ đánh trên mép nước (mép target) ───────────────
    # Tính boundary bằng pooling (morphological) trên GPU
    def _dilate(T: torch.Tensor, kk: int = 3):
        return F.max_pool2d(T.unsqueeze(1), kk, stride=1, padding=kk // 2).squeeze(1)
    tt = t1.view(B, 1, H, W).float()
    dilated = _dilate(tt)
    # boundary = dilated - target (pixels vùng mép)
    tar_bnd = ((dilated > tt).float() | (tt > _dilate(-tt) + 1e-6).float()).bool()
    # prediction trên vùng mép phải khớp
    bp = probs[tar_bnd & valid]
    bt = target[tar_bnd & valid]
    if bp.numel() == 0:
        boundary = 0.0
    else:
        boundary = F.binary_cross_entropy(bp, bt.float().clamp(0, 1))

    loss = ce_weight * ce + dice_weight * dice + boundary_weight * boundary
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# METRICS — IoU, Dice, Precision, Recall, Boundary IoU
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(pred: np.ndarray, gt: np.ndarray, ignore: np.ndarray,
                    boundary_radius: int = 3):
    """pred/gt: (H,W) binary. ignore: padding region. Trả về dict metrics."""
    import cv2
    valid = ~ignore
    p, g = pred.astype(bool), gt.astype(bool)
    pv, gv = p & valid, g & valid

    tp = np.logical_and(pv, gv).sum()
    fp = np.logical_and(pv, ~gv).sum()
    fn = np.logical_and(~pv, gv).sum()

    iou = tp / (tp + fp + fn + 1e-7)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-7)
    prec = tp / (tp + fp + 1e-7)
    rec = tp / (tp + fn + 1e-7)

    # Boundary IoU: dilation mép GT, IoU trong vùng mép
    bG = cv2.dilate(
        boundary_mask(g.astype(np.uint8), radius=1),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * boundary_radius + 1,)*2),
    ).astype(bool) & valid
    if bG.sum() == 0:
        b_iou = 1.0
    else:
        inter = np.logical_and(pv, np.logical_and(gv, bG)).sum()
        union = np.logical_or(pv, np.logical_and(gv, bG)).sum()
        b_iou = inter / (union + 1e-7)

    return {"IoU": iou, "Dice": dice, "Precision": prec, "Recall": rec,
            "BoundaryIoU": b_iou}


# ─────────────────────────────────────────────────────────────────────────────
# PREPARE — chuẩn hóa cạnh dài (1536-2048), không méo ảnh
# ─────────────────────────────────────────────────────────────────────────────

def prepare(args):
    """Sinh folder dataset chuẩn hóa: cạnh dài → long_side, giữ aspect ratio."""
    import cv2
    import os

    src = Path(args.dataset_dir)
    dst = Path(args.output_dir)
    long_side = args.long_side
    for split in ("train", "val", "test"):
        s_img = src / split / "images"
        s_msk = src / split / "labels"
        if not s_img.exists():
            continue
        d_img = dst / split / "images"
        d_msk = dst / split / "labels"
        d_img.mkdir(parents=True, exist_ok=True)
        if s_msk.exists():
            d_msk.mkdir(parents=True, exist_ok=True)
        n = 0
        for img_path in sorted(s_img.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            scale = long_side / max(h, w)
            nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
            # Giữ aspect: chỉ scale cạnh dài, KHÔNG pad/méo trong prepare
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            stem = img_path.stem
            cv2.imwrite(str(d_img / f"{stem}.jpg"), img)

            msk = None
            for ext in (".png", ".jpg", ".jpeg", ".tif"):
                cand = s_msk / f"{stem}{ext}"
                if cand.exists():
                    msk = cv2.imread(str(cand), cv2.IMREAD_GRAYSCALE)
                    break
            if msk is not None and d_msk.parent.exists():
                msk = cv2.resize(msk, (nw, nh), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(d_msk / f"{stem}.png"), msk)
            n += 1
        log.info(f"  Prep {split}: {n} ảnh -> long_side={long_side} ({dst})")


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    class FloodSegDataset(Dataset):
        def __init__(self, items, flood_id, img_size, is_train=True):
            self.items, self.flood_id, self.img_size = items, flood_id, img_size
            self.is_train = is_train
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            ip, mp = self.items[idx]
            return _read_pair(ip, mp, self.flood_id, self.img_size, self.is_train)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = args.precision == "bf16" and torch.cuda.is_available() \
        and torch.cuda.is_bf16_supported() if hasattr(torch.cuda, "is_bf16_supported") \
        else (args.precision == "bf16" and torch.cuda.is_available())
    use_fp16 = not use_bf16 and args.precision in ("fp16", "auto") and torch.cuda.is_available()
    log.info(f"  Device: {device} | precision: "
             f"{'bf16' if use_bf16 else ('fp16' if use_fp16 else 'fp32')}")

    # Model
    ckpt = SEGFORMER_MODELS[args.model]
    from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
    model = SegformerForSemanticSegmentation.from_pretrained(
        ckpt, num_labels=2, ignore_mismatched_sizes=True)
    model.to(device)

    if args.model == "b0":
        log.info("  Model: SegFormer-B0 (baseline)")
    else:
        log.info("  Model: SegFormer-B2 (chính)")

    # Data
    train_items = load_dataset(args.dataset_dir, "train")
    val_items = load_dataset(args.dataset_dir, "val")
    if args.max_samples:
        train_items = train_items[:args.max_samples]
        val_items = val_items[:min(args.max_samples, len(val_items))]
    log.info(f"  Train: {len(train_items)} | Val: {len(val_items)}")

    train_ds = FloodSegDataset(train_items, args.flood_id, args.img_size, True)
    val_ds = FloodSegDataset(val_items, args.flood_id, args.img_size, False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_iou = 0.0
    steps = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        n_batch = 0
        optimizer.zero_grad()
        for imgs, msks, igns in train_dl:
            imgs, msks, igns = imgs.to(device), msks.to(device), igns.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16 if use_bf16 else torch.float16,
                                    enabled=(use_bf16 or use_fp16)):
                out = model(pixel_values=imgs)
                logits = torch.nn.functional.interpolate(
                    out.logits, size=msks.shape[1:], mode="bilinear", align_corners=False)
                loss = combined_seg_loss(logits, msks, igns)
            scaler.scale(loss).backward()
            steps += 1
            if steps % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            train_loss += loss.item()
            n_batch += 1

        # ── Validate ─────────────────────────────────────────────────────
        model.eval()
        agg = {"IoU": 0.0, "Dice": 0.0, "Precision": 0.0, "Recall": 0.0,
               "BoundaryIoU": 0.0}
        n_items = 0
        with torch.no_grad():
            for imgs, msks, igns in val_dl:
                imgs, msks, igns = imgs.to(device), msks.to(device), igns.to(device)
                with torch.amp.autocast("cuda",
                                        dtype=torch.bfloat16 if use_bf16 else torch.float16,
                                        enabled=(use_bf16 or use_fp16)):
                    out = model(pixel_values=imgs)
                    logits = torch.nn.functional.interpolate(
                        out.logits, size=msks.shape[1:], mode="bilinear", align_corners=False)
                pred = logits.argmax(dim=1)
                for bi in range(pred.shape[0]):
                    m = compute_metrics(
                        pred[bi].cpu().numpy(), msks[bi].cpu().numpy(),
                        igns[bi].cpu().numpy())
                    for k in agg:
                        agg[k] += m[k]
                    n_items += 1
        for k in agg:
            agg[k] /= max(n_items, 1)
        scheduler.step()
        avg_loss = train_loss / max(n_batch, 1)
        log.info(f"  Epoch {epoch+1}/{args.epochs} | loss={avg_loss:.4f} | "
                 f"IoU={agg['IoU']:.4f} | Dice={agg['Dice']:.4f} | "
                 f"B-IoU={agg['BoundaryIoU']:.4f} | P={agg['Precision']:.3f} | "
                 f"R={agg['Recall']:.3f}")

        if agg["IoU"] > best_iou:
            best_iou = agg["IoU"]
            model.save_pretrained(str(out_dir))
            seg = SegformerFeatureExtractor.from_pretrained(ckpt)
            seg.save_pretrained(str(out_dir))
            meta = {"model": ckpt, "img_size": args.img_size, "flood_id": args.flood_id,
                    "best_iou": best_iou, **{f"best_{k}": v for k, v in agg.items()}}
            (out_dir / "finetune_meta.json").write_text(json.dumps(meta, indent=2))
            log.info(f"    → Saved best (IoU={best_iou:.4f})")
            if best_iou >= args.target_iou:
                log.info(f"  Đạt target IoU={args.target_iou} → dừng sớm")
                break


def main():
    p = argparse.ArgumentParser(description="Finetune SegFormer water segmentation")
    sub = p.add_subparsers(dest="stage", required=True)

    prep = sub.add_parser("prepare", help="Chuẩn hóa dataset giữ aspect ratio")
    prep.add_argument("--dataset_dir", required=True)
    prep.add_argument("--output_dir", required=True)
    prep.add_argument("--long_side", type=int, default=1536,
                      help="Cạnh dài chuẩn hóa (khuyến nghị 1536-2048)")
    prep.set_defaults(func=prepare)

    tr = sub.add_parser("train")
    tr.add_argument("--dataset_dir", required=True)
    tr.add_argument("--output_dir", default="models/flood_segnet")
    tr.add_argument("--model", choices=["b0", "b2"], default="b2",
                    help="b2 = model chính, b0 = baseline")
    tr.add_argument("--img_size", type=int, default=768,
                    help="768×768 cho A4000 16GB; 1024 nếu VRAM ổn; 512 cho b0")
    tr.add_argument("--batch_size", type=int, default=2)
    tr.add_argument("--grad_accum", type=int, default=4)
    tr.add_argument("--epochs", type=int, default=60)
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--precision", default="auto",
                    help="bf16 | fp16 | auto (A4000 hỗ trợ bf16)")
    tr.add_argument("--flood_id", type=int, default=FLOODNET_FLOOD_ID,
                    help="Class id flood trong label: 6=FloodNet, 1=RescueNet")
    tr.add_argument("--max_samples", type=int, default=0)
    tr.add_argument("--target_iou", type=float, default=0.80)
    tr.set_defaults(func=train)

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.func(args)


if __name__ == "__main__":
    main()