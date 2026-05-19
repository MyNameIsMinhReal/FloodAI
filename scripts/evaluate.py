#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/evaluate.py
====================
Benchmark / đánh giá độ chính xác của pipeline.

Cách dùng:
    python scripts/evaluate.py --dataset datasets/eval --config configs/cpu.yaml
    python scripts/evaluate.py --dataset datasets/eval --verbose

Cấu trúc thư mục eval:
    datasets/eval/
    ├── images/
    │   ├── flood_001.jpg
    │   └── ...
    └── labels.csv

Format labels.csv:
    image,level,depth_cm,location,note
    flood_001.jpg,knee,55,Hanoi,ngập tới đầu gối
    flood_002.jpg,ankle,25,Thai Nguyen,ngập nhẹ
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

# Đảm bảo import được từ root project
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.structured_logger import setup_logging

setup_logging(level=logging.WARNING)
log = logging.getLogger("evaluate")

LEVEL_ORDER = ["dry", "ankle", "knee", "waist", "chest", "submerged", "unknown"]


# ── Load labels ────────────────────────────────────────────────────────────────

def load_labels(csv_path: Path) -> dict:
    """
    Đọc labels.csv, trả về dict: filename -> {level, depth_cm, ...}
    """
    labels = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = Path(row["image"]).name
            labels[fname] = {
                "level":    row.get("level", "unknown").strip().lower(),
                "depth_cm": float(row["depth_cm"]) if row.get("depth_cm") else None,
                "location": row.get("location", ""),
                "note":     row.get("note", ""),
            }
    return labels


# ── Run pipeline on eval set ──────────────────────────────────────────────────

def run_eval_pipeline(image_dir: Path, cfg: dict) -> dict:
    """
    Chạy pipeline trên toàn bộ eval images. Trả về dict: filename -> result.
    """
    from pipeline.orchestrator import FloodPipeline, IMAGE_EXTENSIONS
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise ValueError(f"Không tìm thấy ảnh trong {image_dir}")

    print(f"  Đang phân tích {len(images)} ảnh eval...")
    t0 = time.time()
    pipeline = FloodPipeline(cfg)
    state = pipeline.run(images)
    elapsed = time.time() - t0
    print(f"  Pipeline xong trong {elapsed:.1f}s")

    results = {}
    for r in state.depth_results:
        img_key = Path(getattr(r, "original_path", "") or getattr(r, "image_path", "")).name
        if img_key:
            results[img_key] = r
    return results


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(labels: dict, predictions: dict, verbose: bool = False) -> dict:
    """
    Tính accuracy, MAE, confusion matrix giữa labels và predictions.
    """
    tp_exact   = 0
    tp_off1    = 0  # lệch 1 mức
    total      = 0
    mae_sum    = 0.0
    mae_count  = 0
    confusion: dict = {}  # {true_level: {pred_level: count}}
    missing    = []

    for fname, label in labels.items():
        true_level = label["level"]
        true_depth = label["depth_cm"]

        if fname not in predictions:
            missing.append(fname)
            continue

        r = predictions[fname]
        pred_level = getattr(r, "flood_level", "unknown") or "unknown"
        pred_depth = getattr(r, "depth_cm", None)

        total += 1

        # Confusion matrix
        confusion.setdefault(true_level, {})
        confusion[true_level][pred_level] = confusion[true_level].get(pred_level, 0) + 1

        # Accuracy
        if pred_level == true_level:
            tp_exact += 1

        # Off-by-1
        try:
            ti = LEVEL_ORDER.index(true_level)
            pi = LEVEL_ORDER.index(pred_level)
            if abs(ti - pi) <= 1:
                tp_off1 += 1
        except ValueError:
            pass

        # MAE depth
        if true_depth is not None and pred_depth is not None:
            mae_sum += abs(float(pred_depth) - float(true_depth))
            mae_count += 1

        if verbose:
            match = "✓" if pred_level == true_level else "✗"
            print(f"    {match}  {fname:40s}  true={true_level:10s} pred={pred_level:10s}"
                  + (f"  depth: true={true_depth}cm pred={pred_depth:.0f}cm" if true_depth else ""))

    acc_exact = tp_exact / total if total > 0 else 0.0
    acc_off1  = tp_off1  / total if total > 0 else 0.0
    mae       = mae_sum / mae_count if mae_count > 0 else None

    return {
        "total":        total,
        "missing":      len(missing),
        "accuracy":     acc_exact,
        "accuracy_off1": acc_off1,
        "mae_depth_cm": mae,
        "confusion":    confusion,
        "missing_files": missing,
    }


def print_report(metrics: dict):
    """In report đẹp ra console."""
    print("\n" + "=" * 60)
    print("  FLOOD PIPELINE — EVALUATION REPORT")
    print("=" * 60)
    print(f"  Tổng ảnh eval : {metrics['total']}")
    print(f"  Thiếu dự đoán : {metrics['missing']}")
    print(f"  Accuracy (exact)  : {metrics['accuracy']*100:.1f}%")
    print(f"  Accuracy (±1 mức) : {metrics['accuracy_off1']*100:.1f}%")
    if metrics["mae_depth_cm"] is not None:
        print(f"  MAE depth         : {metrics['mae_depth_cm']:.1f} cm")
    print()
    print("  Confusion Matrix (hàng = thực tế, cột = dự đoán):")
    levels = [l for l in LEVEL_ORDER if l in metrics["confusion"]]
    all_preds = sorted({p for row in metrics["confusion"].values() for p in row})
    # Header
    header = f"  {'':12s}" + "".join(f"{p:12s}" for p in all_preds)
    print(header)
    for true in levels:
        row_data = metrics["confusion"].get(true, {})
        row = f"  {true:12s}" + "".join(f"{row_data.get(p, 0):12d}" for p in all_preds)
        print(row)
    print("=" * 60)

    if metrics["missing_files"]:
        print(f"\n  ⚠ Không có dự đoán cho {len(metrics['missing_files'])} ảnh:")
        for f in metrics["missing_files"][:5]:
            print(f"    - {f}")
        if len(metrics["missing_files"]) > 5:
            print(f"    ... và {len(metrics['missing_files'])-5} ảnh khác")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Flood Pipeline — Evaluation Script")
    parser.add_argument("--dataset", "-d", required=True, help="Thư mục datasets/eval/")
    parser.add_argument("--config",  "-c", default="configs/cpu.yaml")
    parser.add_argument("--output",  "-o", default=None, help="Lưu kết quả JSON ra file")
    parser.add_argument("--verbose", "-v", action="store_true", help="In chi tiết từng ảnh")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    image_dir   = dataset_dir / "images"
    label_file  = dataset_dir / "labels.csv"

    if not image_dir.exists():
        print(f"✗ Không tìm thấy {image_dir}", file=sys.stderr)
        return 1
    if not label_file.exists():
        print(f"✗ Không tìm thấy {label_file}", file=sys.stderr)
        print("  Tạo labels.csv với format: image,level,depth_cm,location,note")
        return 1

    # Load config
    from utils.config_loader import load_config, apply_config_to_cfg
    raw_cfg = load_config(args.config)
    cfg = apply_config_to_cfg(raw_cfg, {"skip_drive": True, "skip_learning": True})

    # Load labels
    labels = load_labels(label_file)
    print(f"  Loaded {len(labels)} labels từ {label_file}")

    # Run pipeline
    predictions = run_eval_pipeline(image_dir, cfg)

    # Tính metrics
    metrics = compute_metrics(labels, predictions, verbose=args.verbose)
    print_report(metrics)

    # Lưu JSON nếu có --output
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  Kết quả JSON → {out}")

    return 0 if metrics["accuracy"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
