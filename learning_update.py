# -*- coding: utf-8 -*-

import json
import logging
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path
from typing import List

log = logging.getLogger("self_learning")

from learning.active_learner import ActiveLearnerV2 as ActiveLearner, ReviewCase
from learning.adaptive_thresholds import AdaptiveThresholdsV2 as AdaptiveThresholds
from learning.error_tracker import ErrorTracker


class SelfLearningPipeline:
    def __init__(self):
        self.learner       = ActiveLearner()
        self.thresholds    = AdaptiveThresholds()
        self.error_tracker = ErrorTracker()
        log.info("[SelfLearning v2] Khởi tạo (scoring + level tracking + log_prediction)")

    # ── Public API ────────────────────────────────────────────────────────────

    def get_adaptive_config(self, base_cfg: dict) -> dict:
        cfg = base_cfg.copy()
        cfg["yolo_conf"]         = self.thresholds.get("yolo_conf",         cfg.get("yolo_conf", 0.35))
        cfg["blur_thresh"]       = self.thresholds.get("blur_threshold",    cfg.get("blur_thresh", 100.0))
        cfg["watermark_conf"]    = self.thresholds.get("watermark_conf",    0.45)
        cfg["enhance_threshold"] = self.thresholds.get("enhance_threshold", 80.0)
        cfg["min_confidence"]    = self.thresholds.get("min_confidence",    0.5)
        log.info("[SelfLearning v2] Applied adaptive thresholds")
        return cfg

    def process_results(self, depth_results: List, _cfg: dict, image_paths: List[Path]) -> None:
        log.info(f"[SelfLearning v2] Processing {len(depth_results)} results")
        review_count = 0

        expired = self.learner.expire_old_cases()
        if expired:
            log.info(f"[SelfLearning v2] Expired {expired} old cases")

        for i, result in enumerate(depth_results):
            if i >= len(image_paths):
                break
            image_path = Path(image_paths[i])

            # [MỚI v2] Ghi nhận mỗi ảnh vào processed_log → tính error rate thực
            pred_depth = float(getattr(result, "water_height_cm", 0) or 0)
            pred_level = getattr(result, "flood_level", "") or ""
            confidence = float(getattr(result, "confidence", 0) or 0)
            self.error_tracker.log_prediction(
                image_path=str(image_path),
                predicted_level=pred_level,
                predicted_depth=pred_depth,
                confidence=confidence,
            )

            features = self._extract_features(result, image_path)
            needs, reason, priority = self.learner.needs_review(result, features)
            if needs:
                # [MỚI v2] Tính score để sắp xếp queue
                score = self.learner.score_case(result, features, reason)
                case = ReviewCase(
                    image_path      = str(image_path),
                    predicted_depth = pred_depth,
                    predicted_level = pred_level,
                    confidence      = confidence,
                    review_reason   = reason,
                    priority        = priority,
                    score           = score,
                    features        = json.dumps(features),
                )
                added = self.learner.add_to_queue(case)
                if added is not False:
                    review_count += 1

        log.info(f"[SelfLearning v2] Added {review_count} cases to review queue")
        stats = self.learner.get_review_stats()
        log.info(f"[SelfLearning v2] Queue: {stats.get('pending',0)} pending, "
                 f"{stats.get('reviewed',0)} reviewed")

        trend = self.thresholds.get_trend()
        log.info(f"[SelfLearning v2] Accuracy trend: {trend.get('direction','unknown')} "
                 f"({trend.get('weeks_of_data',0)} weeks, "
                 f"error_rate={trend.get('current_error_rate',0):.1%})")

    def _extract_features(self, result, image_path: Path) -> dict:

        features = {}

        try:
            import cv2
            import numpy as np
            from PIL import Image as PILImage

            # [FIX] Context manager để đảm bảo PIL Image được close
            with PILImage.open(image_path) as img:
                arr = np.array(img)

            gray = np.mean(arr, axis=2) if arr.ndim == 3 else arr
            features["brightness"]   = float(np.mean(gray))
            features["aspect_ratio"] = arr.shape[1] / max(arr.shape[0], 1)

            gray_cv = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
            features["blur_score"] = float(cv2.Laplacian(gray_cv, cv2.CV_64F).var())

            # [MỚI v2] is_night: tối + hoặc timestamp ban đêm
            brightness = features["brightness"]
            features["is_night"] = bool(brightness < 60)

        except Exception as e:
            log.warning(f"[SelfLearning v2] Feature extraction failed: {e}")

        # [BUG FIX] Đọc từ detected_objects (list of dict) thay vì attr không tồn tại
        detected_objects = getattr(result, "detected_objects", []) or []

        features["num_reference_objects"] = len(detected_objects)
        features["num_people"] = sum(
            1 for d in detected_objects if d.get("class_name") == "person"
        )
        features["has_pose"] = any(
            d.get("keypoints") is not None
            for d in detected_objects
            if d.get("class_name") == "person"
        )

        depth_estimates = [
            float(d.get("water_height_cm", 0))
            for d in detected_objects
            if d.get("water_height_cm") is not None and d.get("water_height_cm") > 0
        ]
        features["depth_estimates"] = depth_estimates

        # [MỚI v2] Reference object class names để track accuracy per object type
        features["reference_objects"] = list({
            d.get("class_name", "") for d in detected_objects
            if d.get("class_name") not in ("", None)
        })

        # [BUG FIX] depth_flood_pct thay vì water_level_pct (không tồn tại)
        features["water_level_pct"] = float(getattr(result, "depth_flood_pct", 0) or 0) / 100.0

        # Watermark metadata
        meta_path = image_path.parent / f"{image_path.stem}_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                features["has_watermark"]   = meta.get("watermark_detected", False)
                features["watermark_count"] = len(meta.get("watermarks", []))
            except Exception:
                pass

        return features

    def _print_threshold_status(self) -> None:
        adjusted = self.thresholds.adjust()
        if adjusted:
            print("✓ Điều chỉnh thresholds:")
            for a in adjusted:
                print(f"  - {a}")
        else:
            print("✓ Không cần điều chỉnh thresholds")
        trend = self.thresholds.get_trend()
        arrow = {"improving": "📈", "worsening": "📉", "stable": "➡️"}.get(trend["direction"], "?")
        print(f"\n{arrow} Xu hướng: {trend['direction']} "
              f"({trend.get('weeks_of_data',0)} tuần, "
              f"error rate: {trend.get('current_error_rate',0):.1%})")

    def _print_error_stats(self) -> None:
        stats = self.error_tracker.get_error_stats(days=30)
        processed = self.error_tracker.get_total_processed(days=30)
        print(f"\n📊 30 ngày: {stats['total']} lỗi / {processed} ảnh đã xử lý "
              f"({stats['total']/max(processed,1):.1%} error rate)")
        print(f"   Depth MAE: {stats['depth_mae']:.1f} cm | RMSE: {stats['depth_rmse']:.1f} cm")
        for t, c in stats.get("by_type", {}).items():
            print(f"   - {t}: {c}")

    def _print_loss_metrics(self) -> None:
        def _bar(val: float, length: int = 20) -> str:
            filled = int(val * length)
            return "█" * filled + "░" * (length - filled)

        try:
            loss_stats = self.error_tracker.get_loss_stats(days=30)
            n_gt = loss_stats.get("n", 0)
            if n_gt == 0:
                print("\n🎯 Loss Function: chưa có reviewed cases với ground truth")
                print("   → Review ảnh trong queue và nhập actual_depth/level để kích hoạt")
                return
            print(f"\n🎯 Loss Function (30 ngày, {n_gt} reviewed cases):")
            avg_c = loss_stats.get("avg_composite",  0.0)
            avg_d = loss_stats.get("avg_depth_loss", 0.0)
            avg_l = loss_stats.get("avg_level_loss", 0.0)
            avg_k = loss_stats.get("avg_conf_loss",  0.0)
            acc   = loss_stats.get("level_accuracy", 0.0)
            mae   = loss_stats.get("depth_mae_cm",   0.0)
            print(f"   Composite  {_bar(avg_c)} {avg_c:.3f}")
            print(f"   Depth      {_bar(avg_d)} {avg_d:.3f}  (MAE={mae:.1f}cm)")
            print(f"   Level      {_bar(avg_l)} {avg_l:.3f}  (acc={acc:.0%})")
            print(f"   Confidence {_bar(avg_k)} {avg_k:.3f}")
            for k, v in loss_stats.get("recommendations", {}).items():
                print(f"      [{k}] {v}")
            for w in loss_stats.get("worst_cases", [])[:3]:
                name = Path(w.get("path", "?")).name
                print(f"      {name}: loss={w['loss']:.3f} depth_err={w.get('depth_err_cm', '?')}cm")
        except Exception as exc:
            log.warning(f"[SelfLearning v2] Loss stats failed: {exc}")

    def _print_level_accuracy(self) -> None:
        level_acc = self.error_tracker.get_accuracy_by_level(days=90)
        if not level_acc:
            return
        print("\n📐 Accuracy theo flood level (90 ngày):")
        for lvl, data in sorted(level_acc.items(), key=lambda x: x[1]["accuracy"]):
            bar = "█" * int(data["accuracy"] * 10) + "░" * (10 - int(data["accuracy"] * 10))
            print(f"   {lvl:12s} {bar} {data['accuracy']:.0%} ({data['errors']}/{data['total']})")
        ref_stats = self.error_tracker.get_reference_object_stats(days=90)
        if ref_stats:
            print("\n🔧 Độ sai theo vật thể tham chiếu (top 5):")
            for obj, data in list(ref_stats.items())[:5]:
                print(f"   {obj}: {data['avg_depth_error']:.1f} cm avg error ({data['count']} lỗi)")

    def _print_queue_status(self) -> None:
        for s in self.error_tracker.get_problematic_scenarios():
            print(f"  - {s}")
        dist = self.learner.get_reason_distribution()
        if dist:
            print("\n📋 Lý do review queue:")
            for reason, cnt in list(dist.items())[:8]:
                print(f"  - {reason}: {cnt}")
        under = self.learner.get_underrepresented_levels()
        if under:
            print(f"\n⚡ Flood levels thiếu training data: {', '.join(under)}")
            print("   → Nên thu thập thêm ảnh cho các level này")
        rs = self.learner.get_review_stats()
        print(f"\n📝 Queue: {rs.get('pending',0)} pending | "
              f"{rs.get('reviewed',0)} reviewed | {rs.get('expired',0)} expired")
        if rs.get("pending", 0) > 0:
            print("  💡 Chạy: python learning/review_ui.py")

    def update_learning(self, verbose: bool = True):
        if verbose:
            print("\n" + "=" * 60)
            print("  Self-Learning Update — v2")
            print("=" * 60 + "\n")
        self._print_threshold_status()
        self._print_error_stats()
        self._print_loss_metrics()
        self._print_level_accuracy()
        self._print_queue_status()

    def export_training_data(self, output_dir: str = "learning/training_data"):
        out = Path(output_dir)
        print(f"\n📦 Exporting training data → {out}")
        n, _ = self.error_tracker.export_for_finetuning(out)
        print(f"✓ Exported {n} validated images")
        if n >= 100:
            print("  💡 Đủ data để fine-tune!")
        return n

    def close(self):
        self.learner.close()
        self.error_tracker.close()


# ── CLI ───────────────────────────────────────────────────────────────────────
def _cmd_stats(learning: "SelfLearningPipeline") -> None:
    print("=" * 60)
    es = learning.error_tracker.get_error_stats(30)
    rs = learning.learner.get_review_stats()
    th = learning.thresholds.get_all()
    tp = learning.error_tracker.get_total_processed(30)
    print(f"📊 Errors: {es['total']}/{tp} | MAE: {es['depth_mae']}cm | "
          f"By type: {es.get('by_type',{})}")
    print(f"📝 Queue: {rs}")
    print(f"⚙️  Thresholds: {th}")
    print(f"📈 Trend: {learning.thresholds.get_trend()}")
    print(f"📜 Last adjustments: {learning.thresholds.get_adjustment_history(5)}")
    print("\n📐 Accuracy by level:")
    for lvl, data in learning.error_tracker.get_accuracy_by_level().items():
        print(f"  {lvl}: {data['accuracy']:.0%} ({data['total']} processed)")


def _cmd_review() -> None:
    print("🌐 Đang khởi động Review UI tại http://localhost:5000 ...")
    proc = subprocess.Popen(
        [sys.executable, "learning/review_ui.py"],
        creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0,
    )
    max_wait = 30
    for i in range(max_wait):
        time.sleep(1)
        try:
            urllib.request.urlopen("http://localhost:5000", timeout=1)
            break
        except Exception:
            if i < max_wait - 1:
                print(f"  ⏳ Đang chờ server... ({i+1}/{max_wait}s)", end="\r")
    else:
        print("\n⚠️  Server chưa phản hồi sau 30s — thử mở trình duyệt thủ công.")
    print(f"\n✅ Review UI đang chạy (PID {proc.pid}).")
    webbrowser.open("http://localhost:5000")


def main():
    import argparse
    p = argparse.ArgumentParser(description="Flood Pipeline Self-Learning v2")
    p.add_argument("--update", action="store_true")
    p.add_argument("--export", action="store_true")
    p.add_argument("--stats",  action="store_true")
    p.add_argument("--review", action="store_true")
    args = p.parse_args()

    learning = SelfLearningPipeline()

    if args.update:
        learning.update_learning(verbose=True)
    elif args.export:
        learning.export_training_data()
    elif args.stats:
        _cmd_stats(learning)
    elif args.review:
        _cmd_review()
    else:
        p.print_help()

    learning.close()


if __name__ == "__main__":
    main()
