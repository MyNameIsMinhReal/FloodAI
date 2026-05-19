# -*- coding: utf-8 -*-
"""
utils/depth_calibrator.py
==========================
Hiệu chỉnh (calibration) mực nước ước tính từ pipeline.

Model thường đo sai khoảng 10–15 cm so với thực tế. Calibrator học
hàm hiệu chỉnh đơn giản từ feedback của reviewer:

    calibrated_depth = a * predicted_depth + b

Có 3 calibration khác nhau theo loại vật tham chiếu:
    - has_person  → calibration A
    - has_vehicle → calibration B
    - no_reference→ calibration C

Dùng:
    cal = DepthCalibrator()
    cal.load("learning/calibration.json")
    corrected = cal.calibrate(predicted_depth=55, has_person=True)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("utils.calibrator")

_DEFAULT_CAL_PATH = Path("learning/calibration.json")

# Calibration groups
GROUP_PERSON   = "person"
GROUP_VEHICLE  = "vehicle"
GROUP_NOREF    = "no_reference"


class LinearCalibration:
    """y = a*x + b calibration đơn giản."""
    def __init__(self, a: float = 1.0, b: float = 0.0, n_samples: int = 0):
        self.a = a
        self.b = b
        self.n_samples = n_samples

    def apply(self, x: float) -> float:
        return max(0.0, self.a * x + self.b)

    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "n_samples": self.n_samples}

    @classmethod
    def from_dict(cls, d: dict) -> "LinearCalibration":
        return cls(a=d.get("a", 1.0), b=d.get("b", 0.0), n_samples=d.get("n_samples", 0))


class DepthCalibrator:
    """
    Học và áp dụng calibration cho depth predictions.

    Dùng:
        cal = DepthCalibrator()
        cal.load()

        # Calibrate single result
        corrected = cal.calibrate(predicted=55, result=depth_result)

        # Calibrate batch
        cal.calibrate_batch(depth_results)

        # Thêm feedback từ reviewer
        cal.add_sample(predicted=55, actual=68, result=depth_result)
        cal.fit()
        cal.save()
    """

    def __init__(self, cal_path: Path = _DEFAULT_CAL_PATH):
        self.cal_path = Path(cal_path)
        self._cals: Dict[str, LinearCalibration] = {
            GROUP_PERSON:  LinearCalibration(),
            GROUP_VEHICLE: LinearCalibration(),
            GROUP_NOREF:   LinearCalibration(),
        }
        self._samples: Dict[str, List[Tuple[float, float]]] = {
            g: [] for g in self._cals
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def load(self, path: Optional[Path] = None) -> bool:
        p = Path(path) if path else self.cal_path
        if not p.exists():
            log.debug("[Calibrator] Không tìm thấy %s — dùng identity calibration", p)
            return False
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            for group, d in data.get("calibrations", {}).items():
                self._cals[group] = LinearCalibration.from_dict(d)
            log.info("[Calibrator] Loaded calibration từ %s", p)
            return True
        except Exception as exc:
            log.warning("[Calibrator] Lỗi load %s: %s", p, exc)
            return False

    def save(self, path: Optional[Path] = None) -> Path:
        p = Path(path) if path else self.cal_path
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "calibrations": {g: c.to_dict() for g, c in self._cals.items()},
        }
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("[Calibrator] Saved → %s", p)
        return p

    # ── Calibrate ─────────────────────────────────────────────────────────────

    def calibrate(
        self,
        predicted: float,
        result: Optional[Any] = None,
        has_person: Optional[bool] = None,
        has_vehicle: Optional[bool] = None,
    ) -> float:
        """
        Áp dụng calibration cho một prediction.

        Args:
            predicted: chiều sâu từ model (cm)
            result: depth_result object (dùng để auto-detect group)
            has_person: override group detection
            has_vehicle: override group detection

        Returns:
            calibrated depth (cm)
        """
        group = self._detect_group(result, has_person, has_vehicle)
        cal = self._cals.get(group, LinearCalibration())
        calibrated = cal.apply(predicted)
        if abs(calibrated - predicted) > 1:
            log.debug("[Calibrator] %s: %.1f → %.1f cm (group=%s)",
                      getattr(result, "image_path", "?"), predicted, calibrated, group)
        return calibrated

    def calibrate_batch(self, results: List[Any]) -> List[Any]:
        """Áp dụng calibration cho toàn bộ depth_results in-place."""
        for r in results:
            pred = _get(r, "depth_cm", 0) or 0
            if pred <= 0:
                continue
            cal = self.calibrate(pred, r)
            if isinstance(r, dict):
                r["depth_cm_raw"] = pred
                r["depth_cm"] = cal
            else:
                try:
                    object.__setattr__(r, "depth_cm_raw", pred)
                    object.__setattr__(r, "depth_cm", cal)
                except Exception:
                    pass
        return results

    # ── Learning ──────────────────────────────────────────────────────────────

    def add_sample(
        self,
        predicted: float,
        actual: float,
        result: Optional[Any] = None,
        has_person: Optional[bool] = None,
        has_vehicle: Optional[bool] = None,
    ):
        """Thêm 1 feedback sample (predicted vs actual) để fit sau."""
        group = self._detect_group(result, has_person, has_vehicle)
        self._samples[group].append((predicted, actual))
        log.debug("[Calibrator] Sample added (group=%s): pred=%.1f actual=%.1f",
                  group, predicted, actual)

    def fit(self, min_samples: int = 5) -> Dict[str, bool]:
        """
        Fit linear regression cho từng group có đủ samples.

        Args:
            min_samples: số mẫu tối thiểu để fit (tránh overfitting)

        Returns:
            dict group → True nếu đã fit thành công
        """
        fitted = {}
        for group, samples in self._samples.items():
            if len(samples) < min_samples:
                log.info("[Calibrator] %s: chỉ có %d samples, cần %d để fit",
                         group, len(samples), min_samples)
                fitted[group] = False
                continue
            xs = [s[0] for s in samples]
            ys = [s[1] for s in samples]
            a, b = _lstsq(xs, ys)
            self._cals[group] = LinearCalibration(a=a, b=b, n_samples=len(samples))
            log.info("[Calibrator] Fit %s: a=%.3f b=%.2f (n=%d)", group, a, b, len(samples))
            fitted[group] = True
        return fitted

    def stats(self) -> dict:
        return {
            g: {"a": c.a, "b": c.b, "n_samples": c.n_samples}
            for g, c in self._cals.items()
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_group(
        result: Optional[Any],
        has_person: Optional[bool],
        has_vehicle: Optional[bool],
    ) -> str:
        if has_person is None and result is not None:
            refs = _get(result, "reference_objects", []) or []
            has_person = any(
                o.get("class") in ("person", "người") for o in refs if isinstance(o, dict)
            )
        if has_vehicle is None and result is not None:
            refs = _get(result, "reference_objects", []) or []
            has_vehicle = any(
                o.get("class") in ("car", "motorcycle", "xe", "truck")
                for o in refs if isinstance(o, dict)
            )
        if has_person:
            return GROUP_PERSON
        if has_vehicle:
            return GROUP_VEHICLE
        return GROUP_NOREF


def _get(obj: Any, key: str, default: Any) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _lstsq(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    """Ordinary Least Squares đơn giản: y = a*x + b."""
    import numpy as np
    x = np.array(xs)
    y = np.array(ys)
    n = len(x)
    if n < 2 or x.std() < 1e-9:
        return 1.0, 0.0
    a = (n * (x * y).sum() - x.sum() * y.sum()) / (n * (x**2).sum() - x.sum()**2)
    b = (y.sum() - a * x.sum()) / n
    return float(a), float(b)
