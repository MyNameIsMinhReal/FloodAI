# -*- coding: utf-8 -*-
"""
core/model_loader.py  —  Singleton Model Loader
=================================================
Giải quyết vấn đề: mỗi stage load model riêng → tốn RAM + chậm startup.

Pattern: Singleton + Lazy Loading + Preload mode

Sử dụng:
    # Preload tất cả khi start pipeline
    loader = ModelLoader.instance()
    loader.preload(["yolo", "depth", "pose"])

    # Trong mỗi stage: lấy model đã cached
    yolo = loader.get("yolo")
    depth_model = loader.get("depth")

    # Sau khi dùng xong: giải phóng model nặng
    loader.unload("depth")

Model Lifecycle:
    preload → cached in memory → auto-unload if memory low
                                  └→ re-load on next get()
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional, Set

log = logging.getLogger("core.model_loader")


# ── Model Registry ─────────────────────────────────────────────────────────────

class _ModelEntry:
    """Metadata cho một model trong cache."""
    def __init__(self, name: str, loader_fn: Callable, config: dict):
        self.name       = name
        self.loader_fn  = loader_fn
        self.config     = config
        self.model: Any = None
        self.loaded_at: Optional[float] = None
        self.last_used: Optional[float] = None
        self.load_count: int = 0
        self.load_time_s: float = 0.0

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> Any:
        if self.is_loaded:
            self.last_used = time.time()
            return self.model

        log.info(f"  [ModelLoader] Loading '{self.name}'…")
        t0 = time.time()
        self.model = self.loader_fn(self.config)
        self.load_time_s = time.time() - t0
        self.loaded_at = time.time()
        self.last_used = time.time()
        self.load_count += 1
        log.info(f"  [ModelLoader] '{self.name}' loaded in {self.load_time_s:.2f}s")
        return self.model

    def unload(self) -> None:
        if not self.is_loaded:
            return
        try:
            import torch
            if hasattr(self.model, "cpu"):
                self.model.cpu()
        except ImportError:
            pass
        del self.model
        self.model = None
        gc.collect()
        _free_gpu()
        log.debug(f"  [ModelLoader] '{self.name}' unloaded")


# ── Singleton Loader ────────────────────────────────────────────────────────────

class ModelLoader:
    """
    Singleton model loader: đảm bảo mỗi model chỉ load 1 lần duy nhất.

    Thread-safe: dùng lock riêng cho từng model entry.

    Ví dụ:
        loader = ModelLoader.instance()
        loader.register("yolo", _load_yolo, cfg)
        loader.register("depth", _load_depth, cfg)

        loader.preload(["yolo", "depth"])      # load ngay khi start
        model = loader.get("yolo")             # lấy từ cache
    """

    _instance: Optional["ModelLoader"] = None
    _init_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "ModelLoader":
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._registry: Dict[str, _ModelEntry] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
        self._loaded_names: Set[str] = set()

    # ── Registration ───────────────────────────────────────────────────────────

    def register(self, name: str, loader_fn: Callable, config: dict | None = None) -> None:
        """
        Đăng ký một model với loader function.

        Args:
            name:      Tên model (e.g., "yolo", "depth", "pose")
            loader_fn: fn(config) → model object
            config:    Dict config truyền vào loader_fn
        """
        with self._global_lock:
            if name not in self._registry:
                self._registry[name] = _ModelEntry(name, loader_fn, config or {})
                self._locks[name] = threading.Lock()
                log.debug(f"  [ModelLoader] Registered '{name}'")

    def register_all_from_config(self, cfg: dict) -> None:
        """
        Tự động register các model phổ biến từ config.yaml.
        Chỉ register nếu chưa có.
        """
        model_cfg = cfg.get("models", {})

        # YOLO detector
        yolo_key = model_cfg.get("detector", cfg.get("yolo_model", "yolov8n.pt"))
        self.register("yolo", _loader_yolo, {"model_path": yolo_key,
                                              "conf": cfg.get("yolo_conf", 0.35)})

        # Pose model
        pose_key = model_cfg.get("pose", cfg.get("pose_model", "yolov8n-pose.pt"))
        self.register("pose", _loader_yolo, {"model_path": pose_key,
                                              "task": "pose"})

        # Depth model
        depth_key = model_cfg.get("depth", "depth-anything/Depth-Anything-V2-Small-hf")
        self.register("depth", _loader_depth, {"model_name": depth_key})

        # DINOv2
        dino_key = model_cfg.get("dino", "facebook/dinov2-small")
        self.register("dino", _loader_dino, {"model_name": dino_key})

        # Flood ResNet-18 (custom fine-tuned, optional)
        resnet_path = model_cfg.get("flood_resnet", "")
        if resnet_path:
            resnet_labels = model_cfg.get(
                "flood_resnet_labels", ["dry", "flood", "heavy_flood"]
            )
            self.register("flood_resnet", _loader_resnet18, {
                "model_path":  resnet_path,
                "num_classes": len(resnet_labels),
                "labels":      resnet_labels,
            })

        # VLM verifier (Qwen2.5-VL, optional — chỉ register khi config có)
        vlm_key = model_cfg.get("vlm", "")
        if vlm_key:
            self.register("vlm", _loader_vlm, {
                "model_name":   vlm_key,
                "device_pref":  (cfg.get("vlm_verify", {}) or {}).get("device", "auto"),
            })

        # [v4 Gap F] SAM2 mask segmentor (optional — chỉ register khi use_sam=true)
        if model_cfg.get("use_sam", False):
            self.register("sam", _loader_sam, {
                "model_type": model_cfg.get("sam_model_type", "sam2_hiera_large"),
                "model_path": model_cfg.get("sam_model_path", ""),
            })

    # ── Loading ────────────────────────────────────────────────────────────────

    def preload(self, names: list) -> None:
        """
        Preload một danh sách models ngay khi khởi động pipeline.
        Chạy tuần tự để tránh OOM khi load nhiều model cùng lúc.
        """
        log.info(f"  [ModelLoader] Preloading {len(names)} models: {names}")
        for name in names:
            try:
                self.get(name)
            except Exception as exc:
                log.warning(f"  [ModelLoader] Preload '{name}' failed: {exc}")

    def get(self, name: str) -> Any:
        """
        Lấy model (từ cache hoặc load mới nếu chưa có).
        Thread-safe: chỉ 1 thread load cùng lúc cho mỗi model.

        Raises:
            KeyError: nếu model chưa được register
        """
        if name not in self._registry:
            raise KeyError(
                f"Model '{name}' chưa được register. "
                f"Các model available: {list(self._registry.keys())}"
            )
        lock = self._locks[name]
        with lock:
            entry = self._registry[name]
            model = entry.load()
            self._loaded_names.add(name)
            return model

    def get_or_none(self, name: str) -> Optional[Any]:
        """Lấy model, trả về None nếu không có / load thất bại."""
        try:
            return self.get(name)
        except Exception as exc:
            log.warning(f"  [ModelLoader] get_or_none('{name}'): {exc}")
            return None

    # ── Unloading ──────────────────────────────────────────────────────────────

    def unload(self, name: str) -> None:
        """Giải phóng một model khỏi memory."""
        if name in self._registry:
            with self._locks[name]:
                self._registry[name].unload()
                self._loaded_names.discard(name)

    def unload_heavy(self) -> None:
        """
        Giải phóng các model nặng (depth, dino) sau khi xử lý xong.
        Giữ lại các model nhẹ (yolo) cho các stages tiếp theo.
        """
        HEAVY = {"depth", "dino", "sam"}
        for name in list(self._loaded_names):
            if name in HEAVY:
                self.unload(name)
                log.info(f"  [ModelLoader] Unloaded heavy model: '{name}'")

    def unload_all(self) -> None:
        """Giải phóng tất cả models."""
        for name in list(self._loaded_names):
            self.unload(name)

    # ── Status ─────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Trả về trạng thái loading của tất cả models."""
        return {
            name: {
                "loaded":      entry.is_loaded,
                "load_count":  entry.load_count,
                "load_time_s": round(entry.load_time_s, 2),
                "last_used":   entry.last_used,
            }
            for name, entry in self._registry.items()
        }

    def log_status(self) -> None:
        """Log trạng thái loading."""
        log.info("  [ModelLoader] Status:")
        for name, info in self.status().items():
            icon = "✅" if info["loaded"] else "⬜"
            log.info(
                f"    {icon} {name:12s} loaded={info['loaded']} "
                f"count={info['load_count']} time={info['load_time_s']}s"
            )


# ── Default loaders ────────────────────────────────────────────────────────────

def _loader_yolo(config: dict) -> Any:
    """Load YOLO model."""
    from ultralytics import YOLO
    model_path = config.get("model_path", "yolov8n.pt")
    model = YOLO(model_path)
    log.debug(f"  YOLO loaded: {model_path}")
    return model


def _loader_depth(config: dict) -> Any:
    """Load Depth Anything V2 từ HuggingFace."""
    import torch
    from transformers import pipeline as hf_pipeline
    model_name = config.get("model_name", "depth-anything/Depth-Anything-V2-Small-hf")
    device = 0 if torch.cuda.is_available() else -1  # HF pipeline: 0=cuda:0, -1=cpu
    pipe = hf_pipeline(task="depth-estimation", model=model_name, device=device)
    log.debug(f"  Depth model loaded: {model_name} (device={'cuda:0' if device == 0 else 'cpu'})")
    return pipe


def _loader_dino(config: dict) -> Any:
    """Load DINOv2 feature extractor."""
    import torch
    from transformers import AutoModel, AutoImageProcessor
    model_name = config.get("model_name", "facebook/dinov2-small")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    log.debug(f"  DINOv2 loaded: {model_name} (device={device})")
    return {"model": model, "processor": processor, "device": device}


def _loader_resnet18(config: dict) -> Any:
    """
    Load ResNet-18 fine-tuned cho flood classification.

    Trả về dict:
        {
            "model":      nn.Module (eval mode),
            "transform":  torchvision transform,
            "labels":     ["dry", "flood", "heavy_flood"],
            "device":     torch.device,
        }

    Raises:
        FileNotFoundError: nếu model_path không tồn tại
        RuntimeError: nếu num_classes không khớp với fc layer trong checkpoint
    """
    import torch
    import torchvision.models as tv_models
    import torchvision.transforms as T
    from pathlib import Path

    model_path = config.get("model_path", "")
    num_classes = config.get("num_classes", 3)
    labels      = config.get("labels", ["dry", "flood", "heavy_flood"])

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"[flood_resnet] Model không tìm thấy: {model_path}\n"
            f"  → Copy file .pth vào đường dẫn trên hoặc sửa config.yaml"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dựng kiến trúc ResNet-18 với fc head khớp checkpoint
    # Checkpoint dùng fc.1.weight → fc là Sequential([layer0_no_params, Linear])
    # Dùng Dropout(0.0) ở index 0 để khớp index mà không ảnh hưởng inference
    model = tv_models.resnet18(weights=None)
    setattr(model, "fc", torch.nn.Sequential(
        torch.nn.Dropout(p=0.0),        # index 0 — không có weight, khớp checkpoint
        torch.nn.Linear(512, num_classes)  # index 1 — fc.1.weight / fc.1.bias
    ))

    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    # Checkpoint có thể là raw state_dict hoặc wrapped {"model_state": ...}
    if "model_state" in state_dict:
        state_dict = state_dict["model_state"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # ImageNet-standard preprocessing (ResNet-18 training convention)
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ])

    log.info(
        f"  [flood_resnet] Loaded ResNet-18 ({num_classes} classes: {labels}) "
        f"from {path.name} on {device}"
    )
    return {
        "model":     model,
        "transform": transform,
        "labels":    labels,
        "device":    device,
    }


def _loader_vlm(config: dict) -> Any:
    """
    Load VLM (Qwen2.5-VL / Qwen2-VL / generic vision2seq).

    Trả về dict:
        {"model": model, "processor": processor, "device": torch.device,
         "verifier": VLMVerifier singleton đã warm}

    Dùng bởi vlm_verify stage — stage tự quản lazy load nếu model này
    chưa được preload.
    """
    import torch
    from transformers import AutoProcessor

    name = config.get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct")
    pref = str(config.get("device_pref", "auto")).lower()
    if pref == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    log.info(f"  [vlm] Loading {name} → {device} …")
    processor = AutoProcessor.from_pretrained(
        name, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28,
    )

    model = None
    for cls_name in ("Qwen2_5_VLForConditionalGeneration",
                     "Qwen2VLForConditionalGeneration"):
        try:
            import transformers as tf
            cls = getattr(tf, cls_name, None)
            if cls is None:
                continue
            model = cls.from_pretrained(name, torch_dtype=dtype)
            break
        except Exception as exc:
            log.debug(f"  [vlm] {cls_name} fail: {exc}")
    if model is None:
        from transformers import AutoModelForVision2Seq
        model = AutoModelForVision2Seq.from_pretrained(name, torch_dtype=dtype)

    model.to(device)
    model.eval()

    # Warm singleton verifier để stage dùng lại (không load 2 lần)
    try:
        from depth_analysis.vlm_verifier import get_vlm_verifier
        verifier = get_vlm_verifier()
        verifier._model, verifier._processor, verifier._device = \
            model, processor, device
    except Exception as exc:
        log.debug(f"  [vlm] Verifier warm-up skip: {exc}")
        verifier = None

    return {
        "model": model, "processor": processor, "device": device,
        "verifier": verifier,
    }


def _free_gpu():
    """Giải phóng GPU memory."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _loader_sam(config: dict) -> Any:
    """
    Load SAM2 mask segmentor (Meta Segment Anything 2).

    Trả về dict:
        {"model": sam_model, "device": torch.device, "model_type": str}

    SAM2 dùng ultralytics.SAM() nếu có, hoặc sam2 package.
    """
    import torch

    model_type = config.get("model_type", "sam2_hiera_large")
    model_path = config.get("model_path", "")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ưu tiên ultralytics SAM2 (cài đơn giản: pip install ultralytics)
    try:
        from ultralytics import SAM
        # ultralytics SAM2: SAM("sam2_hiera_large.pt") auto-downloads
        if model_path:
            sam_model = SAM(model_path)
        else:
            # Map shorthand → ultralytics filename
            _ul_map = {
                "sam2_hiera_large": "sam2_hiera_large.pt",
                "sam2_hiera_small": "sam2_hiera_small.pt",
            }
            sam_model = SAM(_ul_map.get(model_type, model_type))
        log.info(f"  [SAM2] Loaded via ultralytics: {model_type} → {device}")
        return {"model": sam_model, "device": device, "model_type": model_type}
    except ImportError:
        pass

    # Fallback: sam2 package (Meta official)
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        if model_path:
            sam2_model = build_sam2(model_type, model_path, device=str(device))
            predictor = SAM2ImagePredictor(sam2_model)
            log.info(f"  [SAM2] Loaded via sam2 package: {model_type} → {device}")
            return {"model": predictor, "device": device, "model_type": model_type}
    except ImportError:
        pass

    raise RuntimeError(
        f"SAM2 model '{model_type}' không load được.\n"
        f"  → Cài: pip install ultralytics (đơn giản nhất)\n"
        f"  → Hoặc: pip install sam2 (Meta official)"
    )
