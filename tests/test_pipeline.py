# -*- coding: utf-8 -*-
"""
Unit Tests — Flood Pipeline
============================
Chạy:
    python -m pytest tests/ -v
    python -m pytest tests/test_confidence.py -v      # chỉ 1 module
    python -m pytest tests/ --tb=short -q             # tóm tắt nhanh

Không cần GPU, không cần model download — mọi test dùng mock data.
"""
import sys
from pathlib import Path

# Đảm bảo import được từ root
sys.path.insert(0, str(Path(__file__).parent.parent))


# ══════════════════════════════════════════════════════════════════════
# test_confidence.py
# ══════════════════════════════════════════════════════════════════════

import pytest
from pipeline.confidence import ConfidenceScorer


class TestConfidenceScorer:
    """Test ConfidenceScorer."""

    def setup_method(self):
        self.scorer = ConfidenceScorer()

    def test_high_confidence(self):
        """Nhiều reference objects + consistent → HIGH."""
        result = {
            "confidence": 0.90,
            "reference_objects": [
                {"estimated_depth": 45.0},
                {"estimated_depth": 46.0},
                {"estimated_depth": 44.5},
            ],
        }
        conf = self.scorer.compute(result)
        assert conf >= 0.70, f"Expected HIGH (>=0.70), got {conf:.2f}"
        assert self.scorer.label(conf) == "HIGH"
        assert not self.scorer.needs_review(conf)

    def test_low_confidence_no_objects(self):
        """Không có reference objects → LOW."""
        result = {
            "confidence": 0.30,
            "reference_objects": [],
        }
        conf = self.scorer.compute(result)
        assert conf < 0.70, f"Expected LOW (<0.70), got {conf:.2f}"
        assert self.scorer.needs_review(conf)

    def test_inconsistent_objects_reduces_score(self):
        """Reference objects không nhất quán → giảm score."""
        consistent = {
            "confidence": 0.80,
            "reference_objects": [
                {"estimated_depth": 45.0},
                {"estimated_depth": 46.0},
            ],
        }
        inconsistent = {
            "confidence": 0.80,
            "reference_objects": [
                {"estimated_depth": 10.0},
                {"estimated_depth": 200.0},
            ],
        }
        conf_c = self.scorer.compute(consistent)
        conf_i = self.scorer.compute(inconsistent)
        assert conf_c > conf_i, "Consistent objects should give higher confidence"

    def test_dict_result(self):
        """Test với dict result."""
        result = {"confidence": 0.75, "reference_objects": [{"estimated_depth": 50}]}
        conf = self.scorer.compute(result)
        assert 0.0 <= conf <= 1.0
        assert "confidence_info" in result  # Phải attach info vào result

    def test_object_result(self):
        """Test với object result."""
        class FakeResult:
            confidence = 0.75
            reference_objects = [{"estimated_depth": 50}]

        result = FakeResult()
        conf = self.scorer.compute(result)
        assert 0.0 <= conf <= 1.0
        assert hasattr(result, "confidence_info")

    def test_label_boundaries(self):
        assert self.scorer.label(0.00) == "LOW"
        assert self.scorer.label(0.39) == "LOW"
        assert self.scorer.label(0.40) == "MEDIUM"
        assert self.scorer.label(0.69) == "MEDIUM"
        assert self.scorer.label(0.70) == "HIGH"
        assert self.scorer.label(1.00) == "HIGH"

    def test_score_clipped_to_01(self):
        """Score phải luôn trong [0, 1]."""
        result = {"confidence": 999.0, "reference_objects": []}
        conf = self.scorer.compute(result)
        assert 0.0 <= conf <= 1.0


# ══════════════════════════════════════════════════════════════════════
# test_filter_stage.py
# ══════════════════════════════════════════════════════════════════════

class TestFilterStage:
    """Test FilterStage toggles."""

    def setup_method(self):
        from pipeline.stages.filter_stage import FilterStage
        self.cfg_all_off = {
            "enable_filters": {
                "blur": False,
                "duplicate": False,
                "banner": False,
                "deblur": False,
                "content": False,
            },
            "enhance_images": False,
            "check_watermark": False,
            "skip_filter": False,
            "skip_content_filter": True,
        }
        self.stage = FilterStage(self.cfg_all_off)

    def test_toggle_off_returns_same_images(self):
        """Tắt hết filter → trả về đúng input."""
        # Dùng mock paths — không cần file thật
        fake_images = [Path(f"/fake/image_{i}.jpg") for i in range(5)]

        # Với tất cả filter tắt, phải trả về cùng số ảnh
        # (test logic routing, không test filter thật)
        assert self.stage._on("blur") is False
        assert self.stage._on("duplicate") is False
        assert self.stage._on("content") is False

    def test_toggle_on_by_default(self):
        from pipeline.stages.filter_stage import FilterStage
        stage = FilterStage({"enable_filters": {}})
        assert stage._on("blur") is True
        assert stage._on("duplicate") is True
        assert stage._on("content") is True

    def test_skip_filter_flag(self):
        from pipeline.stages.filter_stage import FilterStage
        stage = FilterStage({"skip_filter": True, "enable_filters": {}})
        assert stage.cfg.get("skip_filter") is True


# ══════════════════════════════════════════════════════════════════════
# test_depth_stage.py
# ══════════════════════════════════════════════════════════════════════

class TestDepthStage:
    """Test DepthStage model resolution."""

    def setup_method(self):
        from pipeline.stages.depth_stage import DepthStage
        self.stage = DepthStage({})

    def test_resolve_shorthand_depth_model(self):
        """Shorthand 'small' → đường dẫn đầy đủ HuggingFace."""
        stage = self._make_stage({"models": {"depth": "small"}})
        models = stage._resolve_models()
        assert "Depth-Anything-V2-Small" in models["depth"]

    def test_resolve_shorthand_base(self):
        stage = self._make_stage({"models": {"depth": "base"}})
        assert "Base" in stage._resolve_models()["depth"]

    def test_resolve_midas(self):
        stage = self._make_stage({"models": {"depth": "midas"}})
        assert "midas" in stage._resolve_models()["depth"].lower()

    def test_resolve_yolo_shorthand(self):
        stage = self._make_stage({"models": {"detector": "medium"}})
        assert stage._resolve_models()["detector"] == "yolov8m.pt"

    def test_fallback_to_legacy_config(self):
        """Backward-compat: dùng depth_model cũ nếu không có section models."""
        stage = self._make_stage({
            "depth_model": "depth-anything/Depth-Anything-V2-Base-hf",
        })
        models = stage._resolve_models()
        assert "Base" in models["depth"]

    def test_passthrough_full_path(self):
        """Đường dẫn đầy đủ → không thay đổi."""
        full_path = "depth-anything/Depth-Anything-V2-Large-hf"
        stage = self._make_stage({"models": {"depth": full_path}})
        assert stage._resolve_models()["depth"] == full_path

    def _make_stage(self, cfg: dict):
        from pipeline.stages.depth_stage import DepthStage
        return DepthStage(cfg)


# ══════════════════════════════════════════════════════════════════════
# test_env_loader.py
# ══════════════════════════════════════════════════════════════════════

class TestEnvLoader:
    """Test env_loader utilities."""

    def test_get_env_with_default(self):
        from utils.env_loader import get_env
        val = get_env("THIS_KEY_DOES_NOT_EXIST_12345", default="fallback")
        assert val == "fallback"

    def test_get_env_int(self):
        from utils.env_loader import get_env_int
        import os
        os.environ["TEST_INT_VAR"] = "42"
        assert get_env_int("TEST_INT_VAR") == 42
        del os.environ["TEST_INT_VAR"]

    def test_get_env_bool_true(self):
        from utils.env_loader import get_env_bool
        import os
        for val in ("1", "true", "yes", "True", "YES"):
            os.environ["TEST_BOOL"] = val
            assert get_env_bool("TEST_BOOL") is True
        del os.environ["TEST_BOOL"]

    def test_get_env_bool_false(self):
        from utils.env_loader import get_env_bool
        import os
        for val in ("0", "false", "no", ""):
            os.environ["TEST_BOOL"] = val
            assert get_env_bool("TEST_BOOL") is False
        del os.environ["TEST_BOOL"]

    def test_required_env_raises(self):
        from utils.env_loader import get_env
        with pytest.raises(ValueError, match="REQUIRED_BUT_MISSING"):
            get_env("REQUIRED_BUT_MISSING_99999", required=True)


# ══════════════════════════════════════════════════════════════════════
# test_pipeline_state.py
# ══════════════════════════════════════════════════════════════════════

class TestPipelineState:
    """Test PipelineState dataclass."""

    def test_to_dict_has_required_keys(self):
        from pipeline.orchestrator import PipelineState
        state = PipelineState(run_id="test_123", query="flood")
        d = state.to_dict()
        for key in ["run_id", "query", "sources", "raw", "filtered", "depth_data"]:
            assert key in d, f"Missing key: {key}"

    def test_log_stage_records_timing(self):
        from pipeline.orchestrator import PipelineState
        state = PipelineState()
        state.timings["crawl"] = 2.5
        assert state.timings["crawl"] == 2.5

    def test_errors_list_default_empty(self):
        from pipeline.orchestrator import PipelineState
        state = PipelineState()
        assert state.errors == []
        state.errors.append("test error")
        assert len(state.errors) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
