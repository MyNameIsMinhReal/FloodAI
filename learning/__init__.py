# -*- coding: utf-8 -*-
"""
Self-Learning System for Flood Analysis Pipeline

Modules:
- error_tracker:       Track và phân tích errors
- active_learner:      Active learning để chọn cases cần review
- adaptive_thresholds: Tự động điều chỉnh thresholds
- review_ui:           Web interface để review cases

Usage:
    from learning_update import SelfLearningPipeline

    learning = SelfLearningPipeline()
    cfg = learning.get_adaptive_config(base_cfg)
    learning.process_results(results, cfg, images)
    learning.close()

CLI:
    python learning_update.py --update    # Weekly learning update
    python learning/review_ui.py          # Launch review UI
"""

__version__ = "2.0.0"
__author__ = "Flood Analysis Team"

from .error_tracker import ErrorTracker, ErrorRecord
from .active_learner import ActiveLearnerV2 as ActiveLearner, ReviewCase
from .adaptive_thresholds import AdaptiveThresholdsV2 as AdaptiveThresholds

__all__ = [
    "ErrorTracker",
    "ErrorRecord",
    "ActiveLearner",
    "ReviewCase",
    "AdaptiveThresholds",
]
