# -*- coding: utf-8 -*-
"""
depth_analysis/person_tracker.py
-----------------------------------
Simple IoU-based person tracker (BYTETrack-lite) — không cần heavy deps.

Lợi ích:
  - Track 1 người qua nhiều frame → stable pose + raincoat detection
  - Majority voting raincoat: lưu 5 frames gần nhất, vote kết quả
  - Smooth keypoints theo thời gian: kp_t = 0.7*kp_t + 0.3*kp_t-1
  - Không phụ thuộc bytetrack/deepsort (chỉ cần numpy + scipy optional)

Cách dùng:
    tracker = PersonTracker(max_age=10, history_len=5)

    # Mỗi frame:
    track_results = tracker.update(
        bboxes   = [[x1,y1,x2,y2], ...],
        scores   = [0.85, 0.72, ...],
        keypoints= [kpts_17x3, ...],       # optional
        raincoats= [True, False, ...],     # optional
    )

    for tr in track_results:
        print(tr.track_id, tr.smoothed_keypoints, tr.raincoat_vote)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, cast

import numpy as np

log = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TrackState:
    """Trạng thái 1 track."""
    track_id:    int
    bbox:        List[int]        # [x1, y1, x2, y2] hiện tại
    score:       float
    age:         int              # số frame đã sống
    hits:        int              # số lần detect liên tiếp
    miss:        int              # số frame miss liên tiếp

    # Keypoint smoothing
    keypoints:   Optional[np.ndarray]  # (17, 3)

    # History buffers
    raincoat_history: List[bool]  = field(default_factory=list)
    pose_history:     List[str]   = field(default_factory=list)
    depth_history:    List[float] = field(default_factory=list)
    height_history:   List[float] = field(default_factory=list)  # [IMPROVE] Height smoothing

    # Computed outputs
    raincoat_vote: Optional[bool]  = None    # majority vote
    raincoat_conf: float           = 0.0
    smoothed_keypoints: Optional[np.ndarray] = None
    ema_depth: Optional[float]     = None    # [IMPROVE] EMA smoothed depth
    ema_height: Optional[float]    = None    # [IMPROVE] EMA smoothed height


@dataclass
class TrackResult:
    """Output cho 1 tracked person."""
    track_id:    int
    bbox:        List[int]
    score:       float
    smoothed_keypoints: Optional[np.ndarray]
    raincoat_vote: Optional[bool]
    raincoat_conf: float
    pose_vote:     Optional[str]
    depth_median:  Optional[float]
    depth_ema:     Optional[float]     # [IMPROVE] EMA smoothed depth
    height_ema:    Optional[float]     # [IMPROVE] EMA smoothed height
    is_new_track:  bool


# ── Tracker ───────────────────────────────────────────────────────────────────

class PersonTracker:
    """
    IoU-based multi-person tracker với temporal smoothing.

    Args:
        max_age:       Số frame tối đa không detect được → xóa track
        min_hits:      Số hits tối thiểu để confirm track
        iou_thresh:    Ngưỡng IoU để match detection với track
        history_len:   Số frame lịch sử lưu cho voting
        smooth_alpha:  Hệ số smooth keypoints (0.7 = giữ nhiều kp cũ)
    """

    def __init__(
        self,
        max_age:     int   = 10,
        min_hits:    int   = 2,
        iou_thresh:  float = 0.35,
        history_len: int   = 5,
        smooth_alpha: float = 0.70,
    ):
        self.max_age     = max_age
        self.min_hits    = min_hits
        self.iou_thresh  = iou_thresh
        self.history_len = history_len
        self.smooth_alpha = smooth_alpha

        self._tracks:   Dict[int, TrackState] = {}
        self._next_id:  int = 1
        self._frame_id: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        bboxes:    List[List[int]],
        scores:    Optional[List[float]]       = None,
        keypoints: Optional[List[Optional[np.ndarray]]] = None,
        raincoats: Optional[List[Optional[bool]]]       = None,
        poses:     Optional[List[Optional[str]]]        = None,
        depths:    Optional[List[Optional[float]]]      = None,
        heights:   Optional[List[Optional[float]]]      = None,  # [IMPROVE] Height smoothing
    ) -> List[TrackResult]:
        """
        Cập nhật tracker với detections frame hiện tại.

        Args:
            bboxes:    [[x1,y1,x2,y2], ...] N detections
            scores:    [conf, ...] nếu None → 1.0
            keypoints: [(17,3) array or None, ...] per detection
            raincoats: [True/False/None, ...] per detection
            poses:     ["STANDING"/"WADING"/..., ...] per detection
            depths:    [depth_cm, ...] per detection
            heights:   [height_cm, ...] per detection (optional, for EMA smoothing)

        Returns:
            List[TrackResult] — chỉ các track đủ min_hits (confirmed)
        """
        self._frame_id += 1
        N = len(bboxes)

        if scores    is None: scores    = [1.0] * N
        if keypoints is None: keypoints = cast(List[Optional[np.ndarray]], [None] * N)
        if raincoats is None: raincoats = cast(List[Optional[bool]],       [None] * N)
        if poses     is None: poses     = cast(List[Optional[str]],        [None] * N)
        if depths    is None: depths    = cast(List[Optional[float]],      [None] * N)
        if heights   is None: heights   = cast(List[Optional[float]],      [None] * N)

        # Step 1: predict (tăng age, miss)
        for t in self._tracks.values():
            t.miss += 1

        # Step 2: match detections → existing tracks
        matched, unmatched_dets = self._match(bboxes, scores)

        # Step 3: update matched tracks
        for det_idx, trk_id in matched:
            t = self._tracks[trk_id]
            t.bbox   = bboxes[det_idx]
            t.score  = scores[det_idx]
            t.hits  += 1
            t.miss   = 0
            t.age   += 1

            # Smooth keypoints
            kp = keypoints[det_idx]
            t.keypoints = self._smooth_keypoints(t.keypoints, kp)
            t.smoothed_keypoints = t.keypoints

            # Update history
            rc = raincoats[det_idx]
            if rc is not None:
                t.raincoat_history.append(rc)
                if len(t.raincoat_history) > self.history_len:
                    t.raincoat_history.pop(0)

            ps = poses[det_idx]
            if ps is not None:
                t.pose_history.append(ps)
                if len(t.pose_history) > self.history_len:
                    t.pose_history.pop(0)

            dp = depths[det_idx]
            if dp is not None:
                t.depth_history.append(dp)
                if len(t.depth_history) > self.history_len:
                    t.depth_history.pop(0)
                # [IMPROVE] EMA smoothing cho depth:
                alpha = 0.3
                if t.ema_depth is None:
                    t.ema_depth = dp
                else:
                    t.ema_depth = alpha * dp + (1.0 - alpha) * t.ema_depth

            # [IMPROVE] EMA smoothing cho height:
            hp = heights[det_idx]
            if hp is not None:
                t.height_history.append(hp)
                if len(t.height_history) > self.history_len:
                    t.height_history.pop(0)
                alpha_h = 0.25  # slightly more conservative for height
                if t.ema_height is None:
                    t.ema_height = hp
                else:
                    t.ema_height = alpha_h * hp + (1.0 - alpha_h) * t.ema_height

            # Compute votes
            t.raincoat_vote, t.raincoat_conf = self._majority_vote_bool(t.raincoat_history)

        # Step 4: create new tracks for unmatched detections
        new_ids = set()
        for det_idx in unmatched_dets:
            new_id = self._next_id
            self._next_id += 1
            kp = keypoints[det_idx]
            rc = raincoats[det_idx]
            ps = poses[det_idx]
            dp = depths[det_idx]

            hp = heights[det_idx]
            self._tracks[new_id] = TrackState(
                track_id=new_id,
                bbox=bboxes[det_idx],
                score=scores[det_idx],
                age=1, hits=1, miss=0,
                keypoints=kp,
                smoothed_keypoints=kp,
                raincoat_history=[rc] if rc is not None else [],
                pose_history=[ps] if ps is not None else [],
                depth_history=[dp] if dp is not None else [],
                height_history=[hp] if hp is not None else [],
                raincoat_vote=rc,
                raincoat_conf=float(rc) if rc is not None else 0.0,
            )
            new_ids.add(new_id)

        # Step 5: remove dead tracks
        dead_ids = [tid for tid, t in self._tracks.items() if t.miss > self.max_age]
        for tid in dead_ids:
            del self._tracks[tid]

        # Step 6: collect confirmed outputs
        results = []
        for tid, t in self._tracks.items():
            if t.hits >= self.min_hits or tid in new_ids:
                pose_vote = self._majority_vote_str(t.pose_history)
                depth_med = float(np.median(t.depth_history)) if t.depth_history else None
                height_med = float(np.median(t.height_history)) if t.height_history else None
                results.append(TrackResult(
                    track_id=t.track_id,
                    bbox=t.bbox,
                    score=t.score,
                    smoothed_keypoints=t.smoothed_keypoints,
                    raincoat_vote=t.raincoat_vote,
                    raincoat_conf=t.raincoat_conf,
                    pose_vote=pose_vote,
                    depth_median=depth_med,
                    depth_ema=t.ema_depth,
                    height_ema=t.ema_height,  # [IMPROVE] EMA smoothed height
                    is_new_track=(tid in new_ids),
                ))

        log.debug(
            f"  Tracker frame={self._frame_id}: "
            f"{len(results)} active, {len(dead_ids)} removed, "
            f"{len(unmatched_dets)} new"
        )
        return results

    def get_track(self, track_id: int) -> Optional[TrackState]:
        return self._tracks.get(track_id)

    def reset(self):
        self._tracks   = {}
        self._next_id  = 1
        self._frame_id = 0

    # ── Matching (IoU Hungarian) ──────────────────────────────────────────────

    def _match(
        self,
        bboxes: List[List[int]],
        scores: List[float],
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        Greedy IoU matching giữa detections và existing tracks.

        Returns:
            matched:       [(det_idx, trk_id), ...]
            unmatched_dets: [det_idx, ...]
        """
        if not self._tracks or not bboxes:
            return [], list(range(len(bboxes)))

        trk_ids   = list(self._tracks.keys())
        trk_bboxes = [self._tracks[tid].bbox for tid in trk_ids]

        # IoU matrix (N_det × N_trk)
        iou_matrix = np.zeros((len(bboxes), len(trk_ids)), dtype=np.float32)
        for di, db in enumerate(bboxes):
            for ti, tb in enumerate(trk_bboxes):
                iou_matrix[di, ti] = self._iou(db, tb)

        matched       = []
        used_dets     = set()
        used_trks     = set()

        # Greedy: lấy cặp IoU cao nhất trước
        flat_indices  = np.argsort(-iou_matrix.ravel())
        for idx in flat_indices:
            di = idx // len(trk_ids)
            ti = idx  % len(trk_ids)
            if di in used_dets or ti in used_trks:
                continue
            if iou_matrix[di, ti] < self.iou_thresh:
                break
            matched.append((di, trk_ids[ti]))
            used_dets.add(di)
            used_trks.add(ti)

        unmatched_dets = [i for i in range(len(bboxes)) if i not in used_dets]
        return matched, unmatched_dets

    @staticmethod
    def _iou(a: List[int], b: List[int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
        inter = iw * ih
        if inter == 0:
            return 0.0
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return inter / (area_a + area_b - inter + 1e-6)

    # ── Smoothing ─────────────────────────────────────────────────────────────

    def _smooth_keypoints(
        self,
        prev: Optional[np.ndarray],
        curr: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """
        kp_t = alpha * kp_t-1 + (1-alpha) * kp_t

        Chỉ smooth x, y — giữ confidence của frame hiện tại.
        """
        if curr is None:
            return prev
        if prev is None:
            return curr.copy()

        smoothed = curr.copy()
        # Chỉ smooth các keypoints có confidence hợp lệ ở cả 2 frame
        for k in range(17):
            if prev[k, 2] > 0.3 and curr[k, 2] > 0.3:
                smoothed[k, 0] = self.smooth_alpha * prev[k, 0] + (1 - self.smooth_alpha) * curr[k, 0]
                smoothed[k, 1] = self.smooth_alpha * prev[k, 1] + (1 - self.smooth_alpha) * curr[k, 1]
        return smoothed

    # ── Voting ────────────────────────────────────────────────────────────────

    @staticmethod
    def _majority_vote_bool(history: List[bool]) -> Tuple[Optional[bool], float]:
        if not history:
            return None, 0.0
        pos = sum(1 for x in history if x)
        total = len(history)
        vote = pos > total / 2
        conf = pos / total if vote else (total - pos) / total
        return vote, round(conf, 3)

    @staticmethod
    def _majority_vote_str(history: List[str]) -> Optional[str]:
        if not history:
            return None
        from collections import Counter
        return Counter(history).most_common(1)[0][0]
