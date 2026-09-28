"""Tests for the IoU face tracker: ids, expiry, status rules, and presence."""

from __future__ import annotations

import os
import sys

import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.face_tracker import (
    AUTHORIZED,
    UNCERTAIN,
    UNKNOWN,
    FaceDetection,
    FaceTracker,
    iou,
    summarize_presence,
)


def _det(bbox=(100, 100, 80, 80), **similarities):
    return FaceDetection(bbox=bbox, similarities=similarities)


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)


# ---------------------------------------------------------------------------
# Track association
# ---------------------------------------------------------------------------

class TestTracking:
    def test_keeps_track_id_for_face_that_moves_slightly(self):
        tracker = FaceTracker()
        first = tracker.update([_det((100, 100, 80, 80), Rahul=0.9)])[0]
        moved = tracker.update([_det((106, 104, 84, 80), Rahul=0.9)])[0]
        assert moved.track_id == first.track_id
        assert moved.bbox == (106, 104, 84, 80)
        assert len(tracker.tracks) == 1

    def test_distant_face_starts_a_new_track(self):
        tracker = FaceTracker()
        a = tracker.update([_det((0, 0, 50, 50))])[0]
        b = tracker.update([_det((300, 300, 50, 50))])[0]
        assert b.track_id != a.track_id

    def test_two_faces_keep_their_ids_regardless_of_order(self):
        tracker = FaceTracker()
        left, right = tracker.update([_det((10, 10, 60, 60)), _det((300, 10, 60, 60))])
        right2, left2 = tracker.update([_det((302, 12, 60, 60)), _det((12, 10, 60, 60))])
        assert (left2.track_id, right2.track_id) == (left.track_id, right.track_id)

    def test_removes_track_after_track_max_missed(self):
        tracker = FaceTracker(track_max_missed=3)
        tracker.update([_det()])
        for _ in range(3):
            tracker.update([])
        assert len(tracker.tracks) == 1
        assert tracker.tracks[0].missed == 3
        tracker.update([])  # 4th miss: missed > track_max_missed
        assert tracker.tracks == []

    def test_face_back_within_grace_keeps_track_and_identity(self):
        tracker = FaceTracker(track_max_missed=3)
        first = tracker.update([_det(Rahul=0.9)])[0]
        tracker.update([])
        back = tracker.update([_det(Rahul=0.9)])[0]
        assert back.track_id == first.track_id
        assert back.status == AUTHORIZED

    def test_reset_clears_tracks_but_ids_keep_increasing(self):
        tracker = FaceTracker()
        a = tracker.update([_det()])[0]
        tracker.reset()
        assert tracker.tracks == []
        b = tracker.update([_det()])[0]
        assert b.track_id > a.track_id


# ---------------------------------------------------------------------------
# Status rules
# ---------------------------------------------------------------------------

class TestStatus:
    def test_enrolled_face_is_authorized_with_its_name(self):
        track = FaceTracker().update([_det(Rahul=0.8)])[0]
        assert track.status == AUTHORIZED
        assert track.identity == "Rahul"

    def test_middle_similarity_is_uncertain(self):
        track = FaceTracker().update([_det(Rahul=0.45)])[0]
        assert track.status == UNCERTAIN
        assert track.identity is None

    def test_stranger_uncertain_until_unknown_confirmed(self):
        tracker = FaceTracker(unknown_confirm_frames=5)
        statuses = [tracker.update([_det(Rahul=0.05)])[0].status for _ in range(5)]
        assert statuses == [UNCERTAIN] * 4 + [UNKNOWN]
        assert tracker.tracks[0].identity is None

    def test_nobody_enrolled_means_unknown(self):
        tracker = FaceTracker(unknown_confirm_frames=2)
        statuses = [tracker.update([_det()])[0].status for _ in range(2)]
        assert statuses == [UNCERTAIN, UNKNOWN]

    def test_one_low_frame_does_not_make_authorized_face_unknown(self):
        tracker = FaceTracker()
        for _ in range(5):
            track = tracker.update([_det(Rahul=0.75)])[0]
        assert track.status == AUTHORIZED

        # Head turned away for one frame: ema 0.3*0 + 0.7*0.75 = 0.525.
        track = tracker.update([_det(Rahul=0.0)])[0]
        assert track.status == UNCERTAIN
        assert track.consecutive_unknown == 0

        for _ in range(2):
            track = tracker.update([_det(Rahul=0.75)])[0]
        assert track.status == AUTHORIZED
        assert track.identity == "Rahul"

    def test_unknown_only_after_confirm_frames_below_threshold(self):
        tracker = FaceTracker(unknown_confirm_frames=5)
        for _ in range(5):
            tracker.update([_det(Rahul=0.75)])
        # ema: .525 .368 | .257 .180 .126 .088 .062  (last five are < 0.30)
        statuses = [tracker.update([_det(Rahul=0.0)])[0].status for _ in range(7)]
        assert UNKNOWN not in statuses[:6]
        assert statuses[6] == UNKNOWN

    def test_unknown_counter_resets_when_similarity_recovers(self):
        tracker = FaceTracker(unknown_confirm_frames=3)
        tracker.update([_det(Rahul=0.1)])
        tracker.update([_det(Rahul=0.1)])
        track = tracker.update([_det(Rahul=0.9)])[0]  # ema 0.34: back above 0.30
        assert track.consecutive_unknown == 0
        track = tracker.update([_det(Rahul=0.1)])[0]  # ema 0.268: count restarts at 1
        assert track.consecutive_unknown == 1
        assert track.status == UNCERTAIN

    def test_identity_does_not_jump_on_a_single_frame(self):
        tracker = FaceTracker()
        for _ in range(5):
            tracker.update([_det(Rahul=0.8, Meghna=0.2)])
        # One ambiguous frame where Meghna scores higher than Rahul.
        track = tracker.update([_det(Rahul=0.3, Meghna=0.7)])[0]
        assert track.status == AUTHORIZED
        assert track.identity == "Rahul"

    def test_auth_streak_counts_consecutive_authorized_updates(self):
        tracker = FaceTracker()
        streaks = [tracker.update([_det(Rahul=0.9)])[0].auth_streak for _ in range(3)]
        assert streaks == [1, 2, 3]
        tracker.update([])  # a missed frame breaks the streak
        assert tracker.tracks[0].auth_streak == 0
        assert tracker.update([_det(Rahul=0.9)])[0].auth_streak == 1


# ---------------------------------------------------------------------------
# Presence summary
# ---------------------------------------------------------------------------

class TestPresenceSummary:
    def test_zero_faces(self):
        assert summarize_presence([]) == {
            "authorized": [], "unknown_count": 0, "uncertain_count": 0, "total": 0,
        }

    def test_one_face(self):
        faces = [{"status": AUTHORIZED, "identity": "Rahul Raj"}]
        assert summarize_presence(faces) == {
            "authorized": ["Rahul Raj"], "unknown_count": 0, "uncertain_count": 0, "total": 1,
        }

    def test_three_faces(self):
        faces = [
            {"status": AUTHORIZED, "identity": "Meghna Lal"},
            {"status": UNKNOWN, "identity": None},
            {"status": UNCERTAIN, "identity": None},
        ]
        assert summarize_presence(faces) == {
            "authorized": ["Meghna Lal"], "unknown_count": 1, "uncertain_count": 1, "total": 3,
        }

    def test_authorized_names_are_unique_and_sorted(self):
        faces = [
            {"status": AUTHORIZED, "identity": "Sarthak"},
            {"status": AUTHORIZED, "identity": "Meghna"},
            {"status": AUTHORIZED, "identity": "Sarthak"},
        ]
        summary = summarize_presence(faces)
        assert summary["authorized"] == ["Meghna", "Sarthak"]
        assert summary["total"] == 3
