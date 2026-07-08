"""Tests for CameraManager, CameraConfig, and FrameManager."""

from __future__ import annotations

import sys
import os
import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from camera.cam_config import CameraConfig
from core.event_bus import EventBus, EventType
from core.config import FrameManagerConfig
from core.events import CameraConnected, CameraDisconnected, FrameReady, ProcessedFrame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wait_for_thread(manager, timeout: float = 2.0) -> None:
    """Wait for the capture thread to finish naturally."""
    if manager._capture_thread is not None:
        manager._capture_thread.join(timeout=timeout)


def _make_mock_cap(is_opened: bool = True, read_returns=None):
    """Build a mock cv2.VideoCapture instance."""
    cap = MagicMock()
    cap.isOpened.return_value = is_opened
    if read_returns is not None:
        cap.read.side_effect = read_returns
    else:
        cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    return cap


# ---------------------------------------------------------------------------
# CameraConfig
# ---------------------------------------------------------------------------

class TestCameraConfig:
    def test_defaults(self):
        cfg = CameraConfig()
        assert cfg.index == 0
        assert cfg.resolution_width == 640
        assert cfg.resolution_height == 480
        assert cfg.fps == 30
        assert cfg.name == "default"

    def test_custom_values(self):
        cfg = CameraConfig(index=2, resolution_width=1920, resolution_height=1080, fps=60, name="hd_cam")
        assert cfg.index == 2
        assert cfg.resolution_width == 1920
        assert cfg.name == "hd_cam"


# ---------------------------------------------------------------------------
# CameraManager.enumerate_cameras
# ---------------------------------------------------------------------------

class TestEnumerateCameras:
    @patch("camera.cam_manager.cv2")
    def test_finds_available_cameras(self, mock_cv2):
        from camera.cam_manager import CameraManager

        caps = {}
        for i in range(5):
            m = MagicMock()
            m.isOpened.return_value = i in (0, 2)
            caps[i] = m

        mock_cv2.VideoCapture.side_effect = lambda idx: caps[idx]

        bus = EventBus()
        mgr = CameraManager(bus)
        result = mgr.enumerate_cameras()

        assert result == [0, 2]
        for m in caps.values():
            m.release.assert_called_once()


# ---------------------------------------------------------------------------
# CameraManager start / stop lifecycle
# ---------------------------------------------------------------------------

class TestStartStop:
    @patch("camera.cam_manager.cv2")
    def test_start_publishes_connected_and_frames(self, mock_cv2):
        from camera.cam_manager import CameraManager

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        read_count = 0
        stop_after = 5

        def _read():
            nonlocal read_count
            read_count += 1
            if read_count > stop_after:
                return (False, None)
            return (True, frame.copy())

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.side_effect = _read
        mock_cv2.VideoCapture.return_value = cap
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_FPS = 5

        bus = EventBus()
        cfg = CameraConfig(max_consecutive_failures=100, reconnect_delay=0)
        mgr = CameraManager(bus, cfg)

        connected = []
        frames = []
        bus.subscribe(EventType.CAMERA_CONNECTED, connected.append)
        bus.subscribe(EventType.FRAME_READY, frames.append)

        mgr.start()
        assert mgr.is_running

        time.sleep(0.3)
        mgr.stop()
        bus.flush()

        assert not mgr.is_running
        assert len(connected) >= 1
        assert isinstance(connected[0], CameraConnected)
        assert len(frames) == stop_after

    @patch("camera.cam_manager.cv2")
    def test_stop_releases_camera(self, mock_cv2):
        from camera.cam_manager import CameraManager

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cv2.VideoCapture.return_value = cap
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_FPS = 5

        bus = EventBus()
        mgr = CameraManager(bus, CameraConfig(reconnect_delay=0))
        mgr.start()
        time.sleep(0.05)
        mgr.stop()

        assert not mgr.is_running
        assert mgr._is_connected is False

    @patch("camera.cam_manager.cv2")
    def test_get_status(self, mock_cv2):
        from camera.cam_manager import CameraManager

        call_count = 0

        def _read():
            nonlocal call_count
            call_count += 1
            if call_count > 3:
                return (False, None)
            return (True, np.zeros((2, 2, 3), dtype=np.uint8))

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.side_effect = _read
        mock_cv2.VideoCapture.return_value = cap
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_FPS = 5

        bus = EventBus()
        mgr = CameraManager(bus, CameraConfig(
            max_consecutive_failures=100, reconnect_delay=0,
        ))
        mgr.start()
        time.sleep(0.3)
        mgr.stop()

        status = mgr.get_status()
        assert status["camera_index"] == 0
        assert status["total_frame_count"] == 3


# ---------------------------------------------------------------------------
# Disconnect handling — 5 consecutive failures
# ---------------------------------------------------------------------------

class TestDisconnect:
    @patch("camera.cam_manager.cv2")
    def test_disconnect_after_consecutive_failures(self, mock_cv2):
        from camera.cam_manager import CameraManager

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.return_value = (False, None)
        mock_cv2.VideoCapture.return_value = cap
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_FPS = 5

        bus = EventBus()
        disconnected = []
        bus.subscribe(EventType.CAMERA_DISCONNECTED, disconnected.append)

        cfg = CameraConfig(
            max_consecutive_failures=5,
            max_reconnects=0,
            reconnect_delay=0,
        )
        mgr = CameraManager(bus, cfg)
        mgr.start()
        _wait_for_thread(mgr)
        bus.flush()

        assert len(disconnected) == 1
        assert isinstance(disconnected[0], CameraDisconnected)


# ---------------------------------------------------------------------------
# Reconnection attempts
# ---------------------------------------------------------------------------

class TestReconnection:
    @patch("camera.cam_manager.cv2")
    def test_reconnects_up_to_max(self, mock_cv2):
        from camera.cam_manager import CameraManager

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.return_value = (False, None)
        mock_cv2.VideoCapture.return_value = cap
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_FPS = 5

        bus = EventBus()
        connected = []
        disconnected = []
        bus.subscribe(EventType.CAMERA_CONNECTED, connected.append)
        bus.subscribe(EventType.CAMERA_DISCONNECTED, disconnected.append)

        cfg = CameraConfig(
            max_consecutive_failures=2,
            max_reconnects=3,
            reconnect_delay=0,
        )
        mgr = CameraManager(bus, cfg)
        mgr.start()
        _wait_for_thread(mgr)
        bus.flush()

        # Initial connect + 3 reconnects = 4 connected events
        assert len(connected) == 4
        assert len(disconnected) == 1
        assert disconnected[0].reason == "max_reconnects_exceeded"

    @patch("camera.cam_manager.cv2")
    def test_reconnect_fails_to_open(self, mock_cv2):
        from camera.cam_manager import CameraManager

        cap = MagicMock()
        cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = cap

        bus = EventBus()
        disconnected = []
        bus.subscribe(EventType.CAMERA_DISCONNECTED, disconnected.append)

        cfg = CameraConfig(max_reconnects=2, reconnect_delay=0)
        mgr = CameraManager(bus, cfg)
        mgr.start()
        _wait_for_thread(mgr)
        bus.flush()

        assert len(disconnected) == 1
        assert disconnected[0].reason == "failed_to_open"

    @patch("camera.cam_manager.cv2")
    def test_select_camera_restarts_with_new_index(self, mock_cv2):
        from camera.cam_manager import CameraManager

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.return_value = (True, np.zeros((2, 2, 3), dtype=np.uint8))
        mock_cv2.VideoCapture.return_value = cap
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_FPS = 5

        bus = EventBus()
        mgr = CameraManager(bus, CameraConfig(reconnect_delay=0))
        mgr.start()
        time.sleep(0.05)

        mgr.select_camera(2)
        assert mgr._config.index == 2
        assert mgr.is_running

        mgr.stop()


# ---------------------------------------------------------------------------
# FrameManager — skip rate
# ---------------------------------------------------------------------------

class TestFrameManagerSkip:
    def test_skip_rate_1_processes_all(self):
        from core.frame_manager import FrameManager

        bus = EventBus()
        fm = FrameManager(bus, FrameManagerConfig(frame_skip=1, scale_width=0, scale_height=0))

        processed = []
        bus.subscribe(EventType.PROCESSED_FRAME, processed.append)

        for i in range(10):
            bus.publish(FrameReady(frame=np.zeros((4, 4, 3), dtype=np.uint8), camera_id="cam0"))
        bus.flush()

        assert len(processed) == 10
        stats = fm.get_stats()
        assert stats["frames_received"] == 10
        assert stats["frames_processed"] == 10
        assert stats["frames_skipped"] == 0

    def test_skip_rate_3_processes_expected(self):
        from core.frame_manager import FrameManager

        bus = EventBus()
        fm = FrameManager(bus, FrameManagerConfig(frame_skip=3, scale_width=0, scale_height=0))

        processed = []
        bus.subscribe(EventType.PROCESSED_FRAME, processed.append)

        for i in range(10):
            bus.publish(FrameReady(frame=np.zeros((4, 4, 3), dtype=np.uint8), camera_id="cam0"))
        bus.flush()

        # Frames 0, 3, 6, 9 → 4 processed
        assert len(processed) == 4
        stats = fm.get_stats()
        assert stats["frames_received"] == 10
        assert stats["frames_processed"] == 4
        assert stats["frames_skipped"] == 6
        assert stats["current_skip_rate"] == 3

    def test_set_skip_rate_at_runtime(self):
        from core.frame_manager import FrameManager

        bus = EventBus()
        fm = FrameManager(bus, FrameManagerConfig(frame_skip=1, scale_width=0, scale_height=0))

        processed = []
        bus.subscribe(EventType.PROCESSED_FRAME, processed.append)

        # First 5 frames with skip_rate=1
        for _ in range(5):
            bus.publish(FrameReady(frame=np.zeros((4, 4, 3), dtype=np.uint8)))
        bus.flush()
        assert len(processed) == 5

        # Change skip rate to 2
        fm.set_skip_rate(2)

        # Next 6 frames with skip_rate=2 — received count continues from 5
        # Frames 5,6,7,8,9,10 → indices 5%2=1(skip), 6%2=0(keep), 7%2=1(skip),
        #                         8%2=0(keep), 9%2=1(skip), 10%2=0(keep)
        for _ in range(6):
            bus.publish(FrameReady(frame=np.zeros((4, 4, 3), dtype=np.uint8)))
        bus.flush()

        assert len(processed) == 5 + 3  # 5 from before + 3 new


# ---------------------------------------------------------------------------
# FrameManager — resolution scaling
# ---------------------------------------------------------------------------

class TestFrameManagerResize:
    @patch("core.frame_manager.cv2")
    def test_downscales_to_configured_resolution(self, mock_cv2):
        from core.frame_manager import FrameManager

        small_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        mock_cv2.resize.return_value = small_frame

        bus = EventBus()
        fm = FrameManager(bus, FrameManagerConfig(scale_width=320, scale_height=240))

        processed = []
        bus.subscribe(EventType.PROCESSED_FRAME, processed.append)

        original = np.ones((480, 640, 3), dtype=np.uint8)
        bus.publish(FrameReady(frame=original))
        bus.flush()

        mock_cv2.resize.assert_called_once()
        call_args = mock_cv2.resize.call_args
        np.testing.assert_array_equal(call_args[0][0], original)
        assert call_args[0][1] == (320, 240)

        assert len(processed) == 1
        evt = processed[0]
        assert isinstance(evt, ProcessedFrame)
        np.testing.assert_array_equal(evt.frame, small_frame)
        np.testing.assert_array_equal(evt.original_frame, original)
        assert evt.resolution == (320, 240)

    @patch("core.frame_manager.cv2")
    def test_no_scaling_when_zero(self, mock_cv2):
        from core.frame_manager import FrameManager

        bus = EventBus()
        fm = FrameManager(bus, FrameManagerConfig(scale_width=0, scale_height=0))

        processed = []
        bus.subscribe(EventType.PROCESSED_FRAME, processed.append)

        original = np.ones((480, 640, 3), dtype=np.uint8)
        bus.publish(FrameReady(frame=original))
        bus.flush()

        mock_cv2.resize.assert_not_called()
        np.testing.assert_array_equal(processed[0].frame, original)

    @patch("core.frame_manager.cv2")
    def test_set_resolution_at_runtime(self, mock_cv2):
        from core.frame_manager import FrameManager

        mock_cv2.resize.return_value = np.zeros((120, 160, 3), dtype=np.uint8)

        bus = EventBus()
        fm = FrameManager(bus, FrameManagerConfig(scale_width=320, scale_height=240))
        fm.set_resolution(160, 120)

        processed = []
        bus.subscribe(EventType.PROCESSED_FRAME, processed.append)

        bus.publish(FrameReady(frame=np.ones((480, 640, 3), dtype=np.uint8)))
        bus.flush()

        call_args = mock_cv2.resize.call_args
        assert call_args[0][1] == (160, 120)
        assert processed[0].resolution == (160, 120)


# ---------------------------------------------------------------------------
# End-to-end: CameraManager → FrameManager → ProcessedFrame
# ---------------------------------------------------------------------------

class TestCameraFrameManagerIntegration:
    @patch("core.frame_manager.cv2")
    @patch("camera.cam_manager.cv2")
    def test_full_pipeline_flow(self, mock_cam_cv2, mock_fm_cv2):
        from camera.cam_manager import CameraManager
        from core.frame_manager import FrameManager

        frame = np.ones((480, 640, 3), dtype=np.uint8)
        read_count = 0

        def _read():
            nonlocal read_count
            read_count += 1
            if read_count > 3:
                return (False, None)
            return (True, frame.copy())

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.side_effect = _read
        mock_cam_cv2.VideoCapture.return_value = cap
        mock_cam_cv2.CAP_PROP_FRAME_WIDTH = 3
        mock_cam_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cam_cv2.CAP_PROP_FPS = 5

        scaled = np.zeros((240, 320, 3), dtype=np.uint8)
        mock_fm_cv2.resize.return_value = scaled

        bus = EventBus()
        fm = FrameManager(bus, FrameManagerConfig(frame_skip=1, scale_width=320, scale_height=240))

        processed = []
        bus.subscribe(EventType.PROCESSED_FRAME, processed.append)

        cfg = CameraConfig(max_consecutive_failures=100, reconnect_delay=0)
        mgr = CameraManager(bus, cfg)
        mgr.start()
        time.sleep(0.3)
        mgr.stop()
        bus.flush()

        assert len(processed) == 3
        for evt in processed:
            assert isinstance(evt, ProcessedFrame)
            np.testing.assert_array_equal(evt.frame, scaled)
            assert evt.resolution == (320, 240)
            assert evt.original_frame is not None
