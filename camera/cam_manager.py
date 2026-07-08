from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import replace
from typing import Any, Deque, Dict, List, Optional

import cv2

from camera.cam_config import CameraConfig
from core.event_bus import EventBus
from core.events import CameraConnected, CameraDisconnected, FrameReady

logger = logging.getLogger(__name__)


class CameraManager:

    def __init__(self, event_bus: EventBus, config: Optional[CameraConfig] = None) -> None:
        self._event_bus = event_bus
        self._config = config or CameraConfig()
        self._cap: Optional[Any] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_connected = False
        self._total_frame_count = 0
        self._frame_timestamps: Deque[float] = deque(maxlen=60)

    @property
    def is_running(self) -> bool:
        return self._capture_thread is not None and self._capture_thread.is_alive()

    def enumerate_cameras(self) -> List[int]:
        available: List[int] = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
            cap.release()
        return available

    def select_camera(self, index: int) -> None:
        was_running = self.is_running
        if was_running:
            self.stop()
        self._config = replace(self._config, index=index)
        if was_running:
            self.start()

    def start(self) -> None:
        self._stop_event.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="CameraManager-capture",
        )
        self._capture_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=5.0)
            self._capture_thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_connected = False

    def get_status(self) -> Dict[str, Any]:
        timestamps = list(self._frame_timestamps)
        if len(timestamps) >= 2:
            elapsed = timestamps[-1] - timestamps[0]
            measured_fps = (len(timestamps) - 1) / elapsed if elapsed > 0 else 0.0
        else:
            measured_fps = 0.0
        return {
            "camera_index": self._config.index,
            "is_connected": self._is_connected,
            "measured_fps": round(measured_fps, 2),
            "total_frame_count": self._total_frame_count,
        }

    # ------------------------------------------------------------------
    # Internal capture loop
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        cfg = self._config
        reconnect_count = 0

        while not self._stop_event.is_set():
            cap = cv2.VideoCapture(cfg.index)
            if not cap.isOpened():
                cap.release()
                reconnect_count += 1
                if reconnect_count > cfg.max_reconnects:
                    self._event_bus.publish(CameraDisconnected(
                        camera_id=cfg.name, reason="failed_to_open",
                    ))
                    return
                self._stop_event.wait(cfg.reconnect_delay)
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.resolution_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.resolution_height)
            cap.set(cv2.CAP_PROP_FPS, cfg.fps)

            self._cap = cap
            self._is_connected = True

            self._event_bus.publish(CameraConnected(
                camera_id=cfg.name,
                index=cfg.index,
                resolution=(cfg.resolution_width, cfg.resolution_height),
            ))

            fail_count = 0
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if ret:
                    fail_count = 0
                    reconnect_count = 0
                    self._total_frame_count += 1
                    self._frame_timestamps.append(time.monotonic())
                    # Mirror at the source so detection AND display both see
                    # the same orientation. Without this, hand landmarks and
                    # bounding boxes drift relative to what the user sees in
                    # the dashboard's selfie-view feed.
                    if getattr(cfg, "mirror_horizontally", False):
                        frame = cv2.flip(frame, 1)
                    self._event_bus.publish(FrameReady(
                        frame=frame, camera_id=cfg.name,
                    ))
                else:
                    fail_count += 1
                    if fail_count >= cfg.max_consecutive_failures:
                        logger.error(
                            "Camera %s: %d consecutive read failures, reconnecting",
                            cfg.name, fail_count,
                        )
                        cap.release()
                        self._cap = None
                        self._is_connected = False
                        reconnect_count += 1
                        if reconnect_count > cfg.max_reconnects:
                            self._event_bus.publish(CameraDisconnected(
                                camera_id=cfg.name,
                                reason="max_reconnects_exceeded",
                            ))
                            return
                        self._stop_event.wait(cfg.reconnect_delay)
                        break
            else:
                # stop_event was set inside the inner loop
                cap.release()
                self._cap = None
                self._is_connected = False
                return
