from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

import cv2
import numpy as np

from core.config import FrameManagerConfig
from core.event_bus import EventBus, EventType
from core.events import FrameReady, ProcessedFrame

logger = logging.getLogger(__name__)


class FrameManager:

    def __init__(self, event_bus: EventBus, config: Optional[FrameManagerConfig] = None) -> None:
        self._event_bus = event_bus
        self._config = config or FrameManagerConfig()
        self._skip_rate: int = self._config.frame_skip
        self._scale_width: int = self._config.scale_width
        self._scale_height: int = self._config.scale_height

        self._frames_received: int = 0
        self._frames_processed: int = 0
        self._frames_skipped: int = 0
        self._lock = threading.Lock()
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._gamma_table: Optional[np.ndarray] = None

        self._event_bus.subscribe(EventType.FRAME_READY, self._on_frame_ready)

    def _on_frame_ready(self, event: FrameReady) -> None:
        with self._lock:
            count = self._frames_received
            self._frames_received += 1
            skip_rate = self._skip_rate
            sw, sh = self._scale_width, self._scale_height

        if count % skip_rate != 0:
            with self._lock:
                self._frames_skipped += 1
            return

        original = event.frame
        corrected = self._preprocess(original)
        if sw > 0 and sh > 0:
            scaled = cv2.resize(corrected, (sw, sh))
        else:
            scaled = corrected

        with self._lock:
            self._frames_processed += 1

        self._event_bus.publish(ProcessedFrame(
            frame=scaled,
            camera_id=event.camera_id,
            resolution=(sw, sh) if (sw > 0 and sh > 0) else (),
            original_frame=corrected,
        ))

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """CLAHE + conditional gamma correction for overexposed frames."""
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            return frame
        try:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            l_clahe = self._clahe.apply(l_ch)

            mean_brightness = float(np.mean(l_ch))
            if mean_brightness > 180:
                if self._gamma_table is None:
                    inv_gamma = 1.0 / 1.4
                    self._gamma_table = np.array(
                        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
                        dtype=np.uint8,
                    )
                lab_merged = cv2.merge([l_clahe, a_ch, b_ch])
                corrected = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)
                return cv2.LUT(corrected, self._gamma_table)

            lab_merged = cv2.merge([l_clahe, a_ch, b_ch])
            return cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)
        except Exception:
            return frame

    # ------------------------------------------------------------------
    # Runtime adjustment
    # ------------------------------------------------------------------

    def set_skip_rate(self, n: int) -> None:
        with self._lock:
            self._skip_rate = max(1, n)

    def set_resolution(self, width: int, height: int) -> None:
        with self._lock:
            self._scale_width = width
            self._scale_height = height

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "frames_received": self._frames_received,
                "frames_processed": self._frames_processed,
                "frames_skipped": self._frames_skipped,
                "current_skip_rate": self._skip_rate,
            }
