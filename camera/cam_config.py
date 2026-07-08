from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CameraConfig:
    index: int = 0
    resolution_width: int = 640
    resolution_height: int = 480
    fps: int = 30
    name: str = "default"
    max_consecutive_failures: int = 5
    max_reconnects: int = 3
    reconnect_delay: float = 2.0
    # Mac built-in webcams output frames in their native (un-mirrored)
    # orientation, but users expect a "selfie" view in the dashboard.
    # Mirroring here makes the entire pipeline (detection + display) see
    # the same frame, so hand landmarks and bounding boxes stay aligned
    # with what the user actually sees.
    mirror_horizontally: bool = True
