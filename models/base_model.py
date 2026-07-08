from __future__ import annotations

import gc
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ModelStatus:
    OFFLINE = "OFFLINE"
    LOADING = "LOADING"
    READY = "READY"
    ACTIVE = "ACTIVE"
    ALERT = "ALERT"
    AUTHORIZED = "AUTHORIZED"


class BaseModel(ABC):

    ALERT_HOLD_SECONDS = 3.0

    def __init__(self) -> None:
        self._loaded = False
        self._load_time_ms: float = 0.0
        self._event_bus: Optional[Any] = None
        self._status: str = ModelStatus.OFFLINE
        self._alert_until: float = 0.0

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def load_time_ms(self) -> float:
        return self._load_time_ms

    @property
    def status(self) -> str:
        if self._status == ModelStatus.ALERT and time.perf_counter() > self._alert_until:
            self._status = ModelStatus.ACTIVE if self._loaded else ModelStatus.OFFLINE
        return self._status

    def set_event_bus(self, event_bus: Any) -> None:
        """Optional: when set, load()/unload() publish ModelLoaded/ModelUnloaded events."""
        self._event_bus = event_bus

    def load(self) -> None:
        if self._loaded:
            return
        self._status = ModelStatus.LOADING
        t0 = time.perf_counter()
        self._do_load()
        self._load_time_ms = (time.perf_counter() - t0) * 1000.0
        self._loaded = True
        self._status = ModelStatus.READY
        logger.info(
            "%s loaded in %.1f ms", self.__class__.__name__, self._load_time_ms,
        )
        self._publish_lifecycle("loaded")

    def predict(self, frame: Any) -> Dict[str, Any]:
        if not self._loaded:
            raise RuntimeError(
                f"{self.__class__.__name__}.predict() called before load()"
            )
        self._status = ModelStatus.ACTIVE
        result = self._do_predict(frame)
        if result.get("detected") or result.get("is_authorized"):
            self._status = ModelStatus.ALERT
            self._alert_until = time.perf_counter() + self.ALERT_HOLD_SECONDS
        return result

    def unload(self) -> None:
        if not self._loaded:
            return
        self._do_unload()
        self._loaded = False
        self._status = ModelStatus.OFFLINE
        gc.collect()
        logger.info("%s unloaded", self.__class__.__name__)
        self._publish_lifecycle("unloaded")

    def _publish_lifecycle(self, kind: str) -> None:
        if self._event_bus is None:
            return
        try:
            from core.events import ModelLoaded, ModelUnloaded
            if kind == "loaded":
                self._event_bus.publish(ModelLoaded(
                    model_name=self.__class__.__name__,
                    load_time_ms=self._load_time_ms,
                ))
            else:
                self._event_bus.publish(ModelUnloaded(
                    model_name=self.__class__.__name__,
                ))
        except Exception:
            logger.exception("Failed to publish %s lifecycle event", kind)

    @abstractmethod
    def _do_load(self) -> None:
        ...

    @abstractmethod
    def _do_predict(self, frame: Any) -> Dict[str, Any]:
        ...

    @abstractmethod
    def _do_unload(self) -> None:
        ...
