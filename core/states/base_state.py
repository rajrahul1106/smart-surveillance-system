from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class AbstractState(ABC):

    @property
    @abstractmethod
    def active_models(self) -> List[Any]:
        ...

    @property
    @abstractmethod
    def subscriptions(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def on_enter(self, context: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def on_exit(self, context: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def on_frame(self, frame: Any, context: Dict[str, Any]) -> Optional[str]:
        ...
