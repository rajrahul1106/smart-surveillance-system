from __future__ import annotations

import logging
from typing import Any, Dict

from core.event_bus import EventBus, EventType
from core.events import ProcessedFrame
from core.state_machine import State, StateMachine
from core.config import PipelineConfig
from core.states import (
    ActiveDetectionState,
    CooldownState,
    IdleState,
    VerifyingGestureState,
    VerifyingIdentityState,
)

logger = logging.getLogger(__name__)


class Pipeline:

    def __init__(
        self,
        event_bus: EventBus,
        state_machine: StateMachine,
        config: PipelineConfig,
        gesture_model: Any,
        face_model: Any,
        fire_model: Any,
        injury_model: Any,
        activity_model: Any,
        camera_manager: Any = None,
        shared_frame: Any = None,
        frame_manager: Any = None,
    ) -> None:
        self._event_bus = event_bus
        self._state_machine = state_machine
        self._config = config
        self._shared_frame = shared_frame

        self._context: Dict[str, Any] = {
            "event_bus": event_bus,
            "config": config,
            "gesture_model": gesture_model,
            "face_model": face_model,
            "fire_model": fire_model,
            "injury_model": injury_model,
            "activity_model": activity_model,
            "camera_manager": camera_manager,
            "frame_manager": frame_manager,
            "trigger_callback": self._state_machine.transition,
        }

        self._state_registry = {
            State.IDLE: IdleState(self._context),
            State.VERIFYING_GESTURE: VerifyingGestureState(self._context),
            State.VERIFYING_IDENTITY: VerifyingIdentityState(self._context),
            State.ACTIVE_DETECTION: ActiveDetectionState(self._context),
            State.COOLDOWN: CooldownState(self._context),
        }

        self._state_machine.register_on_transition(self._handle_transition)
        self._event_bus.subscribe(EventType.PROCESSED_FRAME, self._on_processed_frame)

        self._state_registry[self._state_machine.state].on_enter(self._context)

    def _handle_transition(self, from_state: State, to_state: State) -> None:
        self._context["next_state"] = to_state
        self._state_registry[from_state].on_exit(self._context)
        self._state_registry[to_state].on_enter(self._context)
        self._context.pop("next_state", None)

    def _on_processed_frame(self, event: ProcessedFrame) -> None:
        frame = event.frame
        original = event.original_frame if event.original_frame is not None else frame
        current = self._state_registry[self._state_machine.state]
        trigger = current.on_frame(frame, self._context)

        # Publish the full (frame, state, detections) bundle BEFORE running
        # any state transition.  Transitions can mutate ``last_detections``
        # in their on_exit/on_enter handlers (e.g. clearing sos_progress),
        # which would otherwise leak into this tick's snapshot if we
        # published after.  SharedFrame.set() shallow-copies the dict so
        # the snapshot is immutable from the consumer's perspective.
        if self._shared_frame is not None:
            detections = self._context.get("last_detections", {}) or {}
            self._shared_frame.set(
                original,
                self._state_machine.state.name,
                detections,
            )

        if trigger is not None:
            self._state_machine.transition(trigger)
