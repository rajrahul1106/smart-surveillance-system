"""
core/state_machine.py

Thread-safe state machine for the smart-surveillance pipeline.

Stages (in order):
  IDLE → VERIFYING_GESTURE → VERIFYING_IDENTITY → ACTIVE_DETECTION → COOLDOWN → IDLE

Auto-timers:
  ACTIVE_DETECTION → COOLDOWN  after ``active_detection_timeout`` seconds (default 30)
  COOLDOWN        → IDLE       after ``cooldown_duration`` seconds (default 5)

Usage example::

    sm = StateMachine(active_detection_timeout=30, cooldown_duration=5)
    sm.register_on_enter(State.ACTIVE_DETECTION, lambda f, t: start_detectors())
    sm.register_on_exit(State.ACTIVE_DETECTION, lambda f, t: stop_detectors())

    sm.transition("gesture_candidate")   # → VERIFYING_GESTURE
    sm.transition("gesture_confirmed")   # → VERIFYING_IDENTITY
    sm.transition("auth_success")        # → ACTIVE_DETECTION (30 s timer starts)
    sm.transition("manual_override")     # → IDLE from any state
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

class State(Enum):
    IDLE = auto()
    VERIFYING_GESTURE = auto()
    VERIFYING_IDENTITY = auto()
    ACTIVE_DETECTION = auto()
    COOLDOWN = auto()


# ---------------------------------------------------------------------------
# Transition table  (trigger → next_state, keyed by current state)
# "manual_override" is handled separately — it works from every state.
# ---------------------------------------------------------------------------

_TRANSITIONS: Dict[State, Dict[str, State]] = {
    State.IDLE: {
        "gesture_candidate": State.VERIFYING_GESTURE,
    },
    State.VERIFYING_GESTURE: {
        "gesture_confirmed": State.VERIFYING_IDENTITY,
        "gesture_rejected":  State.IDLE,
    },
    State.VERIFYING_IDENTITY: {
        "auth_success": State.ACTIVE_DETECTION,
        "auth_failed":  State.IDLE,
    },
    State.ACTIVE_DETECTION: {
        "detection_complete": State.COOLDOWN,
        "timeout":            State.COOLDOWN,
    },
    State.COOLDOWN: {
        "cooldown_expired": State.IDLE,
    },
}


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class StateMachine:
    """
    Finite-state machine that drives the surveillance pipeline.

    Parameters
    ----------
    active_detection_timeout:
        How long (seconds) ACTIVE_DETECTION may run before automatically
        transitioning to COOLDOWN via an internal "timeout" trigger.
    cooldown_duration:
        How long (seconds) to remain in COOLDOWN before automatically
        returning to IDLE via an internal "cooldown_expired" trigger.
    """

    def __init__(
        self,
        active_detection_timeout: float = 30.0,
        cooldown_duration: float = 5.0,
    ) -> None:
        self._state: State = State.IDLE
        self._lock = threading.Lock()
        self._active_detection_timeout = active_detection_timeout
        self._cooldown_duration = cooldown_duration

        # Per-state callback lists; each callback receives (from_state, to_state).
        self._on_enter: Dict[State, List[Callable[[State, State], None]]] = {
            s: [] for s in State
        }
        self._on_exit: Dict[State, List[Callable[[State, State], None]]] = {
            s: [] for s in State
        }

        # Global transition callbacks: each receives (from_state, to_state).
        self._on_transition: List[Callable[[State, State], None]] = []

        # Background timers for timed states (cancelled on any exit).
        self._active_timer: Optional[threading.Timer] = None
        self._cooldown_timer: Optional[threading.Timer] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> State:
        """Current state (thread-safe read)."""
        with self._lock:
            return self._state

    def transition(self, trigger: str) -> bool:
        """
        Attempt to apply *trigger* to the current state.

        Returns
        -------
        True  — trigger was valid; state has been updated.
        False — trigger is invalid for the current state; state is unchanged.
        """
        from_s: Optional[State] = None
        to_s:   Optional[State] = None
        exit_cbs: List[Callable] = []
        enter_cbs: List[Callable] = []

        with self._lock:
            current = self._state

            if trigger == "manual_override":
                next_state = State.IDLE
            else:
                next_state = _TRANSITIONS.get(current, {}).get(trigger)
                if next_state is None:
                    logger.warning(
                        "[%s] Invalid trigger '%s' in state %s — ignored",
                        _ts(), trigger, current.name,
                    )
                    return False

            from_s, to_s = current, next_state
            exit_cbs, transition_cbs, enter_cbs = self._commit(current, next_state, trigger)

        # Fire callbacks outside the lock so callbacks may call transition()
        # without deadlocking.
        _fire(exit_cbs, from_s, to_s)
        _fire(transition_cbs, from_s, to_s)
        _fire(enter_cbs, from_s, to_s)
        return True

    def register_on_transition(
        self, callback: Callable[[State, State], None]
    ) -> None:
        """Register a callback invoked on every state transition.

        The callback receives ``(from_state, to_state)`` as positional args.
        Fired after per-state exit callbacks and before per-state enter callbacks.
        """
        with self._lock:
            self._on_transition.append(callback)

    def register_on_enter(
        self, state: State, callback: Callable[[State, State], None]
    ) -> None:
        """Register *callback* to be invoked when the machine enters *state*.

        The callback receives ``(from_state, to_state)`` as positional args.
        Exceptions raised inside callbacks are caught and logged.
        """
        with self._lock:
            self._on_enter[state].append(callback)

    def register_on_exit(
        self, state: State, callback: Callable[[State, State], None]
    ) -> None:
        """Register *callback* to be invoked when the machine exits *state*.

        The callback receives ``(from_state, to_state)`` as positional args.
        Exceptions raised inside callbacks are caught and logged.
        """
        with self._lock:
            self._on_exit[state].append(callback)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<StateMachine state={self._state.name}>"

    # ------------------------------------------------------------------
    # Internal helpers  (all must be called with self._lock held,
    # unless stated otherwise)
    # ------------------------------------------------------------------

    def _commit(
        self, from_state: State, to_state: State, trigger: str
    ) -> Tuple[List[Callable], List[Callable], List[Callable]]:
        """
        Update internal state, cancel old timers, start new timers.
        Returns (exit_callbacks, enter_callbacks) to be fired by the caller
        *after* releasing the lock.

        Must be called with self._lock held.
        """
        logger.info(
            "[%s] %s --(%s)--> %s",
            _ts(), from_state.name, trigger, to_state.name,
        )

        self._cancel_timers()
        exit_cbs  = list(self._on_exit[from_state])
        transition_cbs = list(self._on_transition)

        self._state = to_state

        self._start_timer(to_state)
        enter_cbs = list(self._on_enter[to_state])

        return exit_cbs, transition_cbs, enter_cbs

    def _start_timer(self, state: State) -> None:
        """Start the auto-transition timer for timed states.

        Must be called with self._lock held.
        """
        if state == State.ACTIVE_DETECTION:
            self._active_timer = threading.Timer(
                self._active_detection_timeout,
                self._handle_active_timeout,
            )
            self._active_timer.daemon = True
            self._active_timer.start()

        elif state == State.COOLDOWN:
            self._cooldown_timer = threading.Timer(
                self._cooldown_duration,
                self._handle_cooldown_expired,
            )
            self._cooldown_timer.daemon = True
            self._cooldown_timer.start()

    def _cancel_timers(self) -> None:
        """Cancel any pending auto-transition timers.

        Must be called with self._lock held.
        """
        if self._active_timer is not None:
            self._active_timer.cancel()
            self._active_timer = None
        if self._cooldown_timer is not None:
            self._cooldown_timer.cancel()
            self._cooldown_timer = None

    # ------------------------------------------------------------------
    # Timer callbacks  (run in daemon threads — acquire lock themselves)
    # ------------------------------------------------------------------

    def _handle_active_timeout(self) -> None:
        """Auto-triggered when ACTIVE_DETECTION runs past its deadline."""
        logger.info("[%s] ACTIVE_DETECTION timeout fired", _ts())
        from_s: Optional[State] = None
        exit_cbs: List[Callable] = []
        enter_cbs: List[Callable] = []

        with self._lock:
            if self._state is not State.ACTIVE_DETECTION:
                return
            from_s = self._state
            exit_cbs, transition_cbs, enter_cbs = self._commit(from_s, State.COOLDOWN, "timeout")

        _fire(exit_cbs, from_s, State.COOLDOWN)
        _fire(transition_cbs, from_s, State.COOLDOWN)
        _fire(enter_cbs, from_s, State.COOLDOWN)

    def _handle_cooldown_expired(self) -> None:
        """Auto-triggered when COOLDOWN duration has elapsed."""
        logger.info("[%s] COOLDOWN timer expired", _ts())
        from_s: Optional[State] = None
        exit_cbs: List[Callable] = []
        enter_cbs: List[Callable] = []

        with self._lock:
            if self._state is not State.COOLDOWN:
                return
            from_s = self._state
            exit_cbs, transition_cbs, enter_cbs = self._commit(
                from_s, State.IDLE, "cooldown_expired"
            )

        _fire(exit_cbs, from_s, State.IDLE)
        _fire(transition_cbs, from_s, State.IDLE)
        _fire(enter_cbs, from_s, State.IDLE)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    """UTC timestamp string for log messages."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _fire(
    callbacks: List[Callable[[State, State], None]],
    from_state: State,
    to_state: State,
) -> None:
    """Invoke each callback; swallow and log any exceptions."""
    for cb in callbacks:
        try:
            cb(from_state, to_state)
        except Exception:  # noqa: BLE001
            logger.exception("Exception in state-machine callback %s", cb)
