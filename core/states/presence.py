"""Publish PresenceUpdated only when the presence summary actually changes.

Shared by the states that run the face model (VERIFYING_IDENTITY and
ACTIVE_DETECTION).  The last published summary lives in the pipeline
context, so moving between those states doesn't re-announce the same people.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from core.events import PresenceUpdated

_LAST_PRESENCE_KEY = "last_presence"
_EMPTY_SUMMARY: Tuple[Tuple[str, ...], int, int, int] = ((), 0, 0, 0)


def publish_presence_if_changed(
    context: Dict[str, Any], presence: Optional[Dict[str, Any]],
) -> bool:
    """Publish *presence* if it differs from the last published summary."""
    if not presence:
        return False
    summary = (
        tuple(presence.get("authorized") or ()),
        int(presence.get("unknown_count", 0)),
        int(presence.get("uncertain_count", 0)),
        int(presence.get("total", 0)),
    )
    if summary == context.get(_LAST_PRESENCE_KEY, _EMPTY_SUMMARY):
        return False
    context[_LAST_PRESENCE_KEY] = summary
    authorized, unknown_count, uncertain_count, total = summary
    context["event_bus"].publish(PresenceUpdated(
        authorized=list(authorized),
        unknown_count=unknown_count,
        uncertain_count=uncertain_count,
        total=total,
    ))
    return True


def reset_presence(context: Dict[str, Any]) -> None:
    """Forget the last published summary (presence monitoring has stopped)."""
    context.pop(_LAST_PRESENCE_KEY, None)
