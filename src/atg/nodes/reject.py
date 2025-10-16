from __future__ import annotations

from typing import Any, Dict


class RejectNode:
    """Represents a human review/hold state without auto-correction.

    In a real system this could enqueue to a queue or write to a DB. Here we just
    annotate the state and let the CLI present the need for manual action.
    """

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state["needs_review"] = True
        state["events"] = state.get("events", []) + [
            "[RejectNode] Routed to human review (no automatic decision).",
        ]
        return state
