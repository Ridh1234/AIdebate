from __future__ import annotations

from typing import Any, Dict

from ..config import CONFIG


class ConfidenceCheckNode:
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        threshold = CONFIG.runtime.confidence_threshold
        conf = float(state.get("confidence", 0.0))

        if conf >= threshold:
            route = "accept"
            msg = f"[ConfidenceCheckNode] Confidence {conf:.2f} >= {threshold:.2f}. Accepting."
        else:
            route = "fallback"
            msg = f"[ConfidenceCheckNode] Confidence {conf:.2f} < {threshold:.2f}. Triggering fallback..."

        state["route"] = route
        state["events"] = state.get("events", []) + [msg]
        return state
