from __future__ import annotations

from typing import Any, Dict

from rich.prompt import Confirm

from ..config import CONFIG


class FallbackNode:
    def __init__(self) -> None:
        # Placeholder for optional backup strategy (e.g., rule-based or zero-shot)
        pass

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Ask user for confirmation as a human-in-the-loop clarification
        prompt = CONFIG.runtime.ask_confirmation_prompt
        is_negative = Confirm.ask(prompt, default=False)

        # Simple correction logic for binary sentiment labels
        pred = state.get("prediction", "unknown")
        if is_negative and pred != "negative":
            state["prediction"] = "negative"
            state["corrected"] = True
            state["correction_source"] = "user_clarification"
        elif not is_negative and pred != "positive":
            state["prediction"] = "positive"
            state["corrected"] = True
            state["correction_source"] = "user_clarification"
        else:
            state["corrected"] = False
            state["correction_source"] = None

        state["events"] = state.get("events", []) + [
            "[FallbackNode] User clarification handled; state updated accordingly.",
        ]
        return state
