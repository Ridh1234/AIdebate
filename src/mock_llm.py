from __future__ import annotations

import itertools


class MockLLM:
    def __init__(self):
        self.counter = itertools.count(1)

    def generate(self, prompt: str) -> str:
        n = next(self.counter)
        if "Output format (strict):" in prompt:
            # judge
            return "Summary: A balanced debate occurred.\nWinner: Scientist\nReason: More concrete risk-based points."
        if "Summarize the debate transcript" in prompt:
            return "Summary of debate so far: key points listed succinctly."
        if "Extract the opponent's most recent" in prompt:
            return "- Opponent noted concerns or freedoms.\n- Address trade-offs."
        return f"Argument #{n}: A new, non-repetitive point relevant to the topic."
