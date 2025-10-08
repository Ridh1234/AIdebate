from __future__ import annotations

from typing import Tuple
from .state import DebateState, Utterance


def validate_turn(state: DebateState, speaker: str) -> Tuple[bool, str]:
    expected = state.next_speaker()
    if speaker != expected:
        return False, f"Turn violation: expected {expected}, got {speaker}"
    return True, ""


def validate_repetition(state: DebateState, text: str) -> Tuple[bool, str]:
    normalized = text.strip().lower()
    if normalized in state.used_arguments:
        return False, "Repetition detected: argument already used."
    return True, ""


def validate_coherence(prev_text: str, new_text: str) -> Tuple[bool, str]:
    # Lightweight heuristic: ensure new text is not empty and differs materially
    if not new_text or len(new_text.strip()) < 10:
        return False, "Incoherent: response too short or empty."
    if new_text.strip().lower() == prev_text.strip().lower():
        return False, "Incoherent: identical to previous utterance."
    return True, ""
