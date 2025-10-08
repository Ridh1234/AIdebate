from __future__ import annotations

from typing import List, Literal, Optional, Dict, Any, Set
from pydantic import BaseModel, Field


Speaker = Literal["Scientist", "Philosopher"]


class Utterance(BaseModel):
    turn: int
    speaker: Speaker
    text: str


class DebateState(BaseModel):
    topic: str
    turn_index: int = 0  # 0..7
    round_number: int = 1  # 1..8
    current_speaker: Speaker = "Scientist"

    transcript: List[Utterance] = Field(default_factory=list)
    memory_summary: str = ""
    agent_memory: Dict[Speaker, str] = Field(default_factory=lambda: {"Scientist": "", "Philosopher": ""})

    used_arguments: Set[str] = Field(default_factory=set)
    errors: List[str] = Field(default_factory=list)

    final_summary: Optional[str] = None
    final_winner: Optional[Speaker] = None
    final_reason: Optional[str] = None

    log_path: Optional[str] = None
    artifacts_dir: str = "artifacts"
    logs_dir: str = "logs"

    def next_speaker(self) -> Speaker:
        return "Scientist" if self.turn_index % 2 == 0 else "Philosopher"

    def is_finished(self) -> bool:
        return self.round_number > 8
