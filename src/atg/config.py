from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel


class ModelConfig(BaseModel):
    model_id: str = "distilbert-base-uncased"
    model_dir: Path = Path("models") / "distilbert-base-uncased"
    task: str = "sequence-classification"  # used for pipeline selection
    labels: list[str] = ["negative", "positive"]  # default sentiment labels
    max_length: int = 256


class RuntimeConfig(BaseModel):
    confidence_threshold: float = 0.60  # below triggers fallback
    ask_confirmation_prompt: str = (
        "Could you clarify your intent? Was this a negative review? [y/n]: "
    )
    device: str | None = None  # "cpu" or "cuda"; auto if None
    offline: bool = True  # load from local files only
    log_dir: Path = Path("logs")
    artifacts_dir: Path = Path("artifacts")


class AppConfig(BaseModel):
    model: ModelConfig = ModelConfig()
    runtime: RuntimeConfig = RuntimeConfig()


CONFIG = AppConfig()
