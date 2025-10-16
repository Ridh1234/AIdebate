from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn.functional as F

from ..config import CONFIG
from ..utils.offline_loader import load_tokenizer_and_model


@dataclass
class InferenceInput:
    text: str


@dataclass
class InferenceOutput:
    text: str
    label: str
    confidence: float
    logits: list[float]


class InferenceNode:
    def __init__(self) -> None:
        self.tokenizer, self.model, self.device = load_tokenizer_and_model()

    @torch.no_grad()
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        text: str = state["text"]
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=CONFIG.model.max_length,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)
        logits = outputs.logits.squeeze(0)
        probs = F.softmax(logits, dim=-1)
        conf, pred_id = torch.max(probs, dim=-1)
        label = CONFIG.model.labels[pred_id.item()]

        out = InferenceOutput(
            text=text,
            label=label,
            confidence=conf.item(),
            logits=logits.tolist(),
        )
        # Merge into state for downstream nodes
        state.update(
            {
                "prediction": out.label,
                "confidence": out.confidence,
                "logits": out.logits,
                "events": state.get("events", [])
                + [
                    f"[InferenceNode] Predicted label: {label} | Confidence: {out.confidence:.2f}"
                ],
            }
        )
        return state
