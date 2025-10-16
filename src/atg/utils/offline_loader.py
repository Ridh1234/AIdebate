from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import snapshot_download

from ..config import CONFIG


def detect_device(config_device: str | None = None) -> str:
    if config_device:
        return config_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _ensure_local_model(model_path: Path, model_id: str) -> Path:
    """Ensure model files are present locally; download if missing or empty.

    Returns the local path containing the snapshot.
    """
    # If directory missing or appears empty of HF artifacts, download once
    needs_download = not model_path.exists() or not any(model_path.glob("*"))
    if needs_download:
        model_path.mkdir(parents=True, exist_ok=True)
        # Download a snapshot into this folder
        snapshot_download(
            repo_id=model_id,
            local_dir=str(model_path),
            local_dir_use_symlinks=False,
        )
    return model_path


def load_tokenizer_and_model() -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, str]:
    cfg = CONFIG.model
    device = detect_device(CONFIG.runtime.device)

    model_path: Path = cfg.model_dir
    # Ensure we have the model locally; download if needed
    model_path = _ensure_local_model(model_path, cfg.model_id)

    local_files_only = CONFIG.runtime.offline

    hf_config = AutoConfig.from_pretrained(
        model_path,
        num_labels=len(cfg.labels),
        id2label={i: l for i, l in enumerate(cfg.labels)},
        label2id={l: i for i, l in enumerate(cfg.labels)},
        local_files_only=local_files_only,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        config=hf_config,
        local_files_only=local_files_only,
    )
    model.to(device)
    model.eval()

    return tokenizer, model, device
