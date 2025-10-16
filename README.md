# Self-Healing Classification DAG (ATG)

This project implements a LangGraph-based, self-healing text classification pipeline using a fine-tuned DistilBERT model (offline). It prioritizes correctness via a confidence-aware fallback that requests user clarification or uses a backup strategy.

Highlights:
- Offline-ready fine-tuned DistilBERT sentiment/topic classifier.
- LangGraph DAG with InferenceNode, ConfidenceCheckNode, and FallbackNode.
- Clean CLI built with Typer + Rich.
- Structured logging to `logs/run.log`.
- Graph visualization export to `artifacts/graph.png`.

## Quickstart

1) Create a virtual environment and install deps

```powershell
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e .
```

2) Model setup

Use the files provided in the `my_merged_model/content/my_merged_model/` folder. These files contain the fine-tuned model and tokenizer required for offline inference and evaluation. No download or bootstrap is needed.

3) Run the CLI

```powershell
python -m atg.cli run
```

Example interaction:

```
> The movie was painfully slow and boring.
[InferenceNode] Predicted label: negative | Confidence: 0.54
[ConfidenceCheckNode] Confidence too low. Triggering fallback...
[FallbackNode] Could you clarify your intent? Was this a negative review? [y/n]: y
Final Label: negative (Corrected via user clarification)
```

4) Export graph image

```powershell
python -m atg.cli draw
```
Outputs to `artifacts/graph.png`.

## Project Structure

- `src/atg/` core package
  - `cli.py` Typer-based CLI entry
  - `config.py` thresholds and model settings
  - `logging_setup.py` structured logging
  - `graph/flow.py` LangGraph DAG construction and visualization
  - `nodes/` node implementations
  - `inference.py` classifier inference with fine-tuned DistilBERT
    - `confidence.py` confidence check and routing
    - `fallback.py` user clarification and optional backup
  - `utils/offline_loader.py` offline HF model/tokenizer loader
- `artifacts/` exported graph and temp files (created at runtime)
- `logs/` structured logs

## Notes
- This setup avoids network calls by default (`local_files_only=True`). The fine-tuned model is already provided locally as described above.
- You can change labels and prompts in `config.py`.

## Dev
```powershell
ruff check src
pytest -q
```
