from __future__ import annotations

import sys
from typing import Any, Dict

import typer
from rich.console import Console
from rich.table import Table

from .config import CONFIG
from .logging_setup import setup_logging
def _lazy_graph():
    from .graph.flow import build_graph, draw_graph_png
    return build_graph, draw_graph_png
from .utils.offline_loader import _ensure_local_model

app = typer.Typer(add_completion=False, help="Self-Healing Classification DAG CLI")
console = Console()


@app.command()
def run() -> None:
    """Run the interactive CLI for classification with fallback."""
    logger = setup_logging()
    console.print("[bold]Self-Healing Classification DAG[/bold]")

    build_graph, _draw_graph = _lazy_graph()
    graph = build_graph()

    while True:
        try:
            text = typer.prompt("Enter text (or blank to quit)")
        except (EOFError, KeyboardInterrupt):
            console.print("\nExiting.")
            break
        if not text.strip():
            console.print("Bye.")
            break

        state: Dict[str, Any] = {"text": text}
        result = graph.invoke(state)
        pred = result.get("prediction", "unknown")
        conf = float(result.get("confidence", 0.0))
        corrected = bool(result.get("corrected", False))
        needs_review = bool(result.get("needs_review", False))
        correction_source = result.get("correction_source") or "-"
        events = list(result.get("events", []))

        # Pretty output
        table = Table(show_header=True, header_style="bold")
        table.add_column("Field")
        table.add_column("Value")
        table.add_row("Prediction", str(pred))
        table.add_row("Confidence", f"{conf:.2f}")
        table.add_row("Corrected", "Yes" if corrected else "No")
        table.add_row("Correction Source", str(correction_source))
        table.add_row("Needs Review", "Yes" if needs_review else "No")
        console.print(table)

        if events:
            console.rule("Events")
            for e in events:
                console.print(e)
            console.rule()

        # Structured log
        logger.info(
            "text=%s | prediction=%s | confidence=%.4f | corrected=%s | needs_review=%s | correction_source=%s",
            text,
            pred,
            conf,
            corrected,
            needs_review,
            correction_source,
        )


@app.command()
def draw(output: str = "graph.png") -> None:
    """Export a professional-looking DAG visualization PNG to artifacts folder."""
    setup_logging()
    _build_graph, draw_graph_png = _lazy_graph()
    draw_graph_png(output)
    console.print(f"Graph saved to [green]{CONFIG.runtime.artifacts_dir / output}[/green]")


@app.command()
def bootstrap() -> None:
    """Pre-download model files (first-time setup) using Hugging Face Hub."""
    setup_logging()
    local_dir = _ensure_local_model(CONFIG.model.model_dir, CONFIG.model.model_id)
    console.print(f"Model prepared at [green]{local_dir}[/green]")


if __name__ == "__main__":
    app()

