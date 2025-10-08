from __future__ import annotations

import os
from rich import print
from rich.prompt import Prompt
import argparse

from .logging_utils import setup_logger, get_logger
from .state import DebateState
from .llm import configure_gemini
from .graph import build_graph, run_debate
from .diagram import save_dag_png


def main():
    # Ensure output dirs exist and configure logging
    os.makedirs("logs", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    log_path = setup_logger()
    logger = get_logger()

    # Configure Gemini
    if os.getenv("USE_MOCK_LLM", "0") != "1":
        try:
            configure_gemini()
        except Exception as e:
            print(f"[bold red]Gemini configuration error:[/bold red] {e}")
            return
    else:
        logger.info("Using Mock LLM (USE_MOCK_LLM=1)")

    parser = argparse.ArgumentParser(description="Multi-Agent Debate (LangGraph + Gemini)")
    parser.add_argument("--topic", type=str, default=os.getenv("DEBATE_TOPIC"), help="Debate topic")
    args = parser.parse_args()

    print("[bold cyan]Multi-Agent Debate (LangGraph + Gemini)[/bold cyan]")
    topic = args.topic or Prompt.ask("Enter topic for debate", default="Should AI be regulated like medicine?")
    print("Starting debate between Scientist and Philosopher...\n")

    # Initial state
    state = DebateState(topic=topic)
    state.log_path = log_path

    # Build graph and diagram
    graph = build_graph()
    out_path = save_dag_png(os.path.join("artifacts", "debate_dag"))
    if out_path and out_path.lower().endswith(".png"):
        logger.info("DAG diagram saved to %s", out_path)
    elif out_path and out_path.lower().endswith(".dot"):
        logger.warning("Graphviz PNG render unavailable; DOT file saved to %s", out_path)
    else:
        logger.warning("Graphviz not available or failed to render the DAG.")

    # Run debate
    final_state = run_debate(graph, state)

    # Print final outputs
    print("[bold]\n[Judge] Summary of debate:[/bold]")
    print(final_state.final_summary or "(no summary)")
    print(f"[bold][Judge] Winner:[/bold] {final_state.final_winner}")
    print(f"[bold]Reason:[/bold] {final_state.final_reason}")
    print(f"\nLog file: {final_state.log_path}")


if __name__ == "__main__":
    main()
