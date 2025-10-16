from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import END, START, StateGraph
try:
    from langgraph.checkpoint import MemorySaver  # type: ignore
except Exception:  # pragma: no cover - older/newer versions may differ
    MemorySaver = None  # fallback: no checkpointing

from ..nodes.inference import InferenceNode
from ..nodes.confidence import ConfidenceCheckNode
from ..nodes.fallback import FallbackNode
from ..config import CONFIG


def build_graph() -> StateGraph:
    graph = StateGraph(dict)

    # Nodes
    graph.add_node("inference", InferenceNode())
    graph.add_node("confidence_check", ConfidenceCheckNode())
    graph.add_node("fallback", FallbackNode())

    # Edges
    graph.add_edge(START, "inference")
    graph.add_edge("inference", "confidence_check")

    # Conditional route from confidence check
    def route_fn(state: Dict[str, Any]) -> str:
        return state.get("route", "accept")

    graph.add_conditional_edges(
        "confidence_check",
        route_fn,
    {"accept": END, "fallback": "fallback"},
    )

    # After fallback, end
    graph.add_edge("fallback", END)

    # Memory for inspection (optional)
    if MemorySaver is not None:
        memory = MemorySaver()
        return graph.compile(checkpointer=memory)
    return graph.compile()


class GraphRenderError(RuntimeError):
    pass


def draw_graph_png(path: str) -> None:
    # Lazy import to avoid graphviz requirement at runtime if undesired
    from graphviz import Digraph

    dot = Digraph("SelfHealingDAG", format="png")
    dot.attr(rankdir="LR", bgcolor="white")

    # Professional, minimal styling
    node_style = {
        "shape": "box",
        "style": "rounded",
        "color": "#555555",
        "fontname": "Helvetica",
    }
    edge_style = {"color": "#777777"}

    def add_node(name: str, label: str):
        dot.node(name, label=label, **node_style)

    add_node("start", "START")
    add_node("inference", "InferenceNode\n(DistilBERT)")
    add_node("confidence_check", "ConfidenceCheckNode\n(threshold→accept/fallback)")
    add_node("fallback", "FallbackNode\n(user clarification)")
    add_node("end", "END")

    # Edges
    dot.edge("start", "inference", **edge_style)
    dot.edge("inference", "confidence_check", **edge_style)
    dot.edge("confidence_check", "end", label="accept", fontsize="10", **edge_style)
    dot.edge("confidence_check", "fallback", label="fallback", fontsize="10", **edge_style)
    dot.edge("fallback", "end", **edge_style)

    out_dir = CONFIG.runtime.artifacts_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / path
    try:
        dot.render(out_path.with_suffix("")).replace(".gv", ".png")
    except Exception as e:
        # Fallback: write the DOT file for manual rendering
        dot_path = out_path.with_suffix(".dot")
        dot_path.write_text(dot.source, encoding="utf-8")
        raise GraphRenderError(
            f"Graphviz 'dot' not available. Wrote DOT to {dot_path}. "
            "Install Graphviz and ensure 'dot' is on PATH to render PNG."
        ) from e
