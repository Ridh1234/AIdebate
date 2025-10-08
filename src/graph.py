from __future__ import annotations

import os
from typing import Callable

from langgraph.graph import StateGraph, END

from .state import DebateState
from .nodes import user_input_node, agent_node, memory_node, judge_node
from .logging_utils import get_logger


def build_graph() -> StateGraph:
    graph = StateGraph(DebateState)

    # Define nodes
    graph.add_node("user_input", user_input_node)  # topic is already set on state; this node logs
    graph.add_node("agent", agent_node)
    graph.add_node("memory", memory_node)
    graph.add_node("judge", judge_node)

    # Start at user input node
    graph.set_entry_point("user_input")

    def route(state: DebateState) -> str:
        # After memory, either continue to next agent turn or go to judge
        return "judge" if len(state.transcript) >= 8 else "agent"

    # Edges
    graph.add_edge("user_input", "agent")
    graph.add_edge("agent", "memory")
    graph.add_conditional_edges("memory", route, {"agent": "agent", "judge": "judge"})
    graph.add_edge("judge", END)

    return graph


def run_debate(graph: StateGraph, initial_state: DebateState) -> DebateState:
    logger = get_logger()
    app = graph.compile()
    state = initial_state

    # Single pass through the graph will execute:
    # user_input -> (agent -> memory)xN -> judge -> END
    result = app.invoke(state)

    # LangGraph may return a dict-like state; coerce to DebateState for downstream use
    if isinstance(result, dict):
        try:
            state_obj = DebateState(**result)
        except Exception:
            # Fallback: partial hydration
            state_obj = initial_state.model_copy(update=result)  # type: ignore
    else:
        state_obj = result  # already DebateState

    winner = getattr(state_obj, "final_winner", None) or (result.get("final_winner") if isinstance(result, dict) else None)
    logger.info("Debate finished with winner: %s", winner)
    return state_obj
