from __future__ import annotations

from typing import Tuple
from .state import DebateState, Utterance
from .validators import validate_turn, validate_repetition, validate_coherence
from .llm import configure_gemini, gemini_text, build_agent_prompt, build_judge_prompt, SCIENTIST, PHILOSOPHER
from .logging_utils import get_logger
from rich import print


def user_input_node(state: DebateState) -> DebateState:
    logger = get_logger()
    logger.info("[UserInputNode] Topic: %s", state.topic)
    return state


def get_persona(name: str):
    return SCIENTIST if name == "Scientist" else PHILOSOPHER


def agent_node(state: DebateState) -> DebateState:
    logger = get_logger()
    speaker = state.next_speaker()
    ok, msg = validate_turn(state, speaker)
    if not ok:
        state.errors.append(msg)
        logger.error("[AgentNode] %s", msg)
        raise RuntimeError(msg)

    persona = get_persona(speaker)
    memory = state.agent_memory.get(speaker, state.memory_summary)
    prompt = build_agent_prompt(persona, state.topic, state.turn_index + 1, memory)

    # Call Gemini
    text = gemini_text(prompt, temperature=0.7, max_output_tokens=64)

    # Validations
    ok, msg = validate_repetition(state, text)
    if not ok:
        state.errors.append(msg)
        logger.warning("[AgentNode] %s -> re-asking model with adjusted temperature", msg)
        # Re-ask with lower temperature to try different output
        text = gemini_text(
            prompt + "\nAvoid repeating previous arguments explicitly.",
            temperature=0.2,
            max_output_tokens=64,
        )
        ok2, msg2 = validate_repetition(state, text)
        if not ok2:
            raise RuntimeError("Repeated argument after retry")

    prev_text = state.transcript[-1].text if state.transcript else ""
    ok, msg = validate_coherence(prev_text, text)
    if not ok:
        state.errors.append(msg)
        logger.warning("[AgentNode] %s -> adjusting", msg)
        text = gemini_text(
            prompt + "\nEnsure coherence and substantive content.",
            temperature=0.6,
            max_output_tokens=64,
        )

    utt = Utterance(turn=state.turn_index + 1, speaker=speaker, text=text)
    state.transcript.append(utt)
    state.used_arguments.add(text.strip().lower())
    logger.info("[AgentNode][Round %d] %s: %s", state.round_number, speaker, text)
    print(f"[Round {state.round_number}] {speaker}: {text}")

    # Advance turn
    state.turn_index += 1
    state.round_number += 1
    state.current_speaker = state.next_speaker()
    return state


def memory_node(state: DebateState) -> DebateState:
    logger = get_logger()
    # Summarize transcript so far to keep memory succinct
    transcript_text = "\n".join([f"[Round {u.turn}] {u.speaker}: {u.text}" for u in state.transcript])
    # Global concise summary (for judge and logging)
    summary_prompt = (
        "Summarize the debate transcript in 4–6 concise sentences, focusing only on new points per round.\n"
        f"Transcript:\n{transcript_text}"
    )
    summary = gemini_text(summary_prompt, temperature=0.4, max_output_tokens=160)
    state.memory_summary = summary

    # Persona-specific, relevance-routed memories: focus on opponent's latest 1-2 points
    def opponent_of(name: str) -> str:
        return "Philosopher" if name == "Scientist" else "Scientist"

    for speaker in ("Scientist", "Philosopher"):
        opp = opponent_of(speaker)
        opp_utts = [u for u in state.transcript if u.speaker == opp][-2:]
        opp_snippets = "\n".join([f"{u.speaker}: {u.text}" for u in opp_utts])
        routed_prompt = (
            "Extract opponent's last 1–2 distinct points as max 4 bullets."
            " Keep bullets <= 12 words each; avoid repetition.\n"
            f"Opponent: {opp}\n"
            f"Transcript:\n{transcript_text}\n"
            f"Recent opponent snippets:\n{opp_snippets}"
        )
        routed = gemini_text(routed_prompt, temperature=0.2, max_output_tokens=96)
        state.agent_memory[speaker] = routed

    logger.info("[MemoryNode] Memory updated (summary length=%d)", len(summary))
    logger.info("[MemoryNode] Summary: %s", summary)
    logger.info("[MemoryNode] Scientist view: %s", state.agent_memory["Scientist"]) 
    logger.info("[MemoryNode] Philosopher view: %s", state.agent_memory["Philosopher"]) 
    return state


def judge_node(state: DebateState) -> DebateState:
    logger = get_logger()
    transcript_text = "\n".join([f"[Round {u.turn}] {u.speaker}: {u.text}" for u in state.transcript])
    prompt = build_judge_prompt(state.topic, state.memory_summary, transcript_text)
    judgment = gemini_text(prompt, temperature=0.2, max_output_tokens=128)

    # Parse strict format
    summary, winner, reason = "", None, ""
    for line in judgment.splitlines():
        if line.lower().startswith("summary:"):
            summary = line.split(":", 1)[1].strip()
        elif line.lower().startswith("winner:"):
            w = line.split(":", 1)[1].strip()
            if w in ("Scientist", "Philosopher"):
                winner = w
        elif line.lower().startswith("reason:"):
            reason = line.split(":", 1)[1].strip()

    if not winner:
        # fallback: choose based on last turn index parity
        winner = "Scientist" if (len(state.transcript) % 2 == 1) else "Philosopher"

    state.final_summary = summary
    state.final_winner = winner  # type: ignore
    state.final_reason = reason

    logger.info("[Judge] Summary: %s", summary)
    logger.info("[Judge] Winner: %s", winner)
    logger.info("[Judge] Reason: %s", reason)
    return state
