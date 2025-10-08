from __future__ import annotations

import os
from dataclasses import dataclass
import time
import re
import random
from typing import Optional, List

from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import NotFound as GoogleNotFound, ResourceExhausted
from .mock_llm import MockLLM
_MOCK_INSTANCE = MockLLM()
_MODEL_ROTATE_IDX = 0  # round-robin start index across calls


load_dotenv()


def configure_gemini(api_key: Optional[str] = None) -> None:
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set. Provide via .env or environment.")
    genai.configure(api_key=key)


@dataclass
class Persona:
    name: str
    system_preamble: str


SCIENTIST = Persona(
    name="Scientist",
    system_preamble=(
        "You are a pragmatic Scientist. Argue with evidence, risk analysis, and empirical reasoning."
        " Keep claims testable. Avoid repetition. Reply in 1–2 short sentences (<= 40 words)."
    ),
)

PHILOSOPHER = Persona(
    name="Philosopher",
    system_preamble=(
        "You are a thoughtful Philosopher. Argue with principles, ethics, and historical reasoning."
        " Explore trade-offs, autonomy, and societal evolution. Avoid repetition. Reply in 1–2 short sentences (<= 40 words)."
    ),
)


JUDGE_SYSTEM = (
    "You are a neutral Judge. Review the full debate. Provide:"
    " 1) A concise summary (<= 6 sentences) capturing key points from both sides."
    " 2) A verdict: Winner must be exactly 'Scientist' or 'Philosopher'."
    " 3) A brief justification (<= 2 sentences) linked to logical coherence and support."
)


def _resp_to_text(resp) -> str:
    # Try quick accessor first
    try:
        t = getattr(resp, "text", None)
        if t:
            return (t or "").strip()
    except Exception:
        pass

    # Fallback: aggregate from candidates/parts
    out_chunks = []
    try:
        for cand in getattr(resp, "candidates", []) or []:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if parts:
                for p in parts:
                    # part can be dict-like or object with 'text'
                    txt = None
                    if isinstance(p, dict):
                        txt = p.get("text") or p.get("inline_data") or p.get("file_data")
                    else:
                        txt = getattr(p, "text", None)
                    if isinstance(txt, str) and txt.strip():
                        out_chunks.append(txt.strip())
    except Exception:
        pass
    return "\n".join(out_chunks).strip()


def _build_candidates(preferred: Optional[str]) -> List[str]:
    # Allow override via ENV (comma-separated)
    env_list = (os.getenv("GEMINI_FALLBACK_MODELS") or "").strip()
    if env_list:
        items = [s.strip() for s in env_list.split(",") if s.strip()]
    else:
        items = [
            # Current stable family
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-pro",
            # Previous generation
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            # Latest alias
            "gemini-flash-latest",
        ]
    if preferred:
        # Put preferred at the front if not already
        items = [preferred] + [m for m in items if m != preferred]
    # De-dupe while preserving order
    seen = set()
    out: List[str] = []
    for m in items:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def gemini_text(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_output_tokens: Optional[int] = None,
) -> str:
    if os.getenv("USE_MOCK_LLM", "0") == "1":
        return _MOCK_INSTANCE.generate(prompt)

    # Build fallback list of model IDs
    preferred = model or os.getenv("GEMINI_MODEL")
    all_candidates = _build_candidates(preferred)
    # Round-robin start to spread RPM across models
    global _MODEL_ROTATE_IDX
    if all_candidates:
        start = _MODEL_ROTATE_IDX % len(all_candidates)
        candidates = all_candidates[start:] + all_candidates[:start]
        _MODEL_ROTATE_IDX = (_MODEL_ROTATE_IDX + 1) % len(all_candidates)
    else:
        candidates = []

    last_err: Optional[Exception] = None
    rate_waits: List[float] = []
    for attempt in range(2):  # up to two passes across candidate models
        for mid in candidates:
            try:
                model_obj = genai.GenerativeModel(model_name=mid)
                gen_cfg = {"temperature": temperature}
                if max_output_tokens is not None:
                    gen_cfg["max_output_tokens"] = max_output_tokens
                resp = model_obj.generate_content(prompt, generation_config=gen_cfg)
                text = _resp_to_text(resp)
                if text:
                    return text
                # If empty, try one more time with a slightly larger cap or without cap
                retry_cfg = dict(gen_cfg)
                if max_output_tokens is not None:
                    retry_cfg["max_output_tokens"] = min(max_output_tokens + 64, 256)
                resp2 = model_obj.generate_content(prompt, generation_config=retry_cfg)
                text2 = _resp_to_text(resp2)
                if text2:
                    return text2
                raise RuntimeError("Gemini returned no text in candidates.")
            except GoogleNotFound as e:
                last_err = e
                continue
            except ResourceExhausted as e:
                # Rate limited on this model; capture suggested backoff and try next
                last_err = e
                # Try to parse suggested retry seconds from message
                m = re.search(r"retry in\s*([0-9]+(?:\.[0-9]+)?)s", str(e), re.IGNORECASE)
                if m:
                    sec = float(m.group(1))
                    rate_waits.append(sec)
                else:
                    rate_waits.append(8.0)
                continue
            except Exception as e:
                # If the model produced no text, try next candidate
                if isinstance(e, RuntimeError) and "no text" in str(e).lower():
                    last_err = e
                    continue
                # For other errors, keep last and break to surface
                last_err = e
                break
        # If we completed a full pass with only rate limits, backoff and retry once
        if attempt == 0 and rate_waits:
            sleep_for = min(max(min(rate_waits), 3.0), 20.0)  # clamp between 3s and 20s
            # Add small jitter to reduce herd effect
            sleep_for += random.uniform(0, 1.5)
            time.sleep(sleep_for)
            rate_waits.clear()
            continue
        break
    if last_err:
        raise last_err
    raise RuntimeError("No Gemini model candidates available")


def build_agent_prompt(persona: Persona, topic: str, turn_index: int, memory: str) -> str:
    return (
        f"System: {persona.system_preamble}\n"
        f"Debate Topic: {topic}\n"
        f"Your Turn: {persona.name} (turn {turn_index})\n"
        f"Relevant Memory So Far:\n{memory}\n"
    "Instruction: Present a new argument (no repetition), logically coherent with the debate so far."
        " Output MUST be 1–2 sentences, max ~40 words."
    " This is a harmless academic debate; avoid any unsafe content."
    )


def build_judge_prompt(topic: str, memory_summary: str, transcript: str) -> str:
    return (
        f"System: {JUDGE_SYSTEM}\n"
        f"Debate Topic: {topic}\n"
        f"Memory Summary:\n{memory_summary}\n\n"
        f"Full Transcript:\n{transcript}\n\n"
        "Output format (strict):\n"
        "Summary: <<=6 sentences>\nWinner: <Scientist|Philosopher>\nReason: <<=2 sentences>"
    )
