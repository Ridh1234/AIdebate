# Multi-Agent Debate DAG using LangGraph (Gemini)

A CLI-based debate simulation where two AI agents (Scientist vs Philosopher) debate a user-provided topic for 8 rounds. The system uses LangGraph to orchestrate turn-taking, memory, validation, logging, and a judge that declares a winner.

## Features
- Two alternating agents with distinct personas
- Exactly 8 rounds (4 per agent), strict turn control
- Memory node that summarizes and routes only relevant context to each agent
- Validators to prevent repeated arguments and ensure logical coherence
- Judge node produces a debate summary and declares a winner with justification
- Full logging of messages, memory updates, state transitions, and final verdict to a log file
- DAG diagram generation (Graphviz)

## Tech
- LangGraph
- Google Gemini (via `google-generativeai`) — set `GEMINI_API_KEY`
- Pydantic for typed state
- Rich for CLI output
- Graphviz for DAG diagram

## Setup
1. Python 3.10+
2. Install dependencies
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```
3. Configure API key (create `.env` in project root)
```
GEMINI_API_KEY=your_key_here
```

## Run
```powershell
python -m src.main
```
Example interaction:
```
Enter topic for debate: Should AI be regulated like medicine?
Starting debate between Scientist and Philosopher...
[Round 1] Scientist: ...
...
[Round 8] Philosopher: ...
[Judge] Summary of debate:
...
[Judge] Winner: Scientist
Reason: ...
```

## Outputs
- Log file: `logs/debate_<timestamp>.log`
- DAG diagram: `artifacts/debate_dag.png`

## Structure
- `src/`
  - `main.py` — CLI entrypoint
  - `graph.py` — LangGraph DAG construction
  - `nodes.py` — UserInputNode, AgentA/B, MemoryNode, JudgeNode
  - `state.py` — Pydantic state models
  - `validators.py` — turn control, repetition check, coherence
  - `llm.py` — Gemini client wrapper and prompts
  - `logging_utils.py` — structured logging
  - `diagram.py` — DAG export

## Notes
- The system uses minimal prompt templates and relies on validation to keep the debate coherent and non-repetitive.
- If Graphviz isn't installed system-wide, the PNG export may fail; you can still run the debate without the diagram.

## Demo Video
Record a 2–4 minute walkthrough (screen + face-cam) covering project structure, CLI run, and judge decision. Include the file or a shareable link in the repo root.