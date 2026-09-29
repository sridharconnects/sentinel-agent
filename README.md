# 🛰️ sentinel-agent — Real-time Multi-Agent Intel Analyst

I built a multi-agent research system that takes a question, fans out across
specialized agents (supervisor → researcher → analyst → critic), and streams a
**cited briefing** to the browser in real time over SSE. It runs fully offline
in mock mode — no API keys needed.

## Problem framing

Single-prompt LLM answers to research questions are hard to trust: no
citations, no visible process, no second opinion. I wanted a system where
**research, synthesis, and verification are separate agents** that check each
other, and where the user watches the work happen live instead of staring at
a spinner.

## Architecture

```
                    ┌─────────────┐
  question ────────▶│ supervisor  │  decomposes into sub-queries
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │ researcher  │──▶ web_search (DuckDuckGo / mock)
                    └──────┬──────┘
                           ▼ evidence [1..n]
                    ┌─────────────┐
              ┌────▶│ analyst     │  synthesizes cited draft (tokens streamed)
              │     └──────┬──────┘
              │            ▼
              │     ┌─────────────┐
              │     │ critic      │  groundedness check: every bullet cited?
              │     └──────┬──────┘
              │      FAIL  │ (≤3 revision rounds)   PASS
              └────────────┘            │
                                        ▼
                                 ┌─────────────┐
                                 │ finalizer   │──▶ blocklist ──▶ briefing
                                 └─────────────┘

  Tools are ALSO exposed via MCP:  app/mcp_server.py (FastMCP, stdio)
                                   app/mcp_client.py (demo client)
```

Every node yields typed events (`node_start`, `tool_call`, `token`,
`node_end`, `guardrail`, `error`, `done`); the FastAPI layer turns them into
an SSE stream the demo page renders live.

## How to run

```bash
pip install -r requirements.txt
cp .env.example .env   # optional

# Mock mode (default): zero keys, deterministic
uvicorn app.api:app --reload
# open http://localhost:8000  → ask a question, watch it stream

# Real models
LLM_MODE=openai  OPENAI_API_KEY=sk-... uvicorn app.api:app   # OpenAI
LLM_MODE=ollama  OLLAMA_MODEL=llama3.1 uvicorn app.api:app   # local Ollama

# Non-streaming JSON API
curl -X POST localhost:8000/brief -H 'Content-Type: application/json' \
  -d '{"question":"What are the latest developments in agentic AI evaluation?"}'

# MCP server (stdio) + client demo
python -m app.mcp_server        # in one terminal (or attach any MCP host)
python -m app.mcp_client        # lists tools, calls calculator_tool
```

## How to run tests

```bash
pytest -q
```

The suite runs entirely in mock mode (`LLM_MODE=mock` is the default): tool
schemas, calculator safety, critic loop termination (always-fail stub stops
after exactly 3 revisions), guardrail caps, SSE event ordering via
`TestClient`, and the LangGraph wiring (skipped if `langgraph` isn't
installed).

## Guardrails

- **Input cap**: questions over `MAX_INPUT_CHARS` (default 2000) are rejected.
- **Wall-clock timeout**: `RUN_TIMEOUT_SECONDS` (default 120) checked between
  every node and streamed event; the run aborts with an `error` event.
- **Critic revision cap**: `MAX_CRITIC_ROUNDS` (default 3) — the
  analyst↔critic loop always terminates.
- **Output blocklist**: substring redaction on the final briefing
  (documented as illustrative — production would use a real content filter).

## Repo layout

```
app/
  api.py          FastAPI app: /stream (SSE), /brief (JSON), /health, demo page
  agents.py       node functions + stream_briefing() orchestration + guardrails
  graph.py        same nodes wired as a LangGraph StateGraph (optional path)
  llm.py          mock / OpenAI / Ollama backends behind one interface
  tools.py        web_search (ddgs, offline-safe mock fallback), calculator, today
  mcp_server.py   FastMCP server exposing the tools over stdio
  mcp_client.py   demo client: list tools, call one over stdio
web/index.html    EventSource demo page rendering the live briefing
tests/            pytest suite (mock mode, no keys)
```

## What I'd do next in production

1. Replace the substring blocklist with a proper moderation/DLP service and
   add PII redaction on *inputs*.
2. Persistent run store (Postgres/Redis): resume interrupted briefings, audit
   trail of tool calls for compliance.
3. Swap mock search for a paid news API with source-credibility scoring, and
   add a re-ranker over evidence before synthesis.
4. Per-tenant rate limits + API keys on `/stream`; OpenTelemetry tracing
   across agent hops.

## Interview talking points

1. **Separation of concerns beats a bigger prompt**: the critic catches
   uncited claims the analyst makes — I can demo the FAIL→revise loop live.
2. **Streaming architecture**: typed node events → SSE means the UI shows
   *which agent* is working, not just tokens; the same event schema drives
   both the demo page and tests.
3. **Guardrails as first-class code**: timeout, input caps, and revision caps
   are enforced in the orchestrator (not hoped for in the prompt), and each
   one has a failing test proving it triggers.

## Troubleshooting

- `langgraph` / `mcp` / `ddgs` missing: the API, mock pipeline, and tests
  still run — those imports are lazy/guarded. Install full requirements for
  the LangGraph path, live web search, and the MCP server.
- Ollama errors: make sure `ollama serve` is running and the model is pulled
  (`ollama pull llama3.1`).
