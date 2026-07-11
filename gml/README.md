# gml-orchestration

GML Core Orchestration Layer — context assembly for AI memory injection. This package is the orchestration component of the larger GML system, responsible for assembling and shaping the context that gets injected into AI memory at inference time.

> **Status:** under active development, v0.1.0

## Supported platforms

Linux, macOS, and Windows. CI runs the full test suite on
`ubuntu-latest`, `macos-latest`, and `windows-latest` against Python 3.11
and 3.12. Pure Python; no platform-specific dependencies.

## Setup

```bash
# Clone the repo
git clone <repo-url>
cd gml-orchestration
```

Create and activate a virtualenv. Pick the snippet for your shell:

**macOS / Linux (bash, zsh)**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # then edit .env and fill in GEMINI_API_KEY
```

**Windows (PowerShell)**

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env   # then edit .env and fill in GEMINI_API_KEY
```

> If activation is blocked by execution policy, run once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

**Windows (cmd.exe)**

```cmd
py -3 -m venv .venv
.\.venv\Scripts\activate.bat
pip install -e ".[dev]"
copy .env.example .env
```

**Any OS, one command** — once the venv exists and is activated, the
bundled dev runner gives you a shell-agnostic interface:

```bash
python scripts/dev.py setup                  # install + write .env from template
python scripts/dev.py test                   # run pytest
python scripts/dev.py test -- -q -k ranker   # forward args to pytest after --
python scripts/dev.py metrics                # /metrics server on :9090
python scripts/dev.py clean                  # remove build/cache artifacts
```

## Architecture

The orchestration layer is a modular 7-stage pipeline that turns a plain-text
user query into a target-shaped `TranslatedPayload` ready to ship to an
external AI (Claude, GPT, Gemini, Llama, DeepSeek). Each stage is a
swappable module with one job, typed input/output, and no cross-stage
imports — wiring lives only in `orchestration.pipeline.Pipeline.run`.

```
user query
   → Classifier   (intent/entities/hints)
   → Embedder     (text → vector)
   → Retriever.search
   → branch on "found anything?":
       NO  → SAM.reason_from_scratch()        ──┐
       YES → Retriever.get_top_matches(50)      │
           → Reranker.pick_best(10)             │
           → SAM.resolve_conflicts()            │
   → Assembler.package(final=5) ←───────────────┘
   → Translator → TranslatedPayload
```

**SAM** is its own module (`orchestration/sam/`) called by the Pipeline in
exactly the two sites above. It uses a local DeepSeek R1 8B model served by
Ollama to (a) rewrite the user's query into a clearer form for the target
AI, (b) emit reasoning content the target AI should consider, and (c) drop
memories superseded by newer ones. When Ollama is unreachable, SAM falls
back to a heuristic entity/attribute conflict resolver and the query/
reasoning fields stay empty.

**Translator** is its own module (`orchestration/translator/`) using a
strategy pattern — per-target rendering (Claude / GPT / Gemini / Llama /
DeepSeek) lives only in adapter subclasses. Cursor aliases to the GPT
adapter. No formatting logic leaks into Assembler or SAM.

### End-to-end stack

Beyond the pipeline itself, the package ships everything needed to actually
talk to a target AI and grow memory over time:

- `clients/`       — target-AI client layer (Anthropic, OpenAI, Gemini,
  Ollama for DeepSeek + Llama, plus a Stub for tests)
- `memory_store/`  — JSON Lines on-disk persistence for `MemoryItem` records
- `ingestion/`     — `MemoryExtractor` uses local DeepSeek R1 to extract
  durable memories from each user↔AI turn
- `runner.py`      — `Conversation` wires Pipeline → Client → Extractor →
  Store; holds Session state across turns
- `cli.py`         — `gml ask` / `gml chat` CLI

### CLI quickstart

```bash
# One-shot question against local DeepSeek R1 8B (no API key needed)
gml ask "how is auth implemented?" --target deepseek --semantic-retriever -v

# Multi-turn REPL
gml chat --target deepseek --semantic-retriever

# Hit Claude (set ANTHROPIC_API_KEY first, e.g. in .env)
gml ask "..." --target claude
```

Targets: `deepseek` (default, local), `llama` (local), `claude`, `gpt`,
`gemini`. Cloud targets read keys from env or a `.env` file in CWD.

Persisted memories land in `~/.gml/memories.jsonl` (override with
`--memory-path`). With `--semantic-retriever`, every restart reloads
those memories — the system genuinely learns across sessions.

### HTTP API (`gml serve`)

```bash
gml serve --port 8000 --target deepseek --semantic-retriever
```

Endpoints:
- `POST /chat`        — `{"text", "target", "session_id"?}` → `{"text", "session_id", "items_included", "query_was_improved", "extracted_memories", ...}`
- `POST /chat/stream` — same body, returns Server-Sent Events
- `GET  /memories`    — list persisted memories (`?limit=N`, `?entity=...`)
- `POST /memories`    — add a memory manually
- `GET  /sessions`    — list active sessions in-process
- `GET  /health`      — liveness + version + embedder version + memory count
- `GET  /metrics`     — Prometheus text format

CORS is open by default. Run behind a reverse proxy for production.

### Stack defaults — strong out of the box

| Layer | Default (auto) | Override flag |
|---|---|---|
| Embedder | `FastEmbedEmbedder` — BAAI/bge-small-en-v1.5, 384-dim ONNX, top of MTEB at its size | `--embedder {auto,fastembed,ollama,gemini,stub}` |
| Retriever | `SemanticRetriever` (dense cosine) | `--semantic-retriever` enables persistence-backed |
| Hybrid retrieval | `HybridRetriever(dense + BM25)` available — RRF-fused | (programmatic) |
| Reranker | `ScoreReranker` (semantic+recency+authority+pin) | `CrossEncoderReranker` available — second-pass relevance |
| SAM reasoner | local DeepSeek R1 8B via Ollama | `--no-sam-llm` to disable |
| Tokenizers | tiktoken (GPT), HF real BPE (Llama, DeepSeek), API count (Claude), Gemini approx | (per-target) |

### LOCOMO benchmark

A LOCOMO eval harness is shipped at `scripts/locomo_eval.py`. Download the
LOCOMO dataset separately, then:

```bash
python scripts/locomo_eval.py --data-dir locomo_data --target deepseek --max-conversations 5
```

Metrics: token-level F1, exact-match accuracy, latency, per-category
breakdown (single-hop / multi-hop / temporal / open-domain). See the
script docstring for the expected dataset schema.

### Python quickstart

```python
import asyncio
from orchestration.pipeline import Pipeline, TargetDescriptor, load_config
from orchestration.classifier import KeywordClassifier
from orchestration.embedder import StubEmbedder
from orchestration.retriever import SemanticRetriever, default_records
from orchestration.reranker import ScoreReranker
from orchestration.sam import SAM
from orchestration.translator import Translator
from orchestration.clients import build_default_client_for_target
from orchestration.memory_store import JsonlMemoryStore
from orchestration.ingestion import MemoryExtractor
from orchestration.sam._ollama_client import HTTPOllamaClient
from orchestration.runner import Conversation

async def main():
    config = load_config("config/orchestration.toml")
    target = TargetDescriptor.for_deepseek()

    embedder = StubEmbedder()
    retriever = SemanticRetriever(embedder=embedder)
    store = JsonlMemoryStore("~/.gml/memories.jsonl")
    await retriever.ingest(store.load_all() or default_records())

    pipeline = Pipeline(
        classifier=KeywordClassifier(),
        embedder=embedder,
        retriever=retriever,
        reranker=ScoreReranker(config),
        sam=SAM.with_ollama(),       # local DeepSeek R1 8B
        translator=Translator(),
        config=config,
    )

    conv = Conversation(
        pipeline=pipeline,
        client=build_default_client_for_target(target),
        target=target,
        extractor=MemoryExtractor(client=HTTPOllamaClient()),
        memory_store=store,
        retriever_ingest=retriever.ingest,
    )

    result = await conv.ask("how does auth work and what was the last incident?")
    print(result.response.text)
    print(f"\n[{len(result.extracted_memories)} new memories persisted]")

asyncio.run(main())
```

## Running tests

```bash
pytest
```

## Observability

- **Structured logs**: every orchestration component emits JSON to stdout
  via `orchestration.logging.StructuredLogger`. One log line per event with
  `timestamp`, `level`, `component`, `event`, optional `trace_id`, and
  event-specific kwargs.
- **Prometheus metrics**: counters and histograms registered in
  `orchestration.metrics` (`orchestration_requests_total`,
  `orchestration_latency_seconds`, `orchestration_retrieval_latency_seconds`,
  `orchestration_budget_utilization_ratio`,
  `orchestration_conflicts_detected_total`, `orchestration_failures_total`).
- **`/metrics` endpoint**: a FastAPI app at
  `orchestration.metrics_endpoint:app`. Run with:
  ```bash
  uvicorn orchestration.metrics_endpoint:app --port 9090
  ```
  Then `curl localhost:9090/metrics` returns Prometheus text format.
  `curl localhost:9090/health` returns `{"status": "ok"}`.

## Performance characteristics

The Orchestration layer's latency contract is structured in two parts:

- **Orchestration overhead** (excluding the Translation LLM call):
  p50 < 5ms, p99 < 50ms. Measured by integration tests in stub mode.
  Currently achieved: ~0.1ms p50 on 100-candidate workloads.
- **End-to-end latency** (including the Translation LLM call on a cache
  miss with cold start): dominated by the LLM, typically 2–5 seconds with
  Gemini Flash. The orchestration layer itself contributes <0.5% of total
  time.

Production deployments will rely on the 24-hour Translation cache and the
keyword fallback path to keep end-to-end latency low for repeat queries.
Cold-start LLM cost is paid once per unique query per cache window. If
sub-300ms end-to-end is required for cold queries, future work is needed
(better caching strategies, query parallelization, or an alternative
classifier).

## Architecture decisions

- **`ContextBundle.metadata: dict`** gained in Phase D to mirror
  `InjectionPayload.metadata`. Used for orchestrator-internal flags such as
  `pinned_overflow`, `no_items_fit`, and `degraded_mode` — distinct from
  `session_context`, which is for caller-supplied conversation state.
