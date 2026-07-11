# GML — Codebase Architecture (What every piece is)

GML ("Generative Memory Layer") is a context-assembly engine: given a user's
query and a target AI model, it retrieves the right memories, resolves
conflicts between them, and formats a context block the target model can use.
It also ingests conversation turns into durable, structured memories.

This doc explains **what each component is and how they chain**. For the
deployment/data-flow/MCP view see [`DATA_FLOW_AND_DEPLOYMENT.md`](./DATA_FLOW_AND_DEPLOYMENT.md).

---

## 1. The request lifecycle (the Pipeline)

`orchestration/pipeline/pipeline.py` is the **only** orchestrator — stages know
nothing about each other; the branch logic lives here.

```
plain query text + TargetDescriptor + user_id
  │
  1. Classifier   → Classification (intent_type, entities, hints, confidence)
  2. Embedder     → EmbeddedQuery (a dense vector)
  3. Retriever.search (cheap probe) → "did we find anything?"
        │
        ├── NO  → SAM.reason_from_scratch()  → ResolvedMemorySet (no memories)
        │
        └── YES → Retriever.get_top_matches(k≈50)
                → Reranker.pick_best(k≈10)
                → should_skip_sam()?  yes → keep top-k as-is
                                       no → SAM.resolve_conflicts()
                → ResolvedMemorySet (kept, superseded, notes)
  4. Assembler.package(resolved, budget)   → AssembledContext
  5. Translator.translate(context, target) → TranslatedPayload  ← shipped to the model
```

Everything is an ABC with swappable implementations (stub/real), so the same
flow runs in tests, on JSONL dev, and on the Postgres production backend.

---

## 2. Stage-by-stage

### Classifier — `orchestration/classifier/`
Turns the raw query into a `Classification`: intent type, extracted entities,
retrieval hints, confidence. Two impls: `KeywordClassifier` (fast, rule-based,
the default in production) and `LLMClassifier` (LLM-backed, higher fidelity).
Entities feed the embedder (mixed into the embed signal) and the retriever
(entity-boosting).

### Embedder — `orchestration/embedder/`
Turns text into a dense vector. The `Embedder` ABC has `embed(query)` (query-
shaped) and `embed_batch(texts)` (document-shaped, used on the write path).
Implementations:
- `FastEmbedEmbedder` — bge-small ONNX, local, no key (default dev).
- `SentenceTransformerEmbedder` — loads a **local sentence-transformers model**,
  e.g. the **FT'd embedder** (`GML_EMBEDDER=st`, `GML_ST_EMBED_MODEL=…`). Used in production.
- `OllamaEmbedder`, `GeminiEmbedder`, `StubEmbedder`, `HydeEmbedder` (HyDE wrapper).

All output 384-dim vectors to match the `memories.embedding vector(384)` column.
**Query and stored memories must use the same embedder** or their vectors live
in different spaces — the API and the MCP path are both wired to the FT one.

### Retriever — `orchestration/retriever/`
`HybridRetriever` fuses two rankers with **Reciprocal Rank Fusion** (RRF,
`score = Σ 1/(k+rank)`, k≈60):
- **dense** — semantic. `PgvectorSemanticRetriever` (cosine `<=>` over the HNSW
  index) in production; `SemanticRetriever` (in-memory) on JSONL.
- **sparse** — lexical. `PgvectorBM25Retriever` (`ts_rank_cd` over `content_tsv`)
  / `BM25Retriever`. Catches exact terms (version strings, rare names).

Dense = semantic intent, sparse = exact recall; fusing lifts top-k recall ~10–20%.
Optional wrappers refine results: `EntityBoostedRetriever`, `MultiHopAwareRetriever`,
`TimeAwareRetriever`. All are RLS-scoped to the requesting `user_id`.

### Reranker — `orchestration/reranker/`
Re-orders the ~50 candidates down to the best ~10 by *direct-answer relevance*.
`make_reranker(config)` picks the best available with graceful fallback:
- `ScoreReranker` — weighted sum of similarity + recency + authority + pin.
  Last-resort fallback (no model, tiny).
- `CrossEncoderReranker` / `SentenceTransformerCrossEncoder` — a cross-encoder
  scores (query, memory) pairs. Needs torch + sentence-transformers.
- `EnsembleCrossEncoder` — blends two cross-encoders (e.g. the **FT
  `ce_locomo_ft`** + jina-reranker-v2 at weight 0.7), set via `GML_CE_ENSEMBLE`. **(production)**
- `TwoStageCrossEncoderReranker` — cross-encoder then score-weighting.
- `ThreeStageReranker` — cross-encoder → **LLM reranker** → score (highest
  precision, ~1–2s; for hard multi-hop/temporal/paraphrase queries).
- Wrappers: `NegationAwareReranker` (don't surface "we do NOT use X" for "do we
  use X"), `TemporalAwareReranker` (recency/"latest" handling), `LLMReranker`.

### SAM — `orchestration/sam/` — **the conflict resolver**
"SAM" = **Semantic Alignment Module**. The Pipeline calls it in two cases:
1. **`resolve_conflicts(query, ranked)`** — given reranked candidates, SAM
   (a) detects **superseded** memories (newer fact replaces older — e.g. "we use
   Adyen" → later "we switched to Stripe") and drops the stale ones,
   (b) rewrites the user's query informed by the memories, (c) emits reasoning
   for the target AI. **This is the conflict-resolution step.**
2. **`reason_from_scratch(query, classification)`** — retriever found nothing;
   SAM uses its LLM to improve the bare question + add reasoning.

SAM uses an **LLM reasoner** (production: the **FT Qwen** `gml-qwen-ft` via Ollama;
default elsewhere: DeepSeek R1 via Ollama). If the LLM is absent/slow/fails it
falls back to a **heuristic entity-attribute resolver** (`sam/resolvers/`) so
the pipeline never hard-blocks. Supporting pieces: `turn_compressor` (condenses
long turns), `answer_generator`, `llm_reasoner`. SAM is the slow stage (LLM) —
`should_skip_sam()` bypasses it when the top hits are already confident.

### Assembler — `orchestration/assembler/`
`BudgetAssembler` fits the kept memories into the target's token budget, in
score order: try **full content** → `summary_medium` → `summary_short` → drop.
Pinned + most-recent-N items are protected (tried first, dropped last). Produces
an `AssembledContext`. It's decoupled from formatting — the Pipeline passes in
the empty-template overhead so the assembler knows the real remaining budget.

### Translator — `orchestration/translator/`
The **only** module that knows target-specific formatting. `Translator.translate`
looks up the adapter for `target.model_family` (gpt / claude / gemini / llama /
deepseek — `translator/adapters/` + Jinja templates in `translator/templates/`)
and renders the `AssembledContext` into a `TranslatedPayload` — the final string
(e.g. a `<context>…</context>` block) the target model receives, plus provenance.

### Tokenizers — `orchestration/tokenizers/`
Per-family token counting (tiktoken, claude, gemini, llama, deepseek, hf) so the
Assembler budgets accurately for the chosen target.

---

## 3. Memory format & ingestion

### AAL — `orchestration/aal/` — the canonical memory format
**AAL** = the persisted record shape. Every memory carries **two synchronized
views** of the same fact:
- **`simplemem`** — a one-line natural sentence ("We use Stripe for payments.").
  Embeds well; survives prose-style queries.
- **`sjson`** — a structured triple `{subject, verb, object, confidence,
  category, …}`. Wins on precise lookup and lets the reranker/SAM reason over
  discrete entities instead of parsing prose.

`AAL`, `AALBundle` (many AALs from one turn), `AALConverter` (raw/LLM-extracted
→ AAL). Both views are always produced; they're stored as the `aal_simplemem`
(TEXT) and `aal_sjson` (JSONB) columns.

### SDP — `orchestration/sdp/` — Semantic Decomposition Pipeline
The **fast, LLM-free ingest path** (regex + heuristics, ~<50 ms). It decomposes
a conversation turn into atomic facts. `SDPPipeline` chains:
```
raw turn → ConversationParser (normalize)
         → SemanticExtractor  (regex facts: tech stack, versions, ports, URLs…)
         → EntityExtractor     (entities) + RelationshipMapper
         → ImportanceScorer + ConfidenceScorer  (per-unit scores)
         → SemanticSummarizer  (per-unit summaries)
         → AALMemory[]          (canonical AAL objects)
```
Plus helpers: `date_extractor`/`date_resolver` (temporal), `dedup`, `linker`,
`hyde` (hypothetical-doc expansion), `query_router`/`query_decomposer`
(multi-hop), `entity_index`, `entity_synth`, `writer`. Catches only
pattern-detectable facts — paraphrased/implicit facts need the LLM path.

### Ingestion (LLM path) — `orchestration/ingestion/`
`MemoryExtractor` uses an LLM (Ollama) to extract durable facts from a turn —
catches nuanced/paraphrased facts SDP misses, at ~seconds/call. Output also
flows through `AALConverter` → AAL → MemoryItem.

So: **SDP = fast/regex, Ingestion = slow/LLM**, both end as AAL → MemoryItem,
embedded on write, persisted.

---

## 4. Storage — `orchestration/storage/` + `orchestration/memory_store/`
`MemoryStore` ABC with two backends, chosen by `GML_STORAGE_BACKEND`:
- **`JsonlMemoryStore`** — `~/.gml/memories.jsonl`. Single-tenant, in-memory
  after load. Dev/fallback.
- **`PostgresMemoryStore`** — pgvector, per-user RLS, byte-tracking quota.
  Production. **Embeds content on write** (computes the vector before insert),
  **auto-provisions the tenant** row on first write, and stores the AAL columns.

The factory (`storage/__init__.py`) builds the right store/retriever/user-store
for the backend and owns the pooled asyncpg connection (`register_vector` on
each connection so `vector(384)` works natively).

`PostgresUserKeyStore` (`storage/postgres_user_store.py`) handles users + API
keys + email/password auth; `orchestration/auth/` adds password hashing
(PBKDF2, stdlib) and JWT issue/verify.

---

## 5. Server surfaces

- **FastAPI** — `orchestration/server.py` (`gml serve` / `make_app` factory):
  `/api/*` (recall, ingest, sdp_ingest, memories CRUD, synthesize, trace,
  clusters, graph), `/auth/*` (signup/login/me, JWT), `/api/admin/keys`, `/viz`.
  An auth middleware resolves master key / API key / JWT → `request.state.user_id`.
- **MCP** — `orchestration/mcp_server.py` (`gml mcp`): exposes the same
  capabilities as MCP tools — `query`, `recall`, `remember`, `ingest`,
  `sdp_ingest`, `forget`, `list_memories`, `improve_query`, `status`, `analyze`,
  `trace`, `diag`. Runs over stdio (IDE/local) or streamable-http (remote).
- **CLI** — `orchestration/cli.py` (`gml ask | chat | serve | mcp | doctor`).

---

## 6. Cross-cutting

- **Contracts** — `orchestration/pipeline/contracts.py`: the Pydantic types that
  flow between stages (`Query`, `Classification`, `EmbeddedQuery`, `MemoryItem`,
  `RetrievalHit`, `RankedHit`, `ResolvedMemorySet`, `AssembledContext`,
  `TranslatedPayload`, `TargetDescriptor`). The stage boundaries are these types.
- **Clients** — `orchestration/clients/`: outbound LLM clients (anthropic,
  openai, gemini, ollama, stub) for the SAM reasoner / extractor / target calls.
- **Observability** — `orchestration/observability/`: structured logging +
  Prometheus-style metrics (`/metrics`).
- **Config** — `config/orchestration.toml` (top-k, weights, per-stage timeouts),
  loaded by `pipeline/config_loader.py`.

---

## 7. One-line glossary

| Term | What it is |
|---|---|
| **AAL** | Canonical memory record = `simplemem` (sentence) + `sjson` (triple) |
| **SDP** | Semantic Decomposition Pipeline — fast regex/heuristic ingest into AAL |
| **SAM** | Semantic Alignment Module — the conflict resolver (drops superseded memories) + from-scratch reasoning, LLM-backed with heuristic fallback |
| **Reranker** | Re-orders retrieved candidates by answer-relevance (cross-encoder ensemble in prod) |
| **Conflict resolver** | SAM's `resolve_conflicts` — supersession detection + query rewrite |
| **Translator** | Formats the assembled context for the specific target model family |
| **Assembler** | Fits memories into the token budget (full → summary → drop) |
| **Retriever** | Hybrid dense (pgvector) + sparse (BM25) fused via RRF |
| **Embedder** | text → 384-dim vector (FT sentence-transformers in prod) |
| **RLS** | Postgres Row-Level Security — per-user memory isolation |
| **MCP / relay** | How external tools reach the pipeline server-side (see the data-flow doc) |
