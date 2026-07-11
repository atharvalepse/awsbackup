"""``gml`` CLI — end-to-end interaction with the orchestration layer.

Two modes:

    gml ask "your question here"      # one-shot
    gml chat                          # multi-turn REPL

The default target is local DeepSeek R1 8B via Ollama (no API key needed).
Other targets read keys from env: ANTHROPIC_API_KEY, OPENAI_API_KEY,
GEMINI_API_KEY. The CLI auto-loads a ``.env`` file in the current
directory if present.
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Callable

from orchestration import __version__
from orchestration.classifier import KeywordClassifier
from orchestration.clients import (
    Client,
    StubClient,
    build_default_client_for_target,
)
from orchestration.embedder import FastEmbedEmbedder, GeminiEmbedder, OllamaEmbedder, StubEmbedder
from orchestration.embedder.base import Embedder
from orchestration.ingestion import MemoryExtractor
from orchestration.memory_store import JsonlMemoryStore
from orchestration.observability.logging import StructuredLogger
from orchestration.pipeline import Pipeline, TargetDescriptor, load_config
from orchestration.reranker import ScoreReranker, make_reranker
from orchestration.retriever import SemanticRetriever, StubRetriever, default_records
from orchestration.runner import Conversation
from orchestration.sam import SAM
from orchestration.sam._ollama_client import HTTPOllamaClient, make_local_llm_client
from orchestration.translator import Translator


slog = StructuredLogger("cli")


# ---------------------------------------------------------------------------
# .env loader (no python-dotenv dep — tiny inline parser)
# ---------------------------------------------------------------------------


def _load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE pairs from a ``.env`` file into ``os.environ``.

    Ignores blank lines and ``#`` comments. Does not overwrite existing
    env vars — explicit env wins over .env.
    """
    p = path or Path.cwd() / ".env"
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


# ---------------------------------------------------------------------------
# Target + Pipeline construction
# ---------------------------------------------------------------------------


_TARGET_BUILDERS: dict[str, Callable[[], TargetDescriptor]] = {
    "deepseek": TargetDescriptor.for_deepseek,
    "llama": lambda: TargetDescriptor.for_llama(model_version="llama3.2"),
    "claude": TargetDescriptor.for_claude,
    "gpt": TargetDescriptor.for_chatgpt,
    "gemini": TargetDescriptor.for_gemini,
}


def _resolve_target(name: str) -> TargetDescriptor:
    try:
        return _TARGET_BUILDERS[name]()
    except KeyError as exc:
        raise SystemExit(
            f"Unknown target {name!r}. Choices: {sorted(_TARGET_BUILDERS)}"
        ) from exc


def _default_config_path() -> Path:
    here = Path(__file__).resolve().parent.parent
    return here / "config" / "orchestration.toml"


def _ollama_has_model(model: str) -> bool:
    """Sync probe — checks if the given model is pulled into the local Ollama daemon."""
    import httpx
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=1.5)
        r.raise_for_status()
        names = {m.get("name", "").split(":")[0] for m in r.json().get("models", [])}
        return model.split(":")[0] in names
    except Exception:
        return False


def _autodetect_embedder(force: str | None = None) -> Embedder:
    """Pick the best available real Embedder.

    Order: explicit ``--embedder`` flag → FastEmbedEmbedder (in-process, no
    daemon, no API key, top of MTEB at its size) → OllamaEmbedder (if
    nomic-embed-text pulled) → GeminiEmbedder (if GEMINI_API_KEY set) →
    StubEmbedder with a stderr warning about non-semantic retrieval.
    """
    if force == "fastembed":
        sys.stderr.write("• Using FastEmbedEmbedder (BAAI/bge-small-en-v1.5, local ONNX)\n")
        return FastEmbedEmbedder()
    if force == "st":
        # Local sentence-transformers model — e.g. the FT'd embedder at
        # GML_ST_EMBED_MODEL. Needs torch + sentence-transformers installed.
        from orchestration.embedder import SentenceTransformerEmbedder
        device = os.environ.get("GML_ST_DEVICE", "cpu")
        emb = SentenceTransformerEmbedder(device=device)
        sys.stderr.write(f"• Using SentenceTransformerEmbedder ({emb.model_name}, {device})\n")
        return emb
    if force == "stub":
        sys.stderr.write(
            "⚠  Using StubEmbedder (hash-based, non-semantic). "
            "Retrieval similarity is essentially random.\n"
        )
        return StubEmbedder(dim=384)
    if force == "ollama":
        return OllamaEmbedder()
    if force == "gemini":
        return GeminiEmbedder()

    # Auto-detect — prefer FastEmbed (no daemon, no key, top quality).
    try:
        emb = FastEmbedEmbedder()
        sys.stderr.write("• Using FastEmbedEmbedder (BAAI/bge-small-en-v1.5, local ONNX)\n")
        return emb
    except Exception as exc:
        sys.stderr.write(f"• FastEmbedEmbedder unavailable ({type(exc).__name__}); trying alternatives\n")

    if _ollama_has_model("nomic-embed-text"):
        sys.stderr.write("• Using OllamaEmbedder (nomic-embed-text, local)\n")
        return OllamaEmbedder()
    if os.environ.get("GEMINI_API_KEY"):
        sys.stderr.write("• Using GeminiEmbedder (gemini-embedding-001, cloud)\n")
        return GeminiEmbedder()
    sys.stderr.write(
        "⚠  Falling back to StubEmbedder (hash-based, non-semantic). "
        "Retrieval similarity is essentially random.\n"
        "  → For real local retrieval: `pip install fastembed` or `ollama pull nomic-embed-text`\n"
        "  → For cloud retrieval: set GEMINI_API_KEY\n"
    )
    return StubEmbedder(dim=384)


def _build_pipeline(
    *, enable_sam_llm: bool, use_semantic_retriever: bool, embedder_choice: str | None,
) -> tuple[Pipeline, SemanticRetriever | StubRetriever, Embedder]:
    config = load_config(_default_config_path())

    embedder = _autodetect_embedder(force=embedder_choice)

    # Auto-promote to SemanticRetriever when we have a real embedder, so the
    # query vector lives in the same space as the record vectors. With
    # StubRetriever, records are embedded via hash-to-vector at init time —
    # if the query embedder is a real semantic one (FastEmbed, Gemini,
    # Ollama), the two spaces don't align and retrieval is essentially
    # random. The user can still force --semantic-retriever explicitly.
    use_semantic = use_semantic_retriever or not isinstance(embedder, StubEmbedder)
    retriever: SemanticRetriever | StubRetriever
    if use_semantic:
        retriever = SemanticRetriever(embedder=embedder)
    else:
        retriever = StubRetriever(dim=384)

    sam = SAM.with_ollama() if enable_sam_llm else SAM(reasoner=None)

    pipeline = Pipeline(
        classifier=KeywordClassifier(),
        embedder=embedder,
        retriever=retriever,
        reranker=make_reranker(config),
        sam=sam,
        translator=Translator(),
        config=config,
    )
    return pipeline, retriever, embedder


# ---------------------------------------------------------------------------
# Memory wiring
# ---------------------------------------------------------------------------


def _default_memory_path() -> Path:
    return Path.home() / ".gml" / "memories.jsonl"


async def _seed_retriever(retriever, store: JsonlMemoryStore) -> None:
    """Ingest seed records (the 8-item fixture if the store is empty, else
    the store's persisted records) into a SemanticRetriever."""
    if not isinstance(retriever, SemanticRetriever):
        return
    persisted = await store.load_all()
    records = persisted or default_records()
    await retriever.ingest(records)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


async def _run_ask(args) -> int:
    _load_dotenv()
    target = _resolve_target(args.target)

    enable_sam_llm = not args.no_sam_llm
    pipeline, retriever, _embedder = _build_pipeline(
        enable_sam_llm=enable_sam_llm,
        use_semantic_retriever=args.semantic_retriever,
        embedder_choice=args.embedder,
    )

    store = JsonlMemoryStore(args.memory_path or _default_memory_path())
    await _seed_retriever(retriever, store)

    client: Client = StubClient() if args.stub_client else build_default_client_for_target(target)

    extractor: MemoryExtractor | None = None
    if not args.no_extract:
        extractor = MemoryExtractor(client=make_local_llm_client())

    async def _ingest(items):
        if isinstance(retriever, SemanticRetriever):
            await retriever.ingest(items)

    conv = Conversation(
        pipeline=pipeline,
        client=client,
        target=target,
        extractor=extractor,
        memory_store=store,
        retriever_ingest=_ingest,
    )

    result = await conv.ask(args.text)
    print(result.response.text)

    if args.verbose:
        sys.stderr.write(f"\n--- target: {target.model_family.value}:{result.response.model_version} ---\n")
        sys.stderr.write(f"latency: {result.response.latency_ms}ms\n")
        sys.stderr.write(f"items_included: {result.payload.metadata.get('items_included')}\n")
        sys.stderr.write(f"query_was_improved: {result.payload.metadata.get('query_was_improved')}\n")
        sys.stderr.write(f"extracted_memories: {len(result.extracted_memories)}\n")
        for mem in result.extracted_memories:
            sys.stderr.write(f"  + {mem.id}: {mem.content!r}\n")

    return 0


async def _run_chat(args) -> int:
    _load_dotenv()
    target = _resolve_target(args.target)

    enable_sam_llm = not args.no_sam_llm
    pipeline, retriever, _embedder = _build_pipeline(
        enable_sam_llm=enable_sam_llm,
        use_semantic_retriever=args.semantic_retriever,
        embedder_choice=args.embedder,
    )

    store = JsonlMemoryStore(args.memory_path or _default_memory_path())
    await _seed_retriever(retriever, store)

    client = StubClient() if args.stub_client else build_default_client_for_target(target)
    extractor = None if args.no_extract else MemoryExtractor(client=make_local_llm_client())

    async def _ingest(items):
        if isinstance(retriever, SemanticRetriever):
            await retriever.ingest(items)

    conv = Conversation(
        pipeline=pipeline,
        client=client,
        target=target,
        extractor=extractor,
        memory_store=store,
        retriever_ingest=_ingest,
    )

    sys.stderr.write(f"gml chat — target={args.target}, session={conv.session.session_id[:8]}\n")
    sys.stderr.write("Type your message. Ctrl-D or 'exit' to quit.\n\n")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in {"exit", "quit"}:
            break
        result = await conv.ask(text)
        print(result.response.text)
        if args.verbose and result.extracted_memories:
            sys.stderr.write(f"[+{len(result.extracted_memories)} memories persisted]\n")
        print()
    return 0


# ---------------------------------------------------------------------------
# gml serve
# ---------------------------------------------------------------------------


async def _run_serve(args) -> int:
    _load_dotenv()
    embedder = _autodetect_embedder(force=args.embedder)
    memory_path = Path(args.memory_path) if args.memory_path else _default_memory_path()

    from orchestration.server import build_app, build_default_state

    state = await build_default_state(
        embedder=embedder,
        memory_path=memory_path,
        enable_sam_llm=not args.no_sam_llm,
        default_target_name=args.target,
        stub_client=args.stub_client,
    )
    app = build_app(state)

    import uvicorn
    cfg = uvicorn.Config(
        app, host=args.host, port=args.port, log_level=args.log_level
    )
    server = uvicorn.Server(cfg)
    sys.stderr.write(
        f"gml serve — http://{args.host}:{args.port}\n"
        f"  embedder: {embedder.version}\n"
        f"  default target: {args.target}\n"
        f"  memory: {memory_path}\n"
    )
    await server.serve()
    return 0


# ---------------------------------------------------------------------------
# gml doctor — checks setup and tells the user what's missing
# ---------------------------------------------------------------------------


def _run_doctor(args) -> int:
    _load_dotenv()
    out = sys.stdout

    OK = "\033[32m✓\033[0m"
    NO = "\033[31m✗\033[0m"
    WARN = "\033[33m⚠\033[0m"

    out.write("\nGML setup check\n")
    out.write("=" * 60 + "\n\n")

    # 1. Python install + package version
    out.write(f"{OK} gml version: {__version__}\n")

    # 2. Memory file
    mem_path = _default_memory_path()
    if mem_path.exists():
        n = sum(1 for _ in mem_path.open()) if mem_path.stat().st_size > 0 else 0
        out.write(f"{OK} memory file: {mem_path} ({n} memories saved)\n")
    else:
        out.write(f"{WARN} memory file: {mem_path} (will be created on first ask)\n")

    out.write("\n--- Cloud AI API keys ---\n")
    for var, target_name in [
        ("ANTHROPIC_API_KEY", "Claude"),
        ("OPENAI_API_KEY", "GPT / ChatGPT"),
        ("GEMINI_API_KEY", "Gemini"),
    ]:
        val = os.environ.get(var)
        if val:
            out.write(f"{OK} {var} set ({len(val)} chars) → can use --target {target_name.lower().split()[0]}\n")
        else:
            out.write(f"{NO} {var} not set → --target {target_name.lower().split()[0]} won't work\n")

    out.write("\n--- Local AI (Ollama) ---\n")
    import httpx
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        r.raise_for_status()
        models = r.json().get("models", [])
        names = [m.get("name", "?") for m in models]
        out.write(f"{OK} Ollama daemon is running ({len(names)} models pulled)\n")
        for n in names:
            base = n.split(":")[0]
            if base in ("deepseek-r1", "llama3.2", "llama3", "nomic-embed-text"):
                out.write(f"  • {n} (recognized)\n")
            else:
                out.write(f"  • {n}\n")
        # Check key models
        bases = {n.split(":")[0] for n in names}
        if "deepseek-r1" in bases:
            out.write(f"{OK} deepseek-r1 pulled → --target deepseek will work\n")
        else:
            out.write(f"{NO} deepseek-r1 not pulled → run: ollama pull deepseek-r1:8b\n")
        if "nomic-embed-text" in bases:
            out.write(f"{OK} nomic-embed-text pulled → --embedder ollama will work\n")
        else:
            out.write(f"{WARN} nomic-embed-text not pulled → fastembed will be used instead (still good)\n")
    except Exception:
        out.write(f"{NO} Ollama daemon not reachable at localhost:11434\n")
        out.write("   Start it with: ollama serve  (or open Ollama.app)\n")
        out.write("   Then: ollama pull deepseek-r1:8b\n")

    out.write("\n--- Local embedder (FastEmbed) ---\n")
    try:
        from orchestration.embedder import FastEmbedEmbedder
        FastEmbedEmbedder()
        out.write(f"{OK} FastEmbed model cached and ready\n")
    except Exception as exc:
        out.write(f"{WARN} FastEmbed model not yet downloaded ({type(exc).__name__})\n")
        out.write("   First use will download ~130 MB (one-time, automatic)\n")

    out.write("\n--- Suggested next step ---\n")
    has_any_cloud = any(os.environ.get(v) for v in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"])
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=1.0)
        ollama_ok = r.status_code == 200
    except Exception:
        ollama_ok = False

    if os.environ.get("GEMINI_API_KEY"):
        out.write("• Try the fastest cloud option (Gemini, ~3s per answer):\n")
        out.write('    gml ask "how is auth implemented?" --target gemini -v\n')
    if os.environ.get("ANTHROPIC_API_KEY"):
        out.write("• Try Claude:\n")
        out.write('    gml ask "how is auth implemented?" --target claude -v\n')
    if os.environ.get("OPENAI_API_KEY"):
        out.write("• Try GPT:\n")
        out.write('    gml ask "how is auth implemented?" --target gpt -v\n')
    if ollama_ok:
        out.write("• Try local DeepSeek (~30-60s per answer, no keys needed):\n")
        out.write('    gml ask "how is auth implemented?" --target deepseek -v\n')
    if not has_any_cloud and not ollama_ok:
        out.write(f"{NO} No usable target found. Either:\n")
        out.write("   1. Set a cloud key (ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY)\n")
        out.write("      in .env or your shell, then run: gml doctor\n")
        out.write("   2. Or install Ollama from https://ollama.com and run:\n")
        out.write("        ollama pull deepseek-r1:8b\n")

    out.write("\n")
    return 0


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gml", description="GML orchestration CLI")
    p.add_argument("--version", action="version", version=f"gml {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def _common_args(sp):
        sp.add_argument(
            "--target", default="deepseek",
            choices=sorted(_TARGET_BUILDERS),
            help="Target AI family (default: deepseek)",
        )
        sp.add_argument("--memory-path", help="Path to memories.jsonl (default: ~/.gml/memories.jsonl)")
        sp.add_argument("--no-sam-llm", action="store_true", help="Disable SAM LLM reasoner (heuristic only)")
        sp.add_argument("--no-extract", action="store_true", help="Disable memory extraction")
        sp.add_argument("--stub-client", action="store_true", help="Use StubClient (no real LLM call)")
        sp.add_argument(
            "--semantic-retriever", action="store_true",
            help="Use SemanticRetriever (loads from memory store) instead of stub fixture",
        )
        sp.add_argument(
            "--embedder", choices=["auto", "fastembed", "st", "ollama", "gemini", "stub"], default="auto",
            help=(
                "Embedder backend. 'auto' picks FastEmbed → Ollama → Gemini → Stub. "
                "Default: auto."
            ),
        )
        sp.add_argument("-v", "--verbose", action="store_true")

    p_ask = sub.add_parser("ask", help="One-shot ask")
    p_ask.add_argument("text", help="Question to send to the target AI")
    _common_args(p_ask)

    p_chat = sub.add_parser("chat", help="Multi-turn REPL")
    _common_args(p_chat)

    p_serve = sub.add_parser("serve", help="Run the HTTP API server (FastAPI)")
    _common_args(p_serve)
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--log-level", default="info",
                         choices=["critical", "error", "warning", "info", "debug", "trace"])

    sub.add_parser("doctor", help="Check setup and tell you what's missing")

    sub.add_parser(
        "mcp",
        help=(
            "Run the MCP server over stdio so Claude Desktop / Cursor / "
            "VS Code can call GML as a memory tool"
        ),
    )

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "ask":
        return asyncio.run(_run_ask(args))
    if args.cmd == "chat":
        return asyncio.run(_run_chat(args))
    if args.cmd == "serve":
        return asyncio.run(_run_serve(args))
    if args.cmd == "doctor":
        return _run_doctor(args)
    if args.cmd == "mcp":
        # Proxy mode keeps each stdio child tiny (~30 MB): tool calls are
        # forwarded to the running HTTP API instead of loading the embedder
        # and rerankers per process.
        transport = os.environ.get("GML_MCP_TRANSPORT", "stdio").strip().lower()
        if transport == "stdio" and os.environ.get("GML_MCP_PROXY_URL", "").strip():
            from orchestration.mcp_proxy import run as proxy_run
            proxy_run()
            return 0
        from orchestration.mcp_server import run as mcp_run
        mcp_run()
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
