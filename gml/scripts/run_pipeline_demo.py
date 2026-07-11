"""Run the FULL orchestration pipeline against the live Postgres/pgvector DB.

Unlike `gml ask` (which hardcodes the JSONL in-memory retriever), this wires
the pipeline through the storage factory so retrieval hits real pgvector.
Scoped to user 'demo' (the seeded rows). No LLM needed — SAM runs heuristic.

    GML_STORAGE_BACKEND=postgres \
    GML_DATABASE_URL='postgresql://gml_app:PASS@127.0.0.1:5432/gml_test' \
        .venv/bin/python scripts/run_pipeline_demo.py "what do we use for payments?"
"""
import asyncio
import sys
import uuid

from orchestration.cli import _default_config_path
from orchestration.classifier.keyword_classifier import KeywordClassifier
from orchestration.embedder import FastEmbedEmbedder
from orchestration.pipeline.config_loader import load_config
from orchestration.pipeline.contracts import Query, TargetDescriptor
from orchestration.pipeline.pipeline import Pipeline
from orchestration.reranker import make_reranker
from orchestration.sam.sam import SAM
from orchestration.storage import close_pg_pool, make_hybrid_retriever
from orchestration.translator.translator import Translator

B, G, C, X = "\033[1m", "\033[32m", "\033[36m", "\033[0m"
USER = "demo"


async def main() -> int:
    query_text = sys.argv[1] if len(sys.argv) > 1 else "what do we use for payments?"
    config = load_config(_default_config_path())
    embedder = FastEmbedEmbedder()

    # Postgres-backed dense(pgvector)+sparse(bm25) retriever via the factory.
    retriever = await make_hybrid_retriever(embedder)
    print(f"{B}retriever:{X} {type(retriever).__name__} "
          f"(dense={type(retriever.dense).__name__}, sparse={type(retriever.sparse).__name__})")

    pipeline = Pipeline(
        classifier=KeywordClassifier(),
        embedder=embedder,
        retriever=retriever,
        reranker=make_reranker(config),
        sam=SAM(reasoner=None),          # heuristic SAM — no LLM call
        translator=Translator(),
        config=config,
    )

    query = Query(
        text=query_text,
        target=TargetDescriptor.for_claude(),
        trace_id=uuid.uuid4().hex,
        user_id=USER,                    # RLS scopes pgvector to this tenant
    )

    print(f"\n{B}QUERY{X} (user={USER!r}): {C}{query_text}{X}")
    print("=" * 70)

    # --- Show the raw pgvector retrieval the pipeline will consume ---------
    classification = await pipeline._stage_classifier(query)
    embedded = await embedder.embed(query, classification)
    hits = await retriever.dense.get_top_matches(embedded, k=5)
    print(f"\n{B}[pgvector dense hits]{X}  (cosine similarity)")
    if not hits:
        print("  (none — pgvector returned nothing)")
    for h in hits:
        print(f"  {G}{h.similarity:.3f}{X}  {h.record.content}")

    # --- Run the FULL pipeline end-to-end ----------------------------------
    payload = await pipeline.run(query)
    print(f"\n{B}[assembled + translated payload]{X}")
    print(f"  items_included : {payload.metadata.get('items_included')}")
    print(f"  trace_id       : {payload.trace_id}")
    print(f"  target         : {payload.target.model_family.value}:{payload.target.model_version}")
    print(f"\n{B}--- formatted_context (what the target model receives) ---{X}")
    print(payload.formatted_context)

    await close_pg_pool()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
