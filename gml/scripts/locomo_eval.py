"""LOCOMO benchmark eval harness for the GML memory layer.

LOCOMO (LOng COnversation MemOry) is the canonical benchmark for evaluating
long-term conversational memory systems. The dataset is multi-session
dialogues with QA pairs that probe single-hop, multi-hop, temporal, and
open-domain recall.

Usage:

    # Download the LOCOMO dataset first (see paper for instructions); the
    # expected layout is one JSON file per conversation, e.g.
    #     locomo_data/conv_001.json
    #     locomo_data/conv_002.json
    # ...with the structure described in load_conversation() below.

    python scripts/locomo_eval.py \\
        --data-dir locomo_data \\
        --target deepseek \\
        --max-conversations 5

This script intentionally builds a fresh memory store per conversation so
the eval mirrors LOCOMO's per-conversation memory budget — what the
system learns inside one conversation must be enough to answer the QA
pairs at the end.

Metrics:
  - Exact-match accuracy
  - F1 (token-level overlap with reference answer)
  - Latency per question
  - Per-category breakdown when the dataset provides question categories

Notes:
  - LOCOMO answer formats vary; the F1/EM scorers below are deliberately
    lenient (lowercase + strip punctuation + token overlap). Tune per the
    paper's official scorer if comparing against a published number.
  - This harness uses the FULL pipeline (hybrid retrieval + cross-encoder
    + SAM + DeepSeek target). Configure via flags below for ablations.
"""
import argparse
import asyncio
import json
import re
import string
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orchestration.classifier import KeywordClassifier
from orchestration.clients import StubClient, build_default_client_for_target
from orchestration.embedder import FastEmbedEmbedder
from orchestration.ingestion import MemoryExtractor
from orchestration.memory_store import JsonlMemoryStore
from orchestration.pipeline import MemoryItem, Pipeline, TargetDescriptor, load_config
from orchestration.reranker import ScoreReranker
from orchestration.retriever import BM25Retriever, HybridRetriever, SemanticRetriever
from orchestration.runner import Conversation
from orchestration.sam import SAM
from orchestration.sam._ollama_client import HTTPOllamaClient
from orchestration.translator import Translator


# ---------------------------------------------------------------------------
# Dataset loader — flexible to LOCOMO's actual format
# ---------------------------------------------------------------------------


def load_conversation(path: Path) -> dict[str, Any]:
    """Load a single LOCOMO conversation file.

    Expected schema (the actual LOCOMO format varies by release — adjust
    here if your local copy uses different keys):

    {
      "conversation_id": "...",
      "sessions": [
        {
          "session_id": "...",
          "turns": [
            {"speaker": "user", "text": "..."},
            {"speaker": "assistant", "text": "..."},
            ...
          ]
        }, ...
      ],
      "qa": [
        {
          "question": "...",
          "answer": "...",
          "category": "single_hop" | "multi_hop" | "temporal" | "open_domain"
        }, ...
      ]
    }
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_conversations(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(
            f"LOCOMO data dir {data_dir} not found. Download the dataset "
            "first (see https://memorybenchmark.com or the LOCOMO paper) "
            "and point --data-dir at it."
        )
    files = sorted(data_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No .json files in {data_dir}")
    return files


# ---------------------------------------------------------------------------
# Scorers — leniency matches typical LOCOMO scoring conventions
# ---------------------------------------------------------------------------


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def f1_score(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common = Counter(pred_tokens) & Counter(ref_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall = n_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_text(prediction) == normalize_text(reference))


# ---------------------------------------------------------------------------
# Per-conversation eval driver
# ---------------------------------------------------------------------------


async def _build_pipeline(use_cross_encoder: bool, sam_llm: bool):
    config = load_config(Path(__file__).resolve().parent.parent / "config" / "orchestration.toml")
    embedder = FastEmbedEmbedder()
    dense = SemanticRetriever(embedder=embedder)
    sparse = BM25Retriever()
    retriever = HybridRetriever(dense=dense, sparse=sparse)

    reranker = ScoreReranker(config)
    sam = SAM.with_ollama() if sam_llm else SAM(reasoner=None)

    pipeline = Pipeline(
        classifier=KeywordClassifier(),
        embedder=embedder,
        retriever=retriever,
        reranker=reranker,
        sam=sam,
        translator=Translator(),
        config=config,
    )
    return pipeline, retriever, embedder


async def ingest_conversation_history(
    retriever: HybridRetriever, conv_data: dict
) -> int:
    """Turn the dialogue history into MemoryItem records and ingest. Returns
    the number of records ingested.

    We use a simple chunking scheme: each user→assistant turn becomes one
    MemoryItem. This is a reasonable default; more sophisticated schemes
    (segmentation by topic, summarization, entity extraction) would lift
    LOCOMO scores further.
    """
    items: list[MemoryItem] = []
    now = datetime.now(timezone.utc)
    for sess_idx, session in enumerate(conv_data.get("sessions", [])):
        for turn_idx, turn in enumerate(session.get("turns", [])):
            if not turn.get("text"):
                continue
            items.append(MemoryItem(
                id=f"locomo-s{sess_idx}-t{turn_idx}",
                content=f"{turn.get('speaker', 'unknown')}: {turn['text']}",
                timestamp=now,
                source=f"locomo:session_{sess_idx}",
                authority_score=0.7,
            ))
    if items:
        await retriever.ingest(items)
    return len(items)


async def eval_conversation(
    conv_data: dict,
    target_name: str,
    sam_llm: bool,
    stub_client: bool,
) -> dict[str, Any]:
    pipeline, retriever, _embedder = await _build_pipeline(
        use_cross_encoder=False, sam_llm=sam_llm
    )
    target = {
        "deepseek": TargetDescriptor.for_deepseek,
        "claude": TargetDescriptor.for_claude,
        "gpt": TargetDescriptor.for_chatgpt,
        "gemini": TargetDescriptor.for_gemini,
        "llama": lambda: TargetDescriptor.for_llama(model_version="llama3.2"),
    }[target_name]()

    n_ingested = await ingest_conversation_history(retriever, conv_data)

    client = StubClient() if stub_client else build_default_client_for_target(target)

    qa_results: list[dict[str, Any]] = []
    for qa in conv_data.get("qa", []):
        question = qa.get("question") or ""
        reference = qa.get("answer") or ""
        category = qa.get("category") or "unknown"
        if not question or not reference:
            continue

        conv = Conversation(
            pipeline=pipeline,
            client=client,
            target=target,
            extractor=None,             # eval doesn't grow memory
            memory_store=None,
        )

        t0 = time.perf_counter()
        try:
            result = await conv.ask(question)
            prediction = result.response.text
            error = None
        except Exception as exc:
            prediction = ""
            error = f"{type(exc).__name__}: {exc}"
        dt = time.perf_counter() - t0

        qa_results.append({
            "category": category,
            "question": question,
            "reference": reference,
            "prediction": prediction,
            "f1": f1_score(prediction, reference),
            "em": exact_match(prediction, reference),
            "latency_s": dt,
            "error": error,
        })

    return {
        "conversation_id": conv_data.get("conversation_id"),
        "records_ingested": n_ingested,
        "qa_results": qa_results,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    all_qa: list[dict[str, Any]] = []
    for r in results:
        all_qa.extend(r["qa_results"])
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for qa in all_qa:
        by_cat[qa["category"]].append(qa)

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    summary = {
        "n_conversations": len(results),
        "n_questions": len(all_qa),
        "overall_f1": _mean([qa["f1"] for qa in all_qa]),
        "overall_em": _mean([qa["em"] for qa in all_qa]),
        "median_latency_s": sorted([qa["latency_s"] for qa in all_qa])[len(all_qa) // 2] if all_qa else 0.0,
        "by_category": {
            cat: {
                "n": len(qa_list),
                "f1": _mean([qa["f1"] for qa in qa_list]),
                "em": _mean([qa["em"] for qa in qa_list]),
            }
            for cat, qa_list in by_cat.items()
        },
    }
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def main_async(args) -> int:
    data_dir = Path(args.data_dir)
    convs = discover_conversations(data_dir)
    if args.max_conversations:
        convs = convs[: args.max_conversations]

    sys.stderr.write(f"Running LOCOMO eval on {len(convs)} conversation(s)\n")

    results = []
    for i, p in enumerate(convs, start=1):
        sys.stderr.write(f"\n[{i}/{len(convs)}] {p.name}\n")
        conv_data = load_conversation(p)
        r = await eval_conversation(
            conv_data=conv_data,
            target_name=args.target,
            sam_llm=not args.no_sam_llm,
            stub_client=args.stub_client,
        )
        r["file"] = str(p)
        results.append(r)
        qas = r["qa_results"]
        f1s = sum(qa["f1"] for qa in qas) / len(qas) if qas else 0.0
        sys.stderr.write(f"  → {len(qas)} questions, mean F1 {f1s:.3f}\n")

    summary = aggregate(results)
    print(json.dumps({"summary": summary, "results": results}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="LOCOMO eval harness for GML")
    p.add_argument("--data-dir", required=True, help="Directory of LOCOMO conversation JSON files")
    p.add_argument("--target", default="deepseek",
                   choices=["deepseek", "llama", "claude", "gpt", "gemini"])
    p.add_argument("--max-conversations", type=int, default=0,
                   help="Limit to first N conversations (0 = all)")
    p.add_argument("--no-sam-llm", action="store_true")
    p.add_argument("--stub-client", action="store_true",
                   help="Use StubClient (e.g. for plumbing checks)")
    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
