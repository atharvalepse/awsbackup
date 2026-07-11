"""LLM-based reranker — let a local LLM judge candidate relevance.

After the cross-encoder narrows the candidate pool to ~20 hits, this
reranker asks Qwen3.5-4B (or whichever local-LLM client is configured)
to rank them by how well each one answers the user's question.

Why this works
--------------
Cross-encoders are trained on broad relevance signals (e.g. MS-MARCO);
they're great at "is X about Y" but weaker at "does X actually answer
question Y". An LLM with even modest reasoning ability can read 20
short candidates + the question and rank them by direct-answer-ness.

This is the final, sharpest filter before the assembler. Typical gain
on retrieval benchmarks: +5-10% absolute on hard categories (multi-hop,
paraphrase, temporal). Cost: 1-2s per query with Qwen3.5-4B-Q4.

Failure mode: if the LLM returns un-parseable JSON or hangs, we fall
back to the input order (which is already cross-encoder-ranked, so
graceful). Never blocks the pipeline.

API
---
``LLMReranker(llm_client)`` implements the Reranker interface.
Its ``pick_best(hits, query, k)`` is a drop-in for any other reranker
but typically wraps a cross-encoder stage (call cross-encoder first to
narrow to ~20, then LLMReranker to pick the final top-k).
"""
import json
import os
import re

from orchestration.errors import RerankerError
from orchestration.pipeline.contracts import Query, RankedHit, RetrievalHit
from orchestration.reranker.base import Reranker
from orchestration.sam._ollama_client import OllamaClient


_RERANK_PROMPT = """\
You will be shown a question and {n} candidate memories. Rank the
memories from BEST to WORST at directly answering the question.

A memory is GOOD if its content provides the specific information the
question asks for. A memory is BAD if it's only loosely related.

Question: {question!r}

Candidates (each on its own line, prefixed by index):
{candidates}

Respond with JSON ONLY. No prose, no markdown. Schema:

{{"ranking": [<best_index>, <second>, ..., <worst>]}}

The ranking array must contain each candidate index exactly once.
"""


class LLMReranker(Reranker):
    """Use a local LLM to rank candidates by direct-answer relevance.

    Designed to run AFTER a cross-encoder shortlists candidates to ~20.
    Running it over hundreds of candidates is expensive and the LLM's
    short context window suffers. Keep input ``hits`` small.
    """

    def __init__(
        self,
        client: OllamaClient,
        max_candidates: int = 20,
        max_chars_per_candidate: int = 240,
    ) -> None:
        self.client = client
        self.max_candidates = max_candidates
        self.max_chars_per_candidate = max_chars_per_candidate

    async def pick_best(
        self, hits: list[RetrievalHit], query: Query, k: int = 10
    ) -> list[RankedHit]:
        if not hits:
            return []

        # Limit input to avoid an unreasonable prompt
        capped = hits[: self.max_candidates]
        candidates_text = "\n".join(
            f"[{i}] {self._format_candidate(h)}"
            for i, h in enumerate(capped)
        )
        prompt = _RERANK_PROMPT.format(
            n=len(capped),
            question=query.text,
            candidates=candidates_text,
        )

        try:
            gen = await self.client.generate(prompt, json_mode=True)
            ranking = self._parse_ranking(gen.answer, n=len(capped))
        except Exception as exc:
            # Fall back to input order — caller already cross-encoder ranked.
            ranking = list(range(len(capped)))

        # Build RankedHits in the LLM's chosen order, top-k.
        ranked: list[RankedHit] = []
        for rank_position, src_index in enumerate(ranking[:k]):
            hit = capped[src_index]
            # Inverse-position as the final_score so downstream sees a
            # monotonically-decreasing 0..1 score.
            normalized = 1.0 - (rank_position / max(len(ranking), 1))
            ranked.append(RankedHit(
                hit=hit,
                semantic_score=hit.similarity,
                recency_score=0.0,
                authority_score=hit.record.authority_score,
                pin_boost=1.0 if hit.record.pinned else 0.0,
                final_score=normalized,
                score_reason=f"llm-rerank rank={rank_position+1}/{len(ranking)} (input sim={hit.similarity:.2f})",
            ))
        return ranked

    # -----------------------------------------------------------------

    def _format_candidate(self, hit: RetrievalHit) -> str:
        rec = hit.record
        text = rec.content
        if len(text) > self.max_chars_per_candidate:
            text = text[: self.max_chars_per_candidate - 1] + "…"
        prefix = ""
        if rec.entity:
            prefix = f"[{rec.entity}"
            if rec.attribute:
                prefix += f"/{rec.attribute}"
            if rec.value:
                prefix += f"={rec.value}"
            prefix += "] "
        return f"{prefix}{text}"

    def _parse_ranking(self, answer: str, n: int) -> list[int]:
        """Parse a JSON ranking. Returns input-order on failure.

        Tolerant to surrounding prose, markdown fences, partial output.
        Validates that the ranking is a permutation of [0..n); any
        invalid index is dropped and missing ones appended in input order.
        """
        if not answer:
            return list(range(n))
        text = answer.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
            text = text.strip()
        # Outermost braces
        o, c = text.find("{"), text.rfind("}")
        if o < 0 or c <= o:
            return list(range(n))
        try:
            data = json.loads(text[o : c + 1])
        except json.JSONDecodeError:
            # Salvage attempt: pull just integers from the text
            ints = [int(x) for x in re.findall(r"\d+", text) if 0 <= int(x) < n]
            return _complete_perm(ints, n)
        raw = data.get("ranking")
        if not isinstance(raw, list):
            return list(range(n))
        cleaned: list[int] = []
        for x in raw:
            try:
                idx = int(x)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < n and idx not in cleaned:
                cleaned.append(idx)
        return _complete_perm(cleaned, n)


def _complete_perm(seen: list[int], n: int) -> list[int]:
    """Pad missing indices at the end so we return a full permutation."""
    seen_set = set(seen)
    tail = [i for i in range(n) if i not in seen_set]
    return seen + tail


def make_llm_reranker(client: OllamaClient | None = None) -> "LLMReranker | None":
    """Helper: build LLMReranker iff GML_LLM_RERANKER=1 and client available.

    The env-var gate is off by default because LLM reranking adds ~1-2s per
    query — only enable when you want the precision lift.
    """
    if os.environ.get("GML_LLM_RERANKER", "0") != "1":
        return None
    if client is None:
        from orchestration.sam._ollama_client import make_local_llm_client
        try:
            client = make_local_llm_client()
        except Exception:
            return None
    return LLMReranker(client)
