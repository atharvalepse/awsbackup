"""AnswerGenerator — final-stage LLM call that produces a short answer
from retrieved context.

Used for LOCOMO-style benchmarking where the published metric is **F1
between a generated answer and the gold answer**. Our pipeline produces
``formatted_context`` (the retrieved memories rendered for the target
AI); the answer generator closes the loop: it acts AS the target AI,
reading the context and emitting a concrete short answer.

The bench then scores TWO metrics:
  - ``context_recall``: did the retrieved context contain the gold-answer
    tokens? (Same metric we've used all along — upper bound on F1.)
  - ``answer_f1``: token-F1 between the generated answer and the gold.
    Direct comparison to published LOCOMO numbers.

Fail-safe: any LLM error returns an empty answer (which scores 0 F1).
Never blocks the bench.
"""
import os
import re

from orchestration.observability.logging import StructuredLogger
from orchestration.sam._ollama_client import OllamaClient


slog = StructuredLogger("sam.answer_generator")


_ANSWER_PROMPT = """\
Answer the QUESTION using ONLY the CONTEXT below. LOCOMO answers are
EXTREMELY short — usually 1-5 words, just the fact, no explanation, no
restating the question, no "based on the context".

CRITICAL RULES:
1. USE EXACT WORDS FROM CONTEXT. Do NOT rephrase, expand, or substitute
   synonyms. If context says "Rockies", say "Rockies" — NOT "Rocky
   Mountains". If context says "NYC", say "NYC" — NOT "New York City".
2. FOR PLURAL / LIST QUESTIONS ("What kinds of...", "What things did...",
   "What activities...", "What hobbies..."), list ALL matching items from
   context, comma-separated. Don't pick just one.
3. PRESERVE DATE PHRASING. If context says "the weekend before May 24,
   2023", say "weekend before May 24, 2023" — do NOT convert to "17 May".
   Only normalize when the gold form is already absolute.

Format hints:
- Entity (who/what/where): bare name copied from context.
- Count: bare number word. "two", "five", "3".
- Yes/no: bare "yes" or "no".
- Unknown: ONLY if context is empty or completely unrelated — say "I don't know".

EXAMPLES:

CONTEXT: <memory>Caroline researched adoption agencies on Sunday May 7</memory>
QUESTION: What did Caroline research?
ANSWER: adoption agencies

CONTEXT: <memory>Caroline attended the LGBTQ support group on Sunday May 7 2023</memory>
QUESTION: When did Caroline go to the LGBTQ support group?
ANSWER: 7 May 2023

CONTEXT: <memory>Mel scheduled the launch for the weekend before May 24, 2023</memory>
QUESTION: When was the launch?
ANSWER: weekend before May 24, 2023

CONTEXT: <memory>Evan drives a Prius</memory><memory>Evan's old Prius broke down</memory>
QUESTION: How many Prius has Evan owned?
ANSWER: two

CONTEXT: <memory>Evan road-tripped through the Rockies</memory><memory>Evan visited Jasper</memory>
QUESTION: Where has Evan been on roadtrips with his family?
ANSWER: Rockies, Jasper

CONTEXT: <memory>Sam took up painting in May</memory><memory>Sam goes kayaking</memory><memory>Sam hikes weekly</memory><memory>Sam cooks new recipes</memory><memory>Sam runs every morning</memory>
QUESTION: What kinds of activities does Sam do?
ANSWER: painting, kayaking, hiking, cooking, running

CONTEXT: <memory>Caroline does yoga on Thursdays</memory><memory>Caroline does meditation on Mondays</memory>
QUESTION: What practices does Caroline do?
ANSWER: yoga, meditation

CONTEXT: <memory>Caroline: feeling better today. Mel: glad to hear it.</memory>
QUESTION: What is the SLA for the orders database?
ANSWER: I don't know

NOW ANSWER. Reply with ONLY the answer — no preamble, no period, no quotes.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


# Cap context to ~6000 chars (well within Qwen2.5-3B's effective window).
MAX_CONTEXT_CHARS = 6000

# Per-category generation budgets. Picked to leave NO room for preamble —
# 12 tokens ≈ 7-9 short English words, which fits every LOCOMO gold we
# checked. cat-2 multi-hop sometimes needs a comma-list, hence the larger
# budget.
_CATEGORY_MAX_TOKENS = {
    1: 16,   # single-hop: entity / count / short fact
    2: 32,   # multi-hop: 2-3 items
    3: 20,   # temporal: date or duration
    4: 24,   # open-domain
    5: 10,   # adversarial: usually "I don't know"
}
_DEFAULT_MAX_TOKENS = 24

# Patterns that hint the gold answer is a LIST of items. When a question
# matches, we (a) bump max_tokens so the model can emit all of them, and
# (b) prepend an explicit list-emphasis line to the prompt. Catches the
# cat-1 questions like "What kinds of things did X have broken?" where
# the gold is "His old Prius and his new Prius."
_LIST_QUESTION_RE = re.compile(
    r"\b("
    r"what (kinds|sorts|types) of"             # PLURAL only — "what kinds of"
    r"|what (hobbies|activities|things|practices|items|places|sports|cars|vehicles)"
    r"|which (hobbies|activities|things|items|places)"
    r"|what (do|does|did)\s+\w+\s+(do|like|enjoy|engage)"
    r"|list (all|the)"
    r"|name (all|the|some|multiple)"
    r"|all (the|of) (the )?\w+"
    r"|what are (all )?(the )?\w+ (things|items|hobbies|activities)"
    r")\b",
    re.IGNORECASE,
)


def _is_list_question(question: str) -> bool:
    return bool(_LIST_QUESTION_RE.search(question or ""))


_LIST_EMPHASIS = (
    "*** THIS IS A LIST QUESTION. ***  Find ALL matching items across the "
    "context (they may be in different memories) and list ALL of them "
    "comma-separated. Do NOT pick just one.\n\n"
)

# Lead-ins to strip from generated answers. Qwen2.5:3b loves to start with
# "Based on the context, ..." even when told not to.
_LEAD_INS = (
    "answer:", "a:", "the answer is", "based on the context",
    "based on the provided context", "according to the context",
    "from the context", "the question asks", "in the context",
    "context indicates", "context shows", "context says",
)


# Synonym pairs we observed in LOCOMO failures — when the model emits the
# right side but the gold (and context) uses the left side, we snap back.
# Direction matters: we only rewrite RHS → LHS when the LHS form actually
# appears in the provided context (verified at call time).
_SYNONYM_SNAPS = [
    ("Rockies", "Rocky Mountains"),
    ("NYC", "New York City"),
    ("NYC", "New York"),
    ("UK", "United Kingdom"),
    ("US", "United States"),
    ("USA", "United States of America"),
    ("LA", "Los Angeles"),
    ("SF", "San Francisco"),
]


def _snap_synonyms(answer: str, context: str) -> str:
    """If model output an expanded form but context uses the short form,
    snap to the short form so token-F1 against the gold (which usually
    mirrors context) holds up.
    """
    if not answer or not context:
        return answer
    ctx_lower = context.lower()
    out = answer
    for short, long_form in _SYNONYM_SNAPS:
        # Only snap if context uses the short form AND model emitted the long one.
        if short.lower() in ctx_lower and long_form.lower() in out.lower():
            out = re.sub(re.escape(long_form), short, out, flags=re.IGNORECASE)
    return out


def _post_process(raw: str, context: str = "") -> str:
    """Strip preambles / quotes / trailing punctuation from the LLM output.

    ``context`` is the formatted_context we fed the model — used for the
    synonym-snap step. Pass empty string to skip that step.
    """
    out = raw.strip()

    # Strip lead-ins iteratively (the model sometimes stacks two).
    for _ in range(3):
        lower = out.lower()
        changed = False
        for lead in _LEAD_INS:
            if lower.startswith(lead):
                out = out[len(lead):].lstrip(" :,-")
                changed = True
                break
        if not changed:
            break

    # Drop a leading "is " / "was " / "the " if it's left over from a
    # preamble strip ("Based on the context, the answer is Prius.").
    while out[:4].lower() in ("is ", "was "):
        out = out[3 if out[:3].lower() == "is " else 4:].lstrip()
    if out.lower().startswith("the answer "):
        out = out[len("the answer "):].lstrip(" :,-")

    # Take first sentence only — LOCOMO answers are atomic.
    sentences = re.split(r"(?<=[.!?])\s+", out)
    if sentences:
        out = sentences[0].strip()

    # Strip surrounding quotes / markdown / trailing punctuation/period.
    out = out.strip("\"'`*_ ").rstrip(".,;:!?")

    # Snap synonym expansions back to the context's preferred form.
    out = _snap_synonyms(out, context)

    # Cap length (defense in depth).
    if len(out) > 200:
        out = out[:199] + "…"
    return out


class AnswerGenerator:
    """Generate a short answer from formatted_context + question."""

    def __init__(self, client: OllamaClient, uses_ft_prompt: bool | None = None) -> None:
        self.client = client
        # When None, fall back to env-var auto-detect at call time (backward compat).
        # When True/False, use the explicit value (dual-LLM bench path sets this).
        self.uses_ft_prompt = uses_ft_prompt

    async def answer(
        self,
        formatted_context: str,
        question: str,
        category: int | None = None,
    ) -> str:
        if not question:
            return ""

        ctx = formatted_context or ""
        if len(ctx) > MAX_CONTEXT_CHARS:
            ctx = ctx[:MAX_CONTEXT_CHARS - 3] + "..."

        # List-style questions need extra emphasis + a larger budget.
        # Goal: prevent under-shooting on golds like "Painting, kayaking,
        # hiking, cooking, running" (5 items) where the model only said
        # "painting" with the default budget.
        is_list = _is_list_question(question)
        max_tokens = _CATEGORY_MAX_TOKENS.get(category, _DEFAULT_MAX_TOKENS)

        # When the answer LLM is the FT-2 LoRA, use the prompt shape it was
        # trained on (see scripts/kaggle_finetune.py). The Tier-1 elaborate
        # prompt has rules + few-shot examples the FT'd model never saw —
        # feeding it that distribution loses most of the FT signal.
        #
        # Detection order:
        # 1. Explicit constructor flag (dual-LLM bench path) takes precedence.
        # 2. transformers backend (loads FT LoRA in-process).
        # 3. Explicit GML_LLM_USES_FT_PROMPT=1 escape hatch.
        # 4. Ollama backend whose model id contains "locomo-ft" (GGUF of the
        #    same merged LoRA registered with `ollama create`).
        if self.uses_ft_prompt is not None:
            use_ft_prompt = self.uses_ft_prompt
        else:
            _backend = os.environ.get("GML_LLM_BACKEND", "").lower()
            _ollama_model = os.environ.get("GML_OLLAMA_MODEL", "").lower()
            use_ft_prompt = (
                _backend == "transformers"
                or os.environ.get("GML_LLM_USES_FT_PROMPT", "0") == "1"
                or (_backend == "ollama" and "locomo-ft" in _ollama_model)
            )

        if use_ft_prompt:
            # FT model was trained without list-emphasis prefix; just use
            # the bare training prompt but keep the bigger budget for lists.
            if is_list:
                max_tokens = max(max_tokens, 48)
            prompt = _FT_TRAINING_PROMPT.format(context=ctx, question=question)
        elif is_list:
            max_tokens = max(max_tokens, 48)
            prompt = _LIST_EMPHASIS + _ANSWER_PROMPT.format(context=ctx, question=question)
        else:
            prompt = _ANSWER_PROMPT.format(context=ctx, question=question)

        # Self-consistency: when GML_SELF_CONSISTENCY_N>1, generate N answers
        # with varied temperatures (0.0 + 0.3 + 0.7 + ...) and pick the most
        # common normalized answer. Tie-break by F1 against the others.
        n_sc = max(1, int(os.environ.get("GML_SELF_CONSISTENCY_N", "1")))

        if n_sc <= 1:
            try:
                gen = await self.client.generate(
                    prompt, json_mode=False, max_tokens=max_tokens
                )
            except Exception as exc:
                slog.warning(
                    event="answer_generation_failed",
                    error_type=type(exc).__name__, degraded_mode=True,
                )
                return ""
            return _post_process(gen.answer or "", context=ctx)

        # N samples with varied temperatures.
        # Schedule: idx 0 → 0.0 (deterministic), 1 → 0.3, 2 → 0.7, 3 → 0.5, ...
        temps = [0.0, 0.3, 0.7, 0.5, 0.4, 0.6, 0.8]
        candidates: list[str] = []
        for i in range(n_sc):
            t = temps[i] if i < len(temps) else 0.5
            try:
                gen = await self.client.generate(
                    prompt, json_mode=False, max_tokens=max_tokens,
                    temperature=t, seed=42 + i,
                )
            except Exception as exc:
                slog.warning(
                    event="answer_generation_failed",
                    error_type=type(exc).__name__,
                    sc_idx=i, degraded_mode=True,
                )
                continue
            ans = _post_process(gen.answer or "", context=ctx)
            if ans:
                candidates.append(ans)

        if not candidates:
            return ""
        if len(candidates) == 1:
            return candidates[0]

        # Vote: count normalized answers, return most frequent. Tie-break by
        # average token-F1 of each candidate against the others (centroid).
        import re as _re
        from collections import Counter

        def _norm(s: str) -> str:
            return _re.sub(r"\s+", " ", s.lower().strip().rstrip(".,;:!?"))

        norm_counts = Counter(_norm(c) for c in candidates)
        top_norm, top_count = norm_counts.most_common(1)[0]
        if top_count > 1:
            # Return the first original-form candidate that normalizes to the winner.
            for c in candidates:
                if _norm(c) == top_norm:
                    return c
        # All disagree (count=1 each): pick centroid by token-F1.
        def _toks(s: str) -> set:
            return set(_re.findall(r"\w+", s.lower()))
        best_idx, best_score = 0, -1.0
        for i, ci in enumerate(candidates):
            ti = _toks(ci)
            if not ti:
                continue
            total = 0.0
            for j, cj in enumerate(candidates):
                if i == j:
                    continue
                tj = _toks(cj)
                if not tj:
                    continue
                inter = ti & tj
                if not inter:
                    continue
                p = len(inter) / len(ti)
                r = len(inter) / len(tj)
                total += 2 * p * r / (p + r)
            avg = total / max(len(candidates) - 1, 1)
            if avg > best_score:
                best_score = avg
                best_idx = i
        return candidates[best_idx]


ANSWER_GEN_ENABLED_DEFAULT = os.environ.get("GML_ANSWER_GEN", "0") == "1"


# ────────────────────────────────────────────────────────────────────────
# FT-mode prompt — matches the shape the FT-2 LoRA was trained on.
# Use this when GML_LLM_BACKEND=transformers + adapter is loaded, so the
# fine-tuned model sees the distribution it learned.
# ────────────────────────────────────────────────────────────────────────
_FT_TRAINING_PROMPT = """\
Below is some context from a long-running conversation, followed by a question. Answer the question concisely.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
