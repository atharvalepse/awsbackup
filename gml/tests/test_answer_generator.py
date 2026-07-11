"""Tests for AnswerGenerator post-processing, list detection, synonym snap.

Drives the unit-level behavior of the answer generator without involving
a real LLM. The end-to-end ``answer()`` call is tested via the LOCOMO
self-test script.
"""
import pytest

from orchestration.sam.answer_generator import (
    _is_list_question,
    _post_process,
    _snap_synonyms,
)


class TestPostProcess:
    def test_strips_preambles(self):
        assert _post_process("Based on the context, the answer is Prius.") == "Prius"
        assert _post_process("Answer: 7 May 2023") == "7 May 2023"
        assert _post_process("According to the context, Caroline went on Sunday.") == "Caroline went on Sunday"

    def test_strips_stacked_preambles(self):
        # Qwen sometimes piles "Based on the context, based on the provided context, the answer is X"
        assert _post_process("Based on the context, based on the provided context, the answer is Prius") == "Prius"

    def test_strips_quotes_and_markdown(self):
        assert _post_process('"Adyen"') == "Adyen"
        assert _post_process("**Prius**") == "Prius"
        assert _post_process("`Stripe`") == "Stripe"

    def test_takes_only_first_sentence(self):
        raw = "Caroline went on Sunday May 7 2023. This is based on memory dated..."
        assert _post_process(raw) == "Caroline went on Sunday May 7 2023"

    def test_preserves_dont_know(self):
        assert _post_process("I don't know.") == "I don't know"

    def test_empty(self):
        assert _post_process("") == ""
        assert _post_process("   ") == ""


class TestSynonymSnap:
    def test_snaps_when_context_uses_short_form(self):
        out = _snap_synonyms("Jasper, Rocky Mountains", "Evan visited the Rockies")
        assert out == "Jasper, Rockies"

    def test_no_snap_when_context_uses_long_form(self):
        # Context says Rocky Mountains — leave model's output alone.
        out = _snap_synonyms("Rocky Mountains", "We hiked the Rocky Mountains")
        assert out == "Rocky Mountains"

    def test_snaps_nyc(self):
        out = _snap_synonyms("New York City was great", "moved to NYC last year")
        assert out == "NYC was great"

    def test_no_snap_when_neither_form_present(self):
        # Adyen has no synonym pair — leave alone.
        out = _snap_synonyms("Adyen", "switched to Adyen")
        assert out == "Adyen"

    def test_empty_inputs(self):
        assert _snap_synonyms("", "ctx") == ""
        assert _snap_synonyms("ans", "") == "ans"


class TestListQuestionDetection:
    @pytest.mark.parametrize("q", [
        "What kinds of things did Evan have broken?",
        "What hobbies does Sam have?",
        "What activities does Caroline do?",
        "Name all the cars Evan owned",
        "What practices does Caroline do?",
        "What sorts of hobbies does she enjoy?",
        "List the places Evan visited",
    ])
    def test_positive(self, q):
        assert _is_list_question(q) is True

    @pytest.mark.parametrize("q", [
        "What kind of car does Evan drive?",      # singular — not a list
        "When did Evan visit Jasper?",
        "Who is Mel?",
        "What is Sam working on?",
        "Where is the meeting?",
        "How many Prius has Evan owned?",
    ])
    def test_negative(self, q):
        assert _is_list_question(q) is False

    def test_empty(self):
        assert _is_list_question("") is False
        assert _is_list_question(None) is False  # type: ignore[arg-type]


class TestPostProcessWithContext:
    """End-to-end post-processor including the synonym-snap step."""

    def test_full_flow_with_synonym_snap(self):
        raw = "The answer is Rocky Mountains."
        ctx = "Evan visited the Rockies on his trip"
        assert _post_process(raw, context=ctx) == "Rockies"

    def test_full_flow_without_context_skips_snap(self):
        # No context = no snap, but preamble strip still works.
        raw = "Answer: Rocky Mountains"
        assert _post_process(raw) == "Rocky Mountains"
