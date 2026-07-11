"""SDP — Semantic Decomposition Pipeline.

Lightweight (no-LLM) semantic ingestion: parse → extract → entity-link →
score → summarize → emit AALMemory objects ready to convert into the
existing MemoryItem contract.

Use `SDPPipeline().process_turn(user_query, assistant_reply)` to get a
list of AALMemory objects; call `.to_memory_item()` on each to feed the
existing store + retriever.
"""
from orchestration.sdp.aal import AALMemory, SemanticUnit
from orchestration.sdp.aal_tuples import AALTupleExtractor, AAL_ENABLED_DEFAULT
from orchestration.sdp.dedup import MinHashDeduper
from orchestration.sdp.entity_index import EntityIndex, extract_entities
from orchestration.sdp.extractor import SemanticExtractor
from orchestration.sdp.hyde import hyde_fuse, hyde_rewrite, HYDE_ENABLED_DEFAULT
from orchestration.sdp.linker import Entity, EntityExtractor, Relationship, RelationshipMapper
from orchestration.sdp.parser import ConversationParser
from orchestration.sdp.pipeline import SDPPipeline
from orchestration.sdp.query_router import QueryHints, classify_query
from orchestration.sdp.scorer import ConfidenceScorer, ImportanceScorer
from orchestration.sdp.summarizer import SemanticSummarizer
from orchestration.sdp.writer import SDPWriter


__all__ = [
    "AAL_ENABLED_DEFAULT",
    "AALMemory",
    "AALTupleExtractor",
    "ConfidenceScorer",
    "ConversationParser",
    "Entity",
    "EntityExtractor",
    "EntityIndex",
    "extract_entities",
    "HYDE_ENABLED_DEFAULT",
    "hyde_fuse",
    "hyde_rewrite",
    "ImportanceScorer",
    "MinHashDeduper",
    "QueryHints",
    "classify_query",
    "Relationship",
    "RelationshipMapper",
    "SDPPipeline",
    "SDPWriter",
    "SemanticExtractor",
    "SemanticSummarizer",
    "SemanticUnit",
]
