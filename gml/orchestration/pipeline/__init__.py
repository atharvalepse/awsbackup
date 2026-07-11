"""Pipeline package surface.

``Pipeline`` and ``default_tokenizer_factory`` are imported lazily via
:pep:`562` ``__getattr__``. Eager re-export creates an unavoidable cycle:
each stage module imports from ``orchestration.pipeline.contracts``,
which forces ``orchestration.pipeline.__init__`` to run, which would
import ``orchestration.pipeline.pipeline``, which imports the stages —
back into the package that's still mid-init.

By keeping only ``contracts`` (no deps) and ``config_loader`` (depends
only on contracts) eager, the package init finishes cleanly. ``Pipeline``
resolves on first access.
"""
from orchestration.pipeline.contracts import (
    AssembledContext,
    Classification,
    ClassificationSource,
    EmbeddedQuery,
    InterfaceType,
    MemoryItem,
    ModelFamily,
    OrchestrationConfig,
    Query,
    RankedHit,
    ResolvedMemorySet,
    RetrievalHit,
    TargetDescriptor,
    TraceEntry,
    TranslatedPayload,
)
from orchestration.pipeline.config_loader import load_config


def __getattr__(name: str):
    if name == "Pipeline":
        from orchestration.pipeline.pipeline import Pipeline
        return Pipeline
    if name == "default_tokenizer_factory":
        from orchestration.pipeline.pipeline import default_tokenizer_factory
        return default_tokenizer_factory
    raise AttributeError(f"module 'orchestration.pipeline' has no attribute {name!r}")


__all__ = [
    "Pipeline",
    "default_tokenizer_factory",
    "load_config",
    "AssembledContext",
    "Classification",
    "ClassificationSource",
    "EmbeddedQuery",
    "InterfaceType",
    "MemoryItem",
    "ModelFamily",
    "OrchestrationConfig",
    "Query",
    "RankedHit",
    "ResolvedMemorySet",
    "RetrievalHit",
    "TargetDescriptor",
    "TraceEntry",
    "TranslatedPayload",
]
