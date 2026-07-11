from abc import ABC, abstractmethod

from orchestration.pipeline.contracts import AssembledContext, Query, ResolvedMemorySet


class Assembler(ABC):
    """Stage 5: package a ResolvedMemorySet into a budget-fitted AssembledContext.

    The Assembler knows about token budgets but NOT about target-specific
    formatting — that's the Translator's job. The Pipeline supplies the
    template overhead it computed against the chosen Translator adapter.
    """

    @abstractmethod
    def package(
        self,
        resolved: ResolvedMemorySet,
        query: Query,
        template_overhead_tokens: int,
        final: int = 5,
    ) -> AssembledContext: ...
