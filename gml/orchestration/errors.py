class OrchestrationError(Exception):
    """Base class for all errors raised by the orchestration layer."""


class TargetDescriptorError(OrchestrationError):
    """Raised when a TargetDescriptor is malformed, missing, or incompatible."""


class ClassifierError(OrchestrationError):
    """Raised when classification fails or times out."""


class EmbedderError(OrchestrationError):
    """Raised when embedding generation fails."""


class RetrieverError(OrchestrationError):
    """Raised when no Retriever can serve a request."""


class RerankerError(OrchestrationError):
    """Raised when reranking fails."""


class SAMError(OrchestrationError):
    """Raised when SAM cannot resolve conflicts or reason from scratch."""


class BudgetExceededError(OrchestrationError):
    """Raised when assembly cannot fit any memories within the token budget."""


class TranslatorError(OrchestrationError):
    """Raised when the per-target Translator adapter cannot render context."""
