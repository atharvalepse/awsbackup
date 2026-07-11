"""In-memory cosine-similarity Retriever over a fixed MemoryItem fixture.

Stand-in for a real vector DB. Records are embedded at construction time
using the same hash-to-vector helper as the stub Embedder, so cosine
similarity between a stub-embedded query and a stub retriever record is
mathematically meaningful for tests.
"""
from datetime import datetime, timedelta, timezone

from orchestration.pipeline._stub_vectors import hash_to_unit_vector
from orchestration.pipeline.contracts import EmbeddedQuery, MemoryItem, RetrievalHit
from orchestration.retriever.base import Retriever


DEFAULT_DIM = 384
DEFAULT_MATCH_THRESHOLD = 0.0  # cosine, before reranking — kept permissive


def default_records() -> list[MemoryItem]:
    """Eight memory items covering structured/unstructured/pinned/conflicting cases.

    Includes a deliberate entity-attribute conflict (m-1 vs m-2 on
    ``auth_service.framework``) so SAM has something to resolve in tests.
    """
    now = datetime.now(timezone.utc)
    return [
        MemoryItem(
            id="m-1",
            content=(
                "The auth_service is implemented in FastAPI with JWT-based session tokens, "
                "deployed via the orchestrator-managed Kubernetes cluster. Login flow "
                "proxies to the corporate IdP."
            ),
            summary_medium="auth_service runs FastAPI with JWT tokens; login proxies to corporate IdP",
            summary_short="auth: FastAPI + JWT",
            entity="auth_service",
            attribute="framework",
            value="FastAPI",
            timestamp=now - timedelta(days=2),
            source="adr-0001",
            authority_score=0.95,
            pinned=True,
            token_counts={"tiktoken:cl100k_base": 33},
        ),
        MemoryItem(
            id="m-2",
            content=(
                "The auth_service was previously implemented in Flask before the Q1 2025 "
                "migration to FastAPI. The Flask version is no longer supported and the "
                "code paths have been removed from the monorepo."
            ),
            summary_medium="auth_service was Flask before Q1 2025 migration; unsupported now",
            entity="auth_service",
            attribute="framework",
            value="Flask",
            timestamp=now - timedelta(days=400),
            source="historical-note",
            authority_score=0.3,
        ),
        MemoryItem(
            id="m-3",
            content=(
                "The payment_service runs on PostgreSQL 16 with a dedicated read replica "
                "for analytics queries. Schemas are managed by Alembic migrations checked "
                "into the monorepo and applied during deploy."
            ),
            summary_medium="payment_service: PostgreSQL 16 with analytics replica, Alembic migrations",
            summary_short="payments: PG16 + replica",
            entity="payment_service",
            attribute="database",
            value="PostgreSQL 16",
            timestamp=now - timedelta(days=30),
            source="config-snapshot",
            authority_score=0.8,
        ),
        MemoryItem(
            id="m-4",
            content=(
                "User prefers concise, code-first explanations with minimal prose. "
                "They've explicitly asked to skip pleasantries and dive straight into "
                "solutions in past sessions."
            ),
            summary_medium="user prefers concise code-first answers; skip pleasantries",
            timestamp=now - timedelta(days=14),
            source="user-preference",
            authority_score=0.8,
            pinned=True,
        ),
        MemoryItem(
            id="m-5",
            content=(
                "The last production incident root cause was a JWT clock-skew bug between "
                "auth_service nodes and edge proxies; token validation began rejecting "
                "fresh tokens whenever the skew exceeded 30 seconds."
            ),
            timestamp=now - timedelta(days=5),
            source="postmortem",
            authority_score=0.5,
        ),
        MemoryItem(
            id="m-6",
            content=(
                "Team standup is Mondays at 10:00 PT on Zoom; engineering announcements "
                "happen there before going to email or the Slack #eng channel."
            ),
            summary_medium="standup: Mon 10am PT Zoom; eng announcements happen there first",
            summary_short="standup Mon 10am PT",
            timestamp=now - timedelta(days=60),
            source="team-handbook",
            authority_score=0.5,
        ),
        MemoryItem(
            id="m-7",
            content=(
                "Yesterday we deployed v2.4.1 of the orchestrator to staging; canary "
                "checks passed cleanly and the production rollout is gated on tomorrow's "
                "load test results."
            ),
            timestamp=now - timedelta(hours=18),
            source="deploy-log",
            authority_score=0.5,
        ),
        MemoryItem(
            id="m-8",
            content=(
                "Style guide prohibits f-strings inside logger calls — use %-formatting "
                "so the logger can defer string interpolation. Applies to all production "
                "code; tests are exempt."
            ),
            timestamp=now - timedelta(days=120),
            source="style-guide",
            authority_score=0.95,
        ),
    ]


def _cosine(a: list[float], b: list[float]) -> float:
    # Both vectors are L2-normalized in this stub, so dot product == cosine.
    return sum(x * y for x, y in zip(a, b))


def _embed_record(record: MemoryItem, dim: int) -> list[float]:
    """Build the per-record vector. Includes entity/attribute when present so
    structured records cluster slightly differently from pure prose."""
    signal = record.content
    if record.entity:
        signal += " || " + record.entity
        if record.attribute:
            signal += ":" + record.attribute
    return hash_to_unit_vector(signal, dim)


class StubRetriever(Retriever):
    """In-memory vector retriever. Cosine similarity over pre-embedded records.

    Use the default constructor for the canonical 8-item fixture, or pass
    your own ``records`` for custom test scenarios.
    """

    def __init__(
        self,
        records: list[MemoryItem] | None = None,
        dim: int = DEFAULT_DIM,
        match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    ) -> None:
        self.dim = dim
        self.match_threshold = match_threshold
        self.records: list[MemoryItem] = list(records) if records is not None else default_records()
        self._vectors: dict[str, list[float]] = {
            r.id: _embed_record(r, dim) for r in self.records
        }

    async def search(self, embedded: EmbeddedQuery) -> list[RetrievalHit]:
        """Cheap probe used by Pipeline to decide the YES/NO branch.

        Returns every record whose cosine similarity exceeds the configured
        threshold, sorted by similarity descending. May be empty.
        """
        return self._rank(embedded.vector, k=len(self.records), threshold=self.match_threshold)

    async def get_top_matches(
        self, embedded: EmbeddedQuery, k: int = 50
    ) -> list[RetrievalHit]:
        """Top-``k`` retrieval. Called only after ``search`` returned non-empty."""
        return self._rank(embedded.vector, k=k, threshold=self.match_threshold)

    def _rank(self, query_vec: list[float], k: int, threshold: float) -> list[RetrievalHit]:
        if len(query_vec) != self.dim:
            raise ValueError(
                f"StubRetriever expects {self.dim}-dim vectors, got {len(query_vec)}"
            )
        scored: list[RetrievalHit] = []
        for record in self.records:
            sim = _cosine(query_vec, self._vectors[record.id])
            if sim > threshold:
                scored.append(RetrievalHit(record=record, similarity=sim))
        scored.sort(key=lambda h: h.similarity, reverse=True)
        return scored[:k]
