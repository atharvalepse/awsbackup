"""Stub :class:`Client` for tests and demos with no network calls."""
from orchestration.clients.base import AssistantResponse, Client
from orchestration.pipeline.contracts import TranslatedPayload


class StubClient(Client):
    """Returns a canned reply that echoes (or transforms) the input payload."""

    def __init__(self, response_text: str = "stub response") -> None:
        self.response_text = response_text
        self.received: list[TranslatedPayload] = []

    async def send(self, payload: TranslatedPayload) -> AssistantResponse:
        self.received.append(payload)
        return AssistantResponse(
            text=self.response_text,
            target=payload.target,
            model_version="stub",
            latency_ms=0,
            input_tokens=0,
            output_tokens=len(self.response_text),
            raw_metadata={"echo_user_query": payload.user_query},
        )
