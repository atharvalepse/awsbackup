"""Target-AI client layer.

After the pipeline produces a :class:`TranslatedPayload`, a Client sends it
to the actual target AI (Claude / GPT / Gemini / DeepSeek / Llama) and
returns the model's reply as an :class:`AssistantResponse`.

Implementations live one-per-vendor in this package. Clients are pure
ship-the-string objects — no orchestration logic.
"""
from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field

from orchestration.pipeline.contracts import TargetDescriptor, TranslatedPayload


class AssistantResponse(BaseModel):
    """What a target AI returned for one turn."""

    model_config = ConfigDict(extra="forbid")

    text: str
    target: TargetDescriptor
    model_version: str
    latency_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    raw_metadata: dict = Field(default_factory=dict)


class Client(ABC):
    """Stage 8 (post-pipeline): send a payload, receive a reply."""

    @abstractmethod
    async def send(self, payload: TranslatedPayload) -> AssistantResponse: ...
