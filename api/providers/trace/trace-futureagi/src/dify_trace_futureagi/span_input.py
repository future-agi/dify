"""Span values the Future AGI mapper reads.

These mirror the fields of Dify's CanonicalSpan and CanonicalTrace that the
mapper uses. The mapper takes these rather than importing the Dify entities,
because that import pulls in the API app.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class SpanKind(StrEnum):
    CHAIN = "chain"
    LLM = "llm"
    RETRIEVER = "retriever"
    TOOL = "tool"
    AGENT = "agent"


class SpanStatus(StrEnum):
    OK = "ok"
    ERROR = "error"


@dataclass(frozen=True)
class SpanInput:
    id: str
    parent_id: str | None
    name: str
    kind: SpanKind
    start_time: datetime
    end_time: datetime | None
    inputs: Any
    outputs: Any
    status: SpanStatus
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TraceInput:
    trace_id: str
    session_id: str
    spans: tuple[SpanInput, ...]
