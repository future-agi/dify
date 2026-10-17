"""Trace and span ids derived from Dify's own ids, with no Dify runtime import.

Dify retries a trace dispatch when an export fails part way (RetryableTraceDispatchError),
and the retry rebuilds the same canonical trace. Random OpenTelemetry ids would store a
second copy of every span that was already accepted. Ids derived from the canonical trace
and span ids make the retry re-export the same (trace_id, span_id) pairs, which Future AGI
stores once.
"""

import uuid
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

from opentelemetry.sdk.trace.id_generator import IdGenerator, RandomIdGenerator

_NAMESPACE = uuid.UUID("6f0a2d7e-3c1b-5e4f-9a8d-2b7c6e5f4a31")
_SPAN_ID_MASK = (1 << 64) - 1

_next_ids: ContextVar[tuple[int, int] | None] = ContextVar("dify_trace_futureagi_next_ids", default=None)


def stable_trace_id(canonical_trace_id: str) -> int:
    """128-bit OpenTelemetry trace id for a canonical trace id. Same input, same id."""
    return uuid.uuid5(_NAMESPACE, f"trace:{canonical_trace_id}").int or 1


def stable_span_id(canonical_trace_id: str, root_span_name: str, canonical_span_id: str) -> int:
    """64-bit span id. The root span name keeps Dify's workflow trace and message trace
    of one chatflow turn apart when both carry the same external trace id."""
    seed = f"span:{canonical_trace_id}:{root_span_name}:{canonical_span_id}"
    return (uuid.uuid5(_NAMESPACE, seed).int & _SPAN_ID_MASK) or 1


class StableIdGenerator(IdGenerator):
    """Hands out the ids set by `next_ids`; random ids outside of it.

    The tracer asks for a trace id only for a span without a parent in its context,
    so child spans and spans under a restored external parent keep that parent's trace.
    """

    def __init__(self) -> None:
        self._random = RandomIdGenerator()

    def generate_trace_id(self) -> int:
        ids = _next_ids.get()
        return ids[0] if ids is not None else self._random.generate_trace_id()

    def generate_span_id(self) -> int:
        ids = _next_ids.get()
        return ids[1] if ids is not None else self._random.generate_span_id()


@contextmanager
def next_ids(trace_id: int, span_id: int) -> Generator[None]:
    """Use these ids for the span started inside the block."""
    token = _next_ids.set((trace_id, span_id))
    try:
        yield
    finally:
        _next_ids.reset(token)
