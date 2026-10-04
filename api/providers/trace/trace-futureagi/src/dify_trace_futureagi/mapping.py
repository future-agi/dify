"""Map a provider-neutral canonical trace onto Future AGI span attributes.

Kept free of the Dify runtime so the mapper can be tested without a
database, Redis, or a running collector.
"""

import json
import uuid
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from opentelemetry.trace import StatusCode

from dify_trace_futureagi.span_input import SpanInput, SpanKind, SpanStatus, TraceInput

_KIND_MAP: dict[SpanKind, str] = {
    SpanKind.CHAIN: "CHAIN",
    SpanKind.LLM: "LLM",
    SpanKind.RETRIEVER: "RETRIEVER",
    SpanKind.TOOL: "TOOL",
    SpanKind.AGENT: "AGENT",
}

_METADATA_KEYS = ("app_id", "app_mode", "tenant_id", "version")


def trace_id_from_message_id(message_id: str) -> int:
    """Stable 16-byte trace id from a Dify message or workflow run id.

    The same id always yields the same bytes, so a retried trace lands on
    the same Observe trace instead of a new one.
    """
    return uuid.uuid5(uuid.NAMESPACE_URL, f"dify:{message_id}").int


def nanos(value: datetime | None) -> int | None:
    return int(value.timestamp() * 1_000_000_000) if value is not None else None


def _json(value: object) -> str:
    return json.dumps(value, default=str, ensure_ascii=False)


def span_attributes(span: SpanInput, trace: TraceInput, *, redact: bool) -> dict[str, Any]:
    """Attributes one canonical span exports. Never includes a credential."""
    attributes: dict[str, Any] = {
        "openinference.span.kind": _KIND_MAP[span.kind],
        "fi.span.kind": _KIND_MAP[span.kind],
        "session.id": trace.session_id,
        "gen_ai.conversation.id": trace.session_id,
        "dify.span.id": span.id,
    }
    metadata = span.metadata
    for key in _METADATA_KEYS:
        value = metadata.get(key)
        if value is not None:
            attributes[f"dify.{key}"] = value
    user_id = metadata.get("user_id")
    if user_id is not None:
        attributes["user.id"] = user_id
    model_name = metadata.get("model_name")
    if model_name is not None:
        attributes["llm.model_name"] = model_name
    provider = metadata.get("model_provider")
    if provider is not None:
        attributes["llm.provider"] = provider
    for source, target in (
        ("prompt_tokens", "llm.token_count.prompt"),
        ("completion_tokens", "llm.token_count.completion"),
    ):
        count = metadata.get(source)
        if count is not None:
            attributes[target] = count
    if redact:
        attributes["dify.redacted"] = True
    else:
        attributes["input.value"] = _json(span.inputs)
        attributes["input.mime_type"] = "application/json"
        attributes["output.value"] = _json(span.outputs)
        attributes["output.mime_type"] = "application/json"
    return attributes


def span_status(span: SpanInput) -> tuple[StatusCode, str | None]:
    if span.status is SpanStatus.ERROR:
        return StatusCode.ERROR, span.error
    return StatusCode.OK, None


def resource_attributes(project: str) -> dict[str, str]:
    """The two attributes fi-collector requires. openinference.project.name is not a substitute."""
    return {"project_name": project, "project_type": "observe"}


def auth_headers(api_key: str | None, secret_key: str | None) -> Mapping[str, str]:
    """fi-collector reads X-Api-Key and X-Secret-Key. Bearer and api_key are ignored."""
    return {"X-Api-Key": api_key or "", "X-Secret-Key": secret_key or ""}
