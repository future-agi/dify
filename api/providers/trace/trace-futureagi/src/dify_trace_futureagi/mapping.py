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


def _message_content(message: Mapping[str, Any]) -> str | None:
    """Dify prompt messages carry their body in "text"; OpenAI-style ones in "content"."""
    content = message.get("content", message.get("text"))
    if content is None or isinstance(content, str):
        return content
    return _json(content)


def llm_message_attributes(inputs: Any, outputs: Any) -> dict[str, Any]:
    """OpenInference chat messages, so Observe renders an LLM span as a conversation.

    inputs is a list of Dify prompt messages (role, text, tool_calls) or a bare
    query string. outputs is the LLM node's {"text": ...} or the answer string.
    """
    attributes: dict[str, Any] = {}
    if isinstance(inputs, str):
        messages: list[Any] = [{"role": "user", "content": inputs}] if inputs else []
    else:
        messages = inputs if isinstance(inputs, list) else []
    index = 0
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        prefix = f"llm.input_messages.{index}.message"
        role = message.get("role")
        if role:
            attributes[f"{prefix}.role"] = str(role)
        content = _message_content(message)
        if content is not None:
            attributes[f"{prefix}.content"] = content
        tool_calls = message.get("tool_calls")
        for call_index, call in enumerate(tool_calls if isinstance(tool_calls, list) else []):
            function = call.get("function") if isinstance(call, Mapping) else None
            if not isinstance(function, Mapping):
                continue
            call_prefix = f"{prefix}.tool_calls.{call_index}.tool_call"
            if call.get("id"):
                attributes[f"{call_prefix}.id"] = str(call["id"])
            attributes[f"{call_prefix}.function.name"] = str(function.get("name", ""))
            arguments = function.get("arguments")
            attributes[f"{call_prefix}.function.arguments"] = (
                arguments if isinstance(arguments, str) else _json(arguments)
            )
        index += 1
    answer = outputs.get("text") if isinstance(outputs, Mapping) else outputs
    if isinstance(answer, str) and answer:
        attributes["llm.output_messages.0.message.role"] = "assistant"
        attributes["llm.output_messages.0.message.content"] = answer
    return attributes


def span_attributes(span: SpanInput, trace: TraceInput, *, redact: bool) -> dict[str, Any]:
    """Attributes one canonical span exports. Never includes a credential.

    Model and token counts go on LLM spans only. fi-collector sums tokens over
    every span in a session, so a CHAIN span repeating its child's usage would
    count it twice. The synthetic LLM span of a chatflow message trace repeats
    the usage of the workflow's own LLM node spans, so it carries no tokens.
    """
    attributes: dict[str, Any] = {
        "openinference.span.kind": _KIND_MAP[span.kind],
        "fi.span.kind": _KIND_MAP[span.kind],
        "dify.span.id": span.id,
    }
    if trace.session_id:
        attributes["session.id"] = trace.session_id
        attributes["gen_ai.conversation.id"] = trace.session_id
    metadata = span.metadata
    for key in _METADATA_KEYS:
        value = metadata.get(key)
        if value is not None:
            attributes[f"dify.{key}"] = value
    user_id = metadata.get("user_id")
    if user_id is not None:
        attributes["user.id"] = user_id
    if span.kind is SpanKind.LLM:
        model_name = metadata.get("model_name")
        if model_name is not None:
            attributes["llm.model_name"] = model_name
        provider = metadata.get("model_provider")
        if provider is not None:
            attributes["llm.provider"] = provider
        if not (span.synthetic and metadata.get("workflow_run_id")):
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
        if span.kind is SpanKind.LLM:
            attributes.update(llm_message_attributes(span.inputs, span.outputs))
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
