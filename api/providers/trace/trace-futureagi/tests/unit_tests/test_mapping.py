from datetime import UTC, datetime

from dify_trace_futureagi.defaults import COLLECTOR_PATH, DEFAULT_ENDPOINT
from dify_trace_futureagi.mapping import (
    auth_headers,
    resource_attributes,
    span_attributes,
    span_status,
    trace_id_from_message_id,
)
from dify_trace_futureagi.span_input import SpanInput, SpanKind, SpanStatus, TraceInput
from opentelemetry.trace import StatusCode


def _span(**overrides) -> SpanInput:
    values = dict(
        id="span-1",
        parent_id=None,
        name="dify.message",
        kind=SpanKind.LLM,
        start_time=datetime(2026, 10, 4, tzinfo=UTC),
        end_time=datetime(2026, 10, 4, 0, 0, 1, tzinfo=UTC),
        inputs={"query": "hello"},
        outputs={"answer": "world"},
        status=SpanStatus.OK,
        metadata={"app_id": "app-1", "user_id": "user-1", "model_name": "gpt", "prompt_tokens": 3},
    )
    values.update(overrides)
    return SpanInput(**values)


def _trace(span: SpanInput) -> TraceInput:
    return TraceInput(trace_id="trace-1", session_id="conversation-1", spans=(span,))


class TestMapping:
    def test_kinds(self):
        for kind, expected in (
            (SpanKind.LLM, "LLM"),
            (SpanKind.TOOL, "TOOL"),
            (SpanKind.RETRIEVER, "RETRIEVER"),
            (SpanKind.CHAIN, "CHAIN"),
            (SpanKind.AGENT, "AGENT"),
        ):
            attributes = span_attributes(_span(kind=kind), _trace(_span(kind=kind)), redact=False)
            assert attributes["openinference.span.kind"] == expected
            assert attributes["fi.span.kind"] == expected

    def test_absent_token_count_is_absent(self):
        attributes = span_attributes(_span(metadata={"model_name": "gpt"}), _trace(_span()), redact=False)
        assert "llm.token_count.prompt" not in attributes
        assert attributes["llm.model_name"] == "gpt"

    def test_present_token_count_is_kept(self):
        attributes = span_attributes(_span(), _trace(_span()), redact=False)
        assert attributes["llm.token_count.prompt"] == 3

    def test_redact_omits_bodies(self):
        attributes = span_attributes(_span(), _trace(_span()), redact=True)
        assert "input.value" not in attributes
        assert "output.value" not in attributes
        assert attributes["dify.redacted"] is True

    def test_session_and_user_and_no_credential(self):
        attributes = span_attributes(_span(), _trace(_span()), redact=False)
        assert attributes["session.id"] == "conversation-1"
        assert attributes["gen_ai.conversation.id"] == "conversation-1"
        assert attributes["user.id"] == "user-1"
        assert "api_key" not in attributes
        assert "secret_key" not in attributes

    def test_error_status_keeps_the_message(self):
        code, description = span_status(_span(status=SpanStatus.ERROR, error="boom"))
        assert code is StatusCode.ERROR
        assert description == "boom"

    def test_error_status_does_not_invent_a_message(self):
        code, description = span_status(_span(status=SpanStatus.ERROR, error=None))
        assert code is StatusCode.ERROR
        assert description is None

    def test_same_message_id_same_trace_bytes(self):
        assert trace_id_from_message_id("message-9") == trace_id_from_message_id("message-9")
        assert trace_id_from_message_id("message-9") != trace_id_from_message_id("message-10")

    def test_two_configs_keep_separate_headers(self):
        assert auth_headers("key-a", "secret-a") != auth_headers("key-b", "secret-b")
        assert auth_headers("key-a", "secret-a") == {"X-Api-Key": "key-a", "X-Secret-Key": "secret-a"}

    def test_resource_requires_project_name(self):
        attributes = resource_attributes("proj-a")
        assert attributes == {"project_name": "proj-a", "project_type": "observe"}
        assert "openinference.project.name" not in attributes

    def test_collector_path(self):
        from urllib.parse import urlparse

        def endpoint(value: str) -> str:
            parsed = urlparse(value)
            return f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}{COLLECTOR_PATH}"

        assert endpoint(DEFAULT_ENDPOINT) == f"{DEFAULT_ENDPOINT}{COLLECTOR_PATH}"
        assert endpoint("https://example.com/base/") == f"https://example.com/base{COLLECTOR_PATH}"

    def test_one_span_per_message(self):
        assert len(_trace(_span(inputs={"chunks": ["a", "b", "c"]})).spans) == 1
