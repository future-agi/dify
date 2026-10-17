from datetime import UTC, datetime
from typing import Any

from dify_trace_futureagi.defaults import COLLECTOR_PATH, DEFAULT_ENDPOINT
from dify_trace_futureagi.mapping import (
    auth_headers,
    llm_message_attributes,
    resource_attributes,
    span_attributes,
    span_status,
)
from dify_trace_futureagi.span_input import SpanInput, SpanKind, SpanStatus, TraceInput
from opentelemetry.trace import StatusCode


def _span(**overrides) -> SpanInput:
    values: dict[str, Any] = {
        "id": "span-1",
        "parent_id": None,
        "name": "dify.message",
        "kind": SpanKind.LLM,
        "start_time": datetime(2026, 10, 4, tzinfo=UTC),
        "end_time": datetime(2026, 10, 4, 0, 0, 1, tzinfo=UTC),
        "inputs": {"query": "hello"},
        "outputs": {"answer": "world"},
        "status": SpanStatus.OK,
        "metadata": {"app_id": "app-1", "user_id": "user-1", "model_name": "gpt", "prompt_tokens": 3},
    }
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
        code, description = span_status(_span(status=SpanStatus.ERROR, error="boom"), redact=False)
        assert code is StatusCode.ERROR
        assert description == "boom"

    def test_redact_keeps_error_status_but_drops_the_error_text(self):
        """Dify error text can quote the user's input, e.g. a tool repeating its bad argument."""
        span = _span(status=SpanStatus.ERROR, error="Failed to invoke tool time: 'Mars/secret-input'")
        code, description = span_status(span, redact=True)
        assert code is StatusCode.ERROR
        assert description is None

    def test_error_status_does_not_invent_a_message(self):
        code, description = span_status(_span(status=SpanStatus.ERROR, error=None), redact=False)
        assert code is StatusCode.ERROR
        assert description is None

    def test_two_configs_keep_separate_headers(self):
        assert auth_headers("key-a", "secret-a") != auth_headers("key-b", "secret-b")
        assert auth_headers("key-a", "secret-a") == {"X-Api-Key": "key-a", "X-Secret-Key": "secret-a"}

    def test_resource_requires_project_name(self):
        attributes = resource_attributes("proj-a")
        assert attributes == {"project_name": "proj-a", "project_type": "observe"}
        assert "openinference.project.name" not in attributes

    def test_collector_path(self):
        from dify_trace_futureagi.defaults import collector_endpoint

        assert collector_endpoint(DEFAULT_ENDPOINT) == f"{DEFAULT_ENDPOINT}{COLLECTOR_PATH}"
        assert collector_endpoint("https://example.com/base/") == f"https://example.com/base{COLLECTOR_PATH}"
        assert (
            collector_endpoint("https://api.futureagi.com/tracer/v1/traces")
            == "https://api.futureagi.com/tracer/v1/traces"
        )

    def test_one_span_per_message(self):
        assert len(_trace(_span(inputs={"chunks": ["a", "b", "c"]})).spans) == 1

    def test_chain_span_does_not_repeat_model_or_tokens(self):
        """fi-collector sums tokens over every span in a session; a CHAIN copy would double them."""
        span = _span(kind=SpanKind.CHAIN, metadata={"model_name": "gpt", "prompt_tokens": 3, "completion_tokens": 4})
        attributes = span_attributes(span, _trace(span), redact=False)
        assert not [key for key in attributes if key.startswith("llm.")]

    def test_chatflow_message_llm_span_does_not_repeat_workflow_usage(self):
        metadata = {"workflow_run_id": "run-1", "prompt_tokens": 112, "completion_tokens": 71}
        span = _span(synthetic=True, metadata=metadata)
        attributes = span_attributes(span, _trace(span), redact=False)
        assert "llm.token_count.prompt" not in attributes
        assert "llm.token_count.completion" not in attributes

    def test_chat_app_message_llm_span_keeps_its_usage(self):
        span = _span(synthetic=True, metadata={"workflow_run_id": None, "prompt_tokens": 9, "completion_tokens": 2})
        attributes = span_attributes(span, _trace(span), redact=False)
        assert attributes["llm.token_count.prompt"] == 9
        assert attributes["llm.token_count.completion"] == 2

    def test_empty_session_is_omitted(self):
        span = _span()
        attributes = span_attributes(span, TraceInput(trace_id="trace-1", session_id="", spans=(span,)), redact=False)
        assert "session.id" not in attributes
        assert "gen_ai.conversation.id" not in attributes

    def test_llm_span_exports_chat_messages(self):
        prompts = [
            {"role": "system", "text": "Be brief.", "files": []},
            {"role": "user", "text": "What time is it?", "files": []},
            {
                "role": "assistant",
                "text": "",
                "tool_calls": [
                    {"id": "call-1", "type": "function", "function": {"name": "current_time", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "text": "2026-10-12 03:50:42", "files": []},
        ]
        span = _span(inputs=prompts, outputs={"text": "It is 03:50 UTC.", "usage": {"prompt_tokens": 3}})
        attributes = span_attributes(span, _trace(span), redact=False)
        assert attributes["llm.input_messages.0.message.role"] == "system"
        assert attributes["llm.input_messages.0.message.content"] == "Be brief."
        assert attributes["llm.input_messages.1.message.content"] == "What time is it?"
        call = "llm.input_messages.2.message.tool_calls.0.tool_call"
        assert attributes[f"{call}.id"] == "call-1"
        assert attributes[f"{call}.function.name"] == "current_time"
        assert attributes[f"{call}.function.arguments"] == "{}"
        assert attributes["llm.input_messages.3.message.role"] == "tool"
        assert attributes["llm.output_messages.0.message.role"] == "assistant"
        assert attributes["llm.output_messages.0.message.content"] == "It is 03:50 UTC."

    def test_llm_messages_accept_a_bare_query_and_answer(self):
        attributes = llm_message_attributes("hello", "world")
        assert attributes["llm.input_messages.0.message.role"] == "user"
        assert attributes["llm.input_messages.0.message.content"] == "hello"
        assert attributes["llm.output_messages.0.message.content"] == "world"
        assert llm_message_attributes("", None) == {}

    def test_redact_omits_chat_messages(self):
        span = _span(inputs=[{"role": "user", "text": "secret"}], outputs={"text": "secret answer"})
        attributes = span_attributes(span, _trace(span), redact=True)
        assert not [key for key in attributes if "messages" in key]
        assert "secret" not in str(attributes)

    def test_non_llm_spans_carry_no_chat_messages(self):
        span = _span(kind=SpanKind.TOOL, inputs={"format": "%H"}, outputs={"text": "03"})
        attributes = span_attributes(span, _trace(span), redact=False)
        assert not [key for key in attributes if "messages" in key]
