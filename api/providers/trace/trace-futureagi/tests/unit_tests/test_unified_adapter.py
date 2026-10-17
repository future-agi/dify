"""Unified adapter against an in-memory exporter: tree shape, errors, redaction, retry ids, transport."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest
from dify_trace_futureagi.config import FutureAGIConfig
from dify_trace_futureagi.unified_trace import UnifiedFutureAGIAdapter, setup_unified_tracer
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import StatusCode

from configs import dify_config
from core.ops.exceptions import RetryableTraceDispatchError
from core.ops.unified_trace.entities import CanonicalSpan, CanonicalSpanKind, CanonicalSpanStatus, CanonicalTrace

T0 = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)
MARKER = "Mars/marker-7c1e"


class _Exporter:
    """Records exported spans; fails the export numbered `fail_on` (1-based) once."""

    def __init__(self, fail_on: int | None = None) -> None:
        self.spans: list[ReadableSpan] = []
        self.calls = 0
        self.fail_on = fail_on

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.calls += 1
        if self.calls == self.fail_on:
            return SpanExportResult.FAILURE
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS


def _config(*, redact: bool = False) -> FutureAGIConfig:
    return FutureAGIConfig(api_key="k", secret_key="s", project="proj", endpoint="http://127.0.0.1:9", redact=redact)


def _span(span_id: str, parent: str | None, kind: CanonicalSpanKind, offset: int, **extra) -> CanonicalSpan:
    return CanonicalSpan(
        id=span_id,
        parent_id=parent,
        name=f"{kind.value}_{span_id}",
        kind=kind,
        start_time=T0 + timedelta(seconds=offset),
        end_time=T0 + timedelta(seconds=offset + 1),
        inputs={"query": f"what time is it {MARKER}"},
        outputs={"text": "noon"},
        status=extra.pop("status", CanonicalSpanStatus.OK),
        **extra,
    )


def _workflow_trace() -> CanonicalTrace:
    """message > workflow > (llm, tool failing with the user's input in its error)."""
    return CanonicalTrace(
        trace_id="run-1",
        session_id="conversation-1",
        root_span_id="message-1",
        spans=(
            _span("message-1", None, CanonicalSpanKind.CHAIN, 0, publishes_parent_context=True),
            _span("run-1", "message-1", CanonicalSpanKind.CHAIN, 0),
            _span("node-llm", "run-1", CanonicalSpanKind.LLM, 1, metadata={"model_name": "gpt-x", "prompt_tokens": 5}),
            _span(
                "node-tool",
                "run-1",
                CanonicalSpanKind.TOOL,
                2,
                status=CanonicalSpanStatus.ERROR,
                error=f"Failed to invoke tool time: '{MARKER}'",
                can_parent_workflow=True,
            ),
        ),
    )


def _emit(config: FutureAGIConfig, exporter: _Exporter, trace: CanonicalTrace) -> list[tuple[str, object]]:
    tracer, _ = setup_unified_tracer(config)
    published: list[tuple[str, object]] = []
    adapter = UnifiedFutureAGIAdapter(config, tracer=tracer, exporter=exporter)  # type: ignore[arg-type]
    adapter.emit(trace, None, lambda span_id, context: published.append((span_id, context)))
    return published


def _ids(spans: list[ReadableSpan]) -> list[tuple[int, int, int | None]]:
    return [(s.context.trace_id, s.context.span_id, s.parent.span_id if s.parent else None) for s in spans]


def test_one_exported_span_per_canonical_span_in_order_with_parents():
    exporter = _Exporter()
    published = _emit(_config(), exporter, _workflow_trace())
    names = [s.name for s in exporter.spans]
    assert names == ["chain_message-1", "chain_run-1", "llm_node-llm", "tool_node-tool"]
    by_name = {s.name: s for s in exporter.spans}
    assert len({s.context.trace_id for s in exporter.spans}) == 1
    assert by_name["chain_message-1"].parent is None
    assert by_name["chain_run-1"].parent.span_id == by_name["chain_message-1"].context.span_id
    assert by_name["llm_node-llm"].parent.span_id == by_name["chain_run-1"].context.span_id
    assert by_name["tool_node-tool"].parent.span_id == by_name["chain_run-1"].context.span_id
    assert [span_id for span_id, _ in published] == ["message-1", "node-tool"]


def test_failed_node_is_error_with_its_message_when_content_is_on():
    exporter = _Exporter()
    _emit(_config(), exporter, _workflow_trace())
    tool = next(s for s in exporter.spans if s.name == "tool_node-tool")
    assert tool.status.status_code is StatusCode.ERROR
    assert MARKER in (tool.status.description or "")
    assert any(MARKER in str(event.attributes) for event in tool.events)


def test_redact_keeps_error_status_and_exports_no_content_or_error_text():
    exporter = _Exporter()
    _emit(_config(redact=True), exporter, _workflow_trace())
    tool = next(s for s in exporter.spans if s.name == "tool_node-tool")
    assert tool.status.status_code is StatusCode.ERROR
    assert tool.status.description is None
    assert not tool.events
    for span in exporter.spans:
        assert MARKER not in repr(dict(span.attributes or {}))
        assert MARKER not in repr(span.events)
        assert span.attributes["dify.redacted"] is True


def test_retry_after_a_partial_export_reuses_the_same_ids():
    """Dify retries the whole dispatch; the spans accepted before the failure must not get new ids."""
    clean = _Exporter()
    _emit(_config(), clean, _workflow_trace())

    flaky = _Exporter(fail_on=3)
    with pytest.raises(RetryableTraceDispatchError):
        _emit(_config(), flaky, _workflow_trace())
    accepted_before_failure = _ids(flaky.spans)
    assert len(accepted_before_failure) == 2

    retry = _Exporter()
    _emit(_config(), retry, _workflow_trace())
    assert _ids(retry.spans) == _ids(clean.spans)
    assert accepted_before_failure == _ids(clean.spans)[:2]


def test_different_runs_get_different_trace_ids():
    first, second = _Exporter(), _Exporter()
    _emit(_config(), first, _workflow_trace())
    other = _workflow_trace().model_copy(update={"trace_id": "run-2"})
    _emit(_config(), second, other)
    assert first.spans[0].context.trace_id != second.spans[0].context.trace_id


def test_exporter_never_compresses_even_when_the_env_asks_for_gzip(monkeypatch: pytest.MonkeyPatch):
    """fi-collector rejects gzip bodies; Dify operators may set OTEL_* compression for Dify's own telemetry."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_COMPRESSION", "gzip")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_COMPRESSION", "gzip")
    _, exporter = setup_unified_tracer(_config())
    assert exporter._compression is Compression.NoCompression
    assert "Content-Encoding" not in exporter._session.headers


def test_resource_carries_the_project_and_the_dify_release_only():
    tracer, _ = setup_unified_tracer(_config())
    exporter = _Exporter()
    UnifiedFutureAGIAdapter(_config(), tracer=tracer, exporter=exporter).emit(  # type: ignore[arg-type]
        _workflow_trace(), None, lambda *_: None
    )
    resource = dict(exporter.spans[0].resource.attributes)
    assert resource == {"project_name": "proj", "project_type": "observe", "dify.version": dify_config.project.version}
