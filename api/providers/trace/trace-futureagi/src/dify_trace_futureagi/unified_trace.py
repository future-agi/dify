"""Future AGI adapter for Dify's unified tracing runtime.

Exports OTLP/HTTP with the headers and resource attributes fi-collector
accepts. Does not retarget the Phoenix provider and does not import traceAI.
"""

from typing import cast

from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import Span, Status, set_span_in_context
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from core.ops.exceptions import InvalidTraceParentContextError, RetryableTraceDispatchError
from core.ops.unified_trace.entities import CanonicalSpan, CanonicalTrace
from core.ops.unified_trace.parent_context import (
    ParentContextCoordinator,
    ParentResolution,
    ParentResolutionKind,
    ProviderParentContext,
    destination_scope,
    resolve_parent_destination,
)
from core.ops.unified_trace.provider import ParentContextPublisher, UnifiedTraceInstance
from core.ops.unified_trace.trace_builder import CanonicalTraceBuilder, RepositoryWorkflowExecutionLoader
from dify_trace_futureagi.config import FutureAGIConfig
from dify_trace_futureagi.defaults import collector_endpoint
from dify_trace_futureagi.mapping import (
    auth_headers,
    nanos,
    resource_attributes,
    span_attributes,
    span_status,
    trace_id_from_message_id,
)
from dify_trace_futureagi.span_input import SpanInput, SpanKind, SpanStatus, TraceInput
from extensions.ext_redis import redis_client

EXPORT_TIMEOUT_SECONDS = 10


def _span_input(canonical_span: CanonicalSpan) -> SpanInput:
    return SpanInput(
        id=canonical_span.id,
        parent_id=canonical_span.parent_id,
        name=canonical_span.name,
        kind=SpanKind(canonical_span.kind.value),
        start_time=canonical_span.start_time,
        end_time=canonical_span.end_time,
        inputs=canonical_span.inputs,
        outputs=canonical_span.outputs,
        status=SpanStatus(canonical_span.status.value),
        error=canonical_span.error,
        metadata=dict(canonical_span.metadata),
        synthetic=canonical_span.synthetic,
    )


def _trace_input(trace: CanonicalTrace) -> TraceInput:
    return TraceInput(
        trace_id=trace.trace_id,
        session_id=trace.session_id,
        spans=tuple(_span_input(span) for span in trace.spans),
    )


def setup_unified_tracer(config: FutureAGIConfig) -> tuple[trace_sdk.Tracer, OTLPSpanExporter]:
    """An isolated tracer. Never calls set_tracer_provider, so two apps stay separate."""
    exporter = OTLPSpanExporter(
        endpoint=collector_endpoint(config.endpoint),
        headers=dict(auth_headers(config.api_key, config.secret_key)),
        timeout=EXPORT_TIMEOUT_SECONDS,
    )
    provider = trace_sdk.TracerProvider(resource=Resource(attributes=resource_attributes(config.project or "")))
    return cast(trace_sdk.Tracer, provider.get_tracer(f"unified_futureagi_{config.project}")), exporter


class UnifiedFutureAGIAdapter:
    """Translate a canonical trace into Future AGI spans and export them one at a time."""

    provider_name = "futureagi"

    def __init__(
        self,
        config: FutureAGIConfig,
        tracer: trace_sdk.Tracer | None = None,
        exporter: OTLPSpanExporter | None = None,
    ) -> None:
        self._config = config
        if tracer is None or exporter is None:
            tracer, exporter = setup_unified_tracer(config)
        self._tracer = tracer
        self._exporter = exporter
        self._propagator = TraceContextTextMapPropagator()
        self._scope = destination_scope(self.provider_name, config.endpoint, config.project or "")

    @property
    def scope(self) -> str:
        return self._scope

    def _root_context(self, parent: ParentResolution | None) -> Context | None:
        if parent is None or parent.kind is ParentResolutionKind.LINKED_ROOT or parent.context is None:
            return None
        traceparent = parent.context.provider_context.get("traceparent")
        if not traceparent:
            raise InvalidTraceParentContextError("Future AGI parent context is missing traceparent")
        return self._propagator.extract(carrier={"traceparent": traceparent})

    def emit(
        self,
        trace: CanonicalTrace,
        parent: ParentResolution | None,
        publish_parent_context: ParentContextPublisher,
    ) -> None:
        span_by_id: dict[str, Span] = {}
        root_context = self._root_context(parent)
        for canonical_span in trace.spans:
            self._emit_span(canonical_span, trace, span_by_id, root_context, publish_parent_context)

    def _emit_span(
        self,
        canonical_span: CanonicalSpan,
        trace: CanonicalTrace,
        span_by_id: dict[str, Span],
        root_context: Context | None,
        publish_parent_context: ParentContextPublisher,
    ) -> None:
        local_parent = span_by_id.get(canonical_span.parent_id or "")
        context = set_span_in_context(local_parent) if local_parent is not None else root_context
        span = self._tracer.start_span(
            name=canonical_span.name,
            context=context,
            attributes=span_attributes(_span_input(canonical_span), _trace_input(trace), redact=self._config.redact),
            start_time=nanos(canonical_span.start_time),
        )
        span_by_id[canonical_span.id] = span
        status_code, description = span_status(_span_input(canonical_span))
        span.set_status(Status(status_code, description))
        if description:
            span.record_exception(RuntimeError(description))
        provider_parent_context = self._parent_context(canonical_span, trace, span)
        span.end(end_time=nanos(canonical_span.end_time))
        self._export(span, canonical_span.id)
        if provider_parent_context is not None:
            publish_parent_context(canonical_span.id, provider_parent_context)

    def _parent_context(
        self, canonical_span: CanonicalSpan, trace: CanonicalTrace, span: Span
    ) -> ProviderParentContext | None:
        if not (canonical_span.can_parent_workflow or canonical_span.publishes_parent_context):
            return None
        carrier: dict[str, str] = {}
        self._propagator.inject(carrier, context=set_span_in_context(span))
        return ProviderParentContext(
            provider=self.provider_name,
            scope=self.scope,
            trace_id=trace.trace_id,
            parent_id=canonical_span.id,
            provider_context=carrier,
        )

    def _export(self, span: Span, canonical_span_id: str) -> None:
        try:
            export_result = self._exporter.export((cast(trace_sdk.ReadableSpan, span),))
        except Exception as error:
            raise RetryableTraceDispatchError("Future AGI span export failed") from error
        if export_result is not SpanExportResult.SUCCESS:
            raise RetryableTraceDispatchError(f"Future AGI span export failed: canonical_span_id={canonical_span_id}")


class UnifiedFutureAGITrace(UnifiedTraceInstance):
    """Unified Future AGI trace instance selected by the unified registry."""

    def __init__(self, config: FutureAGIConfig) -> None:
        super().__init__(
            config,
            builder=CanonicalTraceBuilder(RepositoryWorkflowExecutionLoader(self.get_service_account_with_tenant)),
            adapter=UnifiedFutureAGIAdapter(config),
            coordinator=ParentContextCoordinator(redis_client, resolve_parent_destination),
        )


__all__ = [
    "UnifiedFutureAGIAdapter",
    "UnifiedFutureAGITrace",
    "collector_endpoint",
    "setup_unified_tracer",
    "trace_id_from_message_id",
]
