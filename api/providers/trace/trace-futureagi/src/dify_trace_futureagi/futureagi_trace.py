"""Future AGI tracer resolved through the legacy registry (OpsTraceProviderConfigMap).

Dify reads this registry to verify credentials when a user saves the tracing
config, and to dispatch traces whenever OPS_TRACE_UNIFIED_ENABLED is off, which
is the default. Future AGI has no separate legacy mapper: trace() hands the
trace to the unified adapter, so both modes export the same spans.
"""

import logging
from typing import TYPE_CHECKING

import requests

from core.ops.base_trace_instance import BaseTraceInstance
from core.ops.entities.trace_entity import BaseTraceInfo
from dify_trace_futureagi.config import FutureAGIConfig
from dify_trace_futureagi.defaults import collector_endpoint
from dify_trace_futureagi.mapping import auth_headers

if TYPE_CHECKING:
    from dify_trace_futureagi.unified_trace import UnifiedFutureAGITrace

logger = logging.getLogger(__name__)

API_CHECK_TIMEOUT_SECONDS = 10


class FutureAGIDataTrace(BaseTraceInstance):
    def __init__(self, config: FutureAGIConfig) -> None:
        super().__init__(config)
        self._config = config
        self._unified: UnifiedFutureAGITrace | None = None

    def trace(self, trace_info: BaseTraceInfo) -> None:
        if self._unified is None:
            from dify_trace_futureagi.unified_trace import UnifiedFutureAGITrace

            self._unified = UnifiedFutureAGITrace(self._config)
        self._unified.trace(trace_info)

    def api_check(self) -> bool:
        """Post an empty OTLP export. fi-collector authenticates it and stores nothing."""
        endpoint = collector_endpoint(self._config.endpoint)
        try:
            response = requests.post(
                endpoint,
                data=b"",
                headers={
                    "Content-Type": "application/x-protobuf",
                    **auth_headers(self._config.api_key, self._config.secret_key),
                },
                timeout=API_CHECK_TIMEOUT_SECONDS,
            )
        except requests.RequestException as error:
            logger.info("[Future AGI] API check could not reach %s", endpoint, exc_info=True)
            raise ValueError(f"[Future AGI] API check failed: could not reach {endpoint}") from error
        if response.status_code in (401, 403):
            raise ValueError("[Future AGI] API check failed: the API key or secret key was rejected")
        if not response.ok:
            raise ValueError(f"[Future AGI] API check failed: {endpoint} returned HTTP {response.status_code}")
        return True
