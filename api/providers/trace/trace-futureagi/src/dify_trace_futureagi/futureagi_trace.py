"""Legacy Future AGI tracer for tasks that still enter through OpsTraceManager.

The unified adapter owns span mapping. This class exists so the legacy
registry can resolve the provider; trace() does not raise into the chat
request.
"""

import logging

from core.ops.base_trace_instance import BaseTraceInstance
from core.ops.entities.trace_entity import BaseTraceInfo
from dify_trace_futureagi.config import FutureAGIConfig

logger = logging.getLogger(__name__)


class FutureAGIDataTrace(BaseTraceInstance):
    def __init__(self, config: FutureAGIConfig) -> None:
        super().__init__(config)
        self._config = config

    def trace(self, trace_info: BaseTraceInfo) -> None:
        logger.debug("Future AGI legacy trace received; unified adapter owns export")
