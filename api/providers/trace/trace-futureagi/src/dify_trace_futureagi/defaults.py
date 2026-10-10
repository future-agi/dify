"""Future AGI endpoint and auth defaults, with no Dify runtime import.

The Pydantic config in config.py subclasses Dify's BaseTracingConfig, which
imports the whole API app. These defaults are the same values, kept here so
the mapper and exporter can be tested without that import.
"""

DEFAULT_ENDPOINT = "https://api.futureagi.com"
COLLECTOR_PATH = "/tracer/v1/traces"


def collector_endpoint(endpoint: str) -> str:
    """fi-collector serves OTLP/HTTP at /tracer/v1/traces. Do not append it twice."""
    trimmed = endpoint.rstrip("/")
    suffix = COLLECTOR_PATH.rstrip("/")
    if trimmed.endswith(suffix):
        return trimmed
    return f"{trimmed}{COLLECTOR_PATH}"
