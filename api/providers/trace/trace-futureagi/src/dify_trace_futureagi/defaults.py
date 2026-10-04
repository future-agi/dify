"""Future AGI endpoint and auth defaults, with no Dify runtime import.

The Pydantic config in config.py subclasses Dify's BaseTracingConfig, which
imports the whole API app. These defaults are the same values, kept here so
the mapper and exporter can be tested without that import.
"""

DEFAULT_ENDPOINT = "https://api.futureagi.com"
COLLECTOR_PATH = "/tracer/v1/traces"
