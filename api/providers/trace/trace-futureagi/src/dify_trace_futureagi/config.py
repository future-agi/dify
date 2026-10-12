from pydantic import ValidationInfo, field_validator

from core.ops.entities.config_entity import BaseTracingConfig
from core.ops.utils import validate_url_with_path
from dify_trace_futureagi.defaults import DEFAULT_ENDPOINT


class FutureAGIConfig(BaseTracingConfig):
    """Future AGI tracing config.

    api_key and secret_key are encrypted at rest by OpsTraceManager
    (secret_keys). project, endpoint and redact are stored in clear
    (other_keys). project is the Future AGI project name, not a Dify
    workspace name.
    """

    api_key: str | None = None
    secret_key: str | None = None
    project: str | None = None
    endpoint: str = DEFAULT_ENDPOINT
    redact: bool = False

    @field_validator("project")
    @classmethod
    def project_validator(cls, v, info: ValidationInfo):
        return cls.validate_project_field(v, "default")

    @field_validator("endpoint")
    @classmethod
    def endpoint_validator(cls, v, info: ValidationInfo):
        return validate_url_with_path(v, DEFAULT_ENDPOINT)

    @field_validator("redact", mode="before")
    @classmethod
    def redact_validator(cls, v: object) -> object:
        # OpsTraceManager stores a missing other_key as "", which is not a bool.
        return False if v is None or v == "" else v
