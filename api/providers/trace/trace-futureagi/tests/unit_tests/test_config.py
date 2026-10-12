import pytest
from dify_trace_futureagi.config import FutureAGIConfig
from dify_trace_futureagi.defaults import DEFAULT_ENDPOINT


@pytest.mark.parametrize("missing", ["", None])
def test_redact_left_out_of_a_saved_config_means_off(missing: str | None) -> None:
    """OpsTraceManager fills an absent other_key with "", so redact must accept it."""
    config = FutureAGIConfig(api_key="key", secret_key="secret", project="proj", endpoint="", redact=missing)
    assert config.redact is False
    assert config.endpoint == DEFAULT_ENDPOINT


def test_redact_keeps_an_explicit_value() -> None:
    assert FutureAGIConfig(redact=True).redact is True
    assert FutureAGIConfig(redact="true").redact is True
    assert FutureAGIConfig(redact=False).redact is False
