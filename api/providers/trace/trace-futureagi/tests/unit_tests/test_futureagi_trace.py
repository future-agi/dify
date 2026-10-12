"""Legacy-registry entry point: credential check on save, and trace dispatch with the unified flag off."""

import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer

import dify_trace_futureagi.unified_trace as unified_trace_module
import pytest
from dify_trace_futureagi.config import FutureAGIConfig
from dify_trace_futureagi.futureagi_trace import FutureAGIDataTrace

from core.ops.ops_trace_manager import OpsTraceManager


class _Collector(BaseHTTPRequestHandler):
    status = 200
    requests: list[tuple[str, dict[str, str], bytes]] = []

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        _Collector.requests.append((self.path, dict(self.headers), body))
        self.send_response(_Collector.status)
        self.end_headers()

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def collector() -> Iterator[str]:
    _Collector.status = 200
    _Collector.requests = []
    server = HTTPServer(("127.0.0.1", 0), _Collector)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def _config(endpoint: str) -> FutureAGIConfig:
    return FutureAGIConfig(api_key="key-a", secret_key="secret-a", project="proj", endpoint=endpoint)


def test_api_check_posts_an_empty_export_with_both_headers(collector: str) -> None:
    assert FutureAGIDataTrace(_config(collector)).api_check() is True
    path, headers, body = _Collector.requests[0]
    assert path == "/tracer/v1/traces"
    assert headers["X-Api-Key"] == "key-a"
    assert headers["X-Secret-Key"] == "secret-a"
    assert body == b""


@pytest.mark.parametrize("status", [401, 403])
def test_api_check_rejects_bad_keys_without_echoing_them(collector: str, status: int) -> None:
    _Collector.status = status
    with pytest.raises(ValueError, match="rejected") as error:
        FutureAGIDataTrace(_config(collector)).api_check()
    assert "key-a" not in str(error.value)
    assert "secret-a" not in str(error.value)


def test_api_check_reports_server_errors(collector: str) -> None:
    _Collector.status = 503
    with pytest.raises(ValueError, match="503"):
        FutureAGIDataTrace(_config(collector)).api_check()


def test_api_check_reports_an_unreachable_endpoint() -> None:
    with pytest.raises(ValueError, match="could not reach"):
        FutureAGIDataTrace(_config("http://127.0.0.1:9")).api_check()


def test_saving_the_config_runs_the_check(collector: str) -> None:
    """The console save path calls check_trace_config_is_effective on the legacy registry."""
    config = {"api_key": "key-a", "secret_key": "secret-a", "project": "proj", "endpoint": collector}
    assert OpsTraceManager.check_trace_config_is_effective(config, "futureagi") is True


def test_legacy_trace_delegates_to_the_unified_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[object] = []

    class _Unified:
        def __init__(self, config: FutureAGIConfig) -> None:
            self.config = config

        def trace(self, trace_info: object) -> None:
            received.append(trace_info)

    monkeypatch.setattr(unified_trace_module, "UnifiedFutureAGITrace", _Unified)
    instance = FutureAGIDataTrace(_config("https://api.futureagi.com"))
    sentinel_a, sentinel_b = object(), object()
    instance.trace(sentinel_a)  # type: ignore[arg-type]
    instance.trace(sentinel_b)  # type: ignore[arg-type]
    assert received == [sentinel_a, sentinel_b]
