from orchestration.observability import (
    CounterRegistry,
    ORCHESTRATION_REQUESTS_TOTAL,
    StructuredLogger,
)


def test_counter_increments():
    before = ORCHESTRATION_REQUESTS_TOTAL.get(target_family="gpt")
    ORCHESTRATION_REQUESTS_TOTAL.inc(target_family="gpt")
    after = ORCHESTRATION_REQUESTS_TOTAL.get(target_family="gpt")
    assert after == before + 1


def test_registry_renders_prometheus():
    text = CounterRegistry.get().render_prometheus()
    assert "orchestration_requests_total" in text
    assert "orchestration_latency_seconds" in text


def test_structured_logger_emits(capsys):
    log = StructuredLogger("test")
    log.info(event="hello", trace_id="t-1", extra="value")
    out = capsys.readouterr().out
    assert '"event": "hello"' in out
    assert '"trace_id": "t-1"' in out
    assert '"component": "test"' in out
