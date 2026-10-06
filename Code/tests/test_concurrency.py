import json
import subprocess
from threading import Barrier, Lock
import time

import httpx
import pytest

from fusion_agent.concurrency import ConcurrencyController, GPUMemoryProbe, RequestMeasurement
from fusion_agent.config import AdaptiveConcurrencySettings, LLMSettings
from fusion_agent.llm import LLMClient, LLMError


class FakeClock:
    value = 100.0

    def __call__(self):
        return self.value


class Memory:
    free = 4096

    def read(self):
        return None if self.free is None else {"gpu_index": 0, "total_mb": 16384, "free_mb": self.free}


def controller(maximum=4):
    clock, memory = FakeClock(), Memory()
    result = ConcurrencyController(maximum, AdaptiveConcurrencySettings(enabled=True, sample_size=2),
                                   server_slots=3, probe=memory, clock=clock)
    result.begin_batch()
    result.observe(RequestMeasurement(network_requests=1, started=99, success=True), result.epoch, True)
    return result, clock, memory


def sample_window(result, clock, rate):
    count = max(result.settings.sample_size, 2 * result.limit)
    start = clock.value
    epoch = result.epoch
    for i in range(count):
        clock.value = start + (i + 1) / rate
        result.observe(RequestMeasurement(network_requests=1, started=start, success=True, output_tokens=10),
                       epoch, True)


def test_ramps_one_step_and_returns_to_faster_level():
    result, clock, _ = controller()
    sample_window(result, clock, 1)
    assert result.limit == 2
    sample_window(result, clock, 1.8)
    assert result.limit == 3
    sample_window(result, clock, 1.7)
    assert result.limit == 2
    assert result.ceiling == 2
    assert result.snapshot()["best_concurrency"] == 2
    assert result.history[-1]["reason"] == "no_throughput_gain"


def test_clamps_to_server_slots_even_when_more_memory_available():
    result, clock, _ = controller()
    for rate in [1, 2, 3, 4]:
        sample_window(result, clock, rate)
    assert result.maximum == result.limit == 3


def test_cache_hits_and_short_tail_do_not_train_throughput():
    result, clock, _ = controller()
    for _ in range(20):
        result.observe(RequestMeasurement(), result.epoch, True)
        result.observe(RequestMeasurement(network_requests=1, started=clock(), success=True), result.epoch, False)
    assert result.limit == 1
    assert result.best_rate == 0


def test_memory_guard_and_missing_gpu_decrease_concurrency():
    result, clock, memory = controller()
    sample_window(result, clock, 1)
    memory.free = 800
    result.poll_memory()
    assert result.limit == 1
    sample_window(result, clock, 2)
    assert result.limit == 1  # no growth while below the memory reserve
    memory.free = 4096
    result.poll_memory()
    sample_window(result, clock, 2)
    assert result.limit == 2
    memory.free = None
    result.poll_memory()
    assert result.limit == 1
    assert result.history[-1]["reason"] == "gpu_memory_unavailable"


def test_failed_retry_cannot_raise_concurrency():
    result, clock, _ = controller()
    sample_window(result, clock, 1)
    result.observe(RequestMeasurement(network_requests=2, started=clock(), success=True, pressure=True), result.epoch, True)
    assert result.limit == result.ceiling == 1
    assert result.history[-1]["reason"] == "server_pressure"


def test_old_inflight_results_do_not_contaminate_new_level():
    result, clock, _ = controller()
    old_epoch = result.epoch
    sample_window(result, clock, 1)
    for _ in range(20):
        result.observe(RequestMeasurement(network_requests=1, started=99, success=True), old_epoch, True)
    assert result.limit == 2 and not result.samples


def test_gpu_probe_reads_and_caches_correct_card(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "16384, 12345\n")

    monkeypatch.setattr(subprocess, "run", run)
    probe = GPUMemoryProbe(1, 10)
    assert probe.read()["free_mb"] == 12345
    assert probe.read()["used_mb"] == 4039
    assert len(calls) == 1 and "--id=1" in calls[0]


@pytest.mark.parametrize("error", [FileNotFoundError(), subprocess.TimeoutExpired("nvidia-smi", 5)])
def test_gpu_probe_fails_closed(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(subprocess, "run", fail)
    assert GPUMemoryProbe(0, 2).read() is None


def test_scheduler_runs_overlapping_http_requests_without_exceeding_server_slots(tmp_path):
    barrier, lock = Barrier(2), Lock()
    active = maximum = 0
    settings = LLMSettings(provider="llama_cpp", concurrency=8, cache=False, retries=0)

    def handler(request):
        nonlocal active, maximum
        if request.method == "GET":
            assert request.url.path == "/props"
            return httpx.Response(200, json={"total_slots": 2})
        with lock:
            active += 1
            maximum = max(maximum, active)
        try:
            # A serial implementation fails this barrier; this proves actual HTTP overlap.
            barrier.wait(timeout=5)
            user = json.loads(request.content)["messages"][1]["content"]
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"value": user})}}]})
        finally:
            with lock:
                active -= 1

    client = LLMClient(settings, tmp_path)
    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        results = client.map_tasks(lambda item: client.complete_json("JSON", str(item), {}), range(6))
        assert results == [{"value": str(i)} for i in range(6)]
        assert maximum == 2 and client.calls == 6
        assert client.concurrency_snapshot()["maximum"] == 2
    finally:
        client.close()


def test_strict_failure_stops_submitting_remaining_dataset(tmp_path):
    settings = LLMSettings(provider="llama_cpp", concurrency=1, cache=False, retries=0)
    client = LLMClient(settings, tmp_path)
    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={"total_slots": 4}) if r.method == "GET" else httpx.Response(401)))
    try:
        with pytest.raises(LLMError, match="401"):
            client.map_tasks(lambda item: client.complete_json("JSON", str(item), {}), range(100))
        assert client.calls == 1
    finally:
        client.close()


def test_adaptive_scheduler_ramps_real_requests_and_ignores_cached_articles(tmp_path, monkeypatch):
    monkeypatch.setattr(GPUMemoryProbe, "read", lambda self: {"total_mb": 16384, "free_mb": 4096})
    settings = LLMSettings(provider="llama_cpp", concurrency=4, retries=0,
                           adaptive=AdaptiveConcurrencySettings(enabled=True, sample_size=2))
    lock = Lock()
    active = maximum = 0

    def handler(request):
        nonlocal active, maximum
        if request.method == "GET":
            return httpx.Response(200, json={"total_slots": 2})
        with lock:
            active += 1
            maximum = max(maximum, active)
        try:
            time.sleep(0.02)
            return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}],
                                            "usage": {"completion_tokens": 10}})
        finally:
            with lock:
                active -= 1

    client = LLMClient(settings, tmp_path)
    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        for item in range(3):
            client.complete_json("JSON", str(item), {})
        assert client.map_tasks(lambda item: client.complete_json("JSON", str(item), {}), range(16)) == [{}] * 16
        audit = client.concurrency_snapshot()
        assert maximum == 2
        assert audit["maximum"] == 2 and audit["completed_network_tasks"] == 13
        assert any(e.get("previous") == 1 and e["concurrency"] == 2 for e in audit["history"])
        assert client.cache_hits == 3
    finally:
        client.close()
