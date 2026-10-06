"""Bounded request scheduling and measured concurrency tuning for local llama-server.

The server allocates slots/KV memory at startup. This controls active requests within
those slots; free VRAM is a guard, not a formula for estimating slot capacity.
"""
from dataclasses import dataclass
import logging
import subprocess
import time

from .config import AdaptiveConcurrencySettings

logger = logging.getLogger(__name__)


@dataclass
class RequestMeasurement:
    network_requests: int = 0
    started: float | None = None
    success: bool = False
    pressure: bool = False
    output_tokens: int = 0


class GPUMemoryProbe:
    def __init__(self, gpu_index: int, poll_seconds: float):
        self.gpu_index = gpu_index
        self.poll_seconds = poll_seconds
        self.last_read = -float("inf")
        self.value = None

    def read(self) -> dict | None:
        now = time.perf_counter()
        if now - self.last_read < self.poll_seconds:
            return self.value
        self.last_read = now
        try:
            result = subprocess.run(
                ["nvidia-smi", f"--id={self.gpu_index}",
                 "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            total, free = [int(v.strip()) for v in result.stdout.strip().split(",")]
            if total <= 0 or not 0 <= free <= total:
                raise ValueError("invalid GPU memory reading")
            self.value = {"gpu_index": self.gpu_index, "total_mb": total,
                          "free_mb": free, "used_mb": total - free}
        except (OSError, ValueError, subprocess.SubprocessError):
            self.value = None
        return self.value


class ConcurrencyController:
    """Called by the scheduler thread only; persists its learned limit across days."""

    def __init__(self, maximum: int, adaptive: AdaptiveConcurrencySettings,
                 server_slots: int | None = None, probe=None, clock=time.perf_counter):
        self.settings = adaptive
        self.maximum = min(maximum, server_slots) if server_slots is not None else maximum
        self.server_slots = server_slots
        self.limit = min(adaptive.start, self.maximum) if adaptive.enabled else self.maximum
        self.ceiling = self.maximum
        self.clock = clock
        self.probe = probe or GPUMemoryProbe(adaptive.gpu_index, adaptive.memory_poll_seconds)
        self.memory = None
        self.epoch = 0
        self.warmed_up = False
        self.best_rate = 0.0
        self.best_limit = self.limit
        self.samples = []
        self.window_started = None
        self.history = []
        self.completed_network_tasks = 0
        self.history.append({"reason": "start", "concurrency": self.limit,
                             "maximum": self.maximum, "server_slots": self.server_slots})

    def _change(self, value: int, reason: str, rate: float | None = None):
        old = self.limit
        self.limit = max(1, min(value, self.ceiling))
        self.epoch += 1
        self.samples = []
        self.window_started = None
        event = {"reason": reason, "previous": old, "concurrency": self.limit,
                 "requests_per_second": round(rate, 4) if rate is not None else None,
                 "gpu_memory": self.memory}
        self.history.append(event)
        # Avoid an unbounded audit file during long runs.
        self.history = self.history[-100:]
        logger.info("模型并发 %d → %d：%s；显存余量=%s MiB，吞吐=%s 请求/秒",
                    old, self.limit, reason, self.memory["free_mb"] if self.memory else "未知",
                    event["requests_per_second"])

    def poll_memory(self):
        if not self.settings.enabled:
            return
        self.memory = self.probe.read()
        if self.memory is None:
            if self.limit > 1:
                self._change(1, "gpu_memory_unavailable")
        elif self.memory["free_mb"] < self.settings.min_free_mb and self.limit > 1:
            self._change(self.limit - 1, "low_free_vram")

    def begin_batch(self):
        # Don't include ingestion, alignment, cached days, or export time in throughput.
        self.epoch += 1
        self.samples = []
        self.window_started = None
        self.poll_memory()

    def observe(self, measurement: RequestMeasurement, epoch: int, saturated: bool):
        if measurement.network_requests:
            self.completed_network_tasks += 1
        if not self.settings.enabled or not measurement.network_requests:
            return
        if measurement.pressure:
            # Failed requests/retries are never treated as improved throughput.
            self.ceiling = min(self.ceiling, max(1, self.limit - 1))
            self._change(self.ceiling, "server_pressure")
            return
        if epoch != self.epoch or not measurement.success or not saturated:
            return
        if not self.warmed_up:
            self.warmed_up = True
            self._change(self.limit, "warmup_complete")
            return
        if measurement.started is None:
            return
        self.window_started = min(self.window_started or measurement.started, measurement.started)
        self.samples.append(measurement)
        if len(self.samples) < max(self.settings.sample_size, 2 * self.limit):
            return
        elapsed = max(self.clock() - self.window_started, 1e-6)
        rate = len(self.samples) / elapsed
        self.history.append({"reason": "measurement", "concurrency": self.limit,
                             "successful_tasks": len(self.samples), "seconds": round(elapsed, 3),
                             "requests_per_second": round(rate, 4),
                             "output_tokens_per_second": round(sum(s.output_tokens for s in self.samples) / elapsed, 3),
                             "gpu_memory": self.memory})
        if self.best_rate and self.limit > self.best_limit and rate < self.best_rate * (1 + self.settings.min_gain):
            self.ceiling = min(self.ceiling, self.limit - 1)
            self._change(self.best_limit, "no_throughput_gain", rate)
            return
        if rate > self.best_rate:
            self.best_rate, self.best_limit = rate, self.limit
        enough_memory = (self.memory is not None and
                         self.memory["free_mb"] >= self.settings.min_free_mb + self.settings.growth_headroom_mb)
        if self.limit < self.ceiling and enough_memory:
            self._change(self.limit + 1, "throughput_and_vram_allow_growth", rate)
        else:
            self._change(self.limit, "hold", rate)

    def snapshot(self) -> dict:
        return {"adaptive": self.settings.enabled, "current": self.limit,
                "maximum": self.maximum, "ceiling": self.ceiling,
                "server_slots": self.server_slots, "gpu_memory": self.memory,
                "best_concurrency": self.best_limit,
                "best_requests_per_second": round(self.best_rate, 4),
                "completed_network_tasks": self.completed_network_tasks,
                "history": list(self.history)}
