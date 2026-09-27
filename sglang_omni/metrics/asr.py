# SPDX-License-Identifier: Apache-2.0
"""Prometheus metrics for speech recognition requests."""

from __future__ import annotations

from dataclasses import dataclass, field

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from sglang_omni.metrics.types import (
    FAST_BUCKETS_S,
    LATENCY_BUCKETS_S,
    ORPHANED_STATE_TTL_NS,
    REQUEST_STATUSES,
    RTF_BUCKETS,
    TERMINAL_STATUSES,
    MetricDetails,
    RuntimeMetricEvent,
)


@dataclass(kw_only=True)
class AsrState:
    started_ns: int
    first_text_ns: int | None = None
    last_text_ns: int | None = None
    text_intervals_s: list[float] = field(default_factory=list)


class AsrMetrics:
    """Aggregate ASR request and streaming text events."""

    def __init__(self, registry: CollectorRegistry) -> None:
        self.requests = Counter(
            "sglang_omni:asr_requests_total",
            "ASR requests by outcome.",
            ["status"],
            registry=registry,
        )
        for status in REQUEST_STATUSES:
            self.requests.labels(status).inc(0)
        self.running = Gauge(
            "sglang_omni:asr_requests_running",
            "ASR HTTP requests currently in progress.",
            registry=registry,
        )
        self.running.set(0)
        self.request_latency = Histogram(
            "sglang_omni:asr_request_latency_seconds",
            "ASR HTTP request latency for completed responses.",
            buckets=LATENCY_BUCKETS_S,
            registry=registry,
        )
        self.input_duration = Histogram(
            "sglang_omni:asr_input_duration_seconds",
            "Input audio duration for completed ASR requests.",
            buckets=LATENCY_BUCKETS_S,
            registry=registry,
        )
        self.engine_latency = Histogram(
            "sglang_omni:asr_engine_latency_seconds",
            "Summed engine execution time for a completed ASR request.",
            buckets=LATENCY_BUCKETS_S,
            registry=registry,
        )
        self.engine_rtf = Histogram(
            "sglang_omni:asr_engine_rtf",
            "ASR engine execution time divided by input audio duration.",
            buckets=RTF_BUCKETS,
            registry=registry,
        )
        self.request_rtf = Histogram(
            "sglang_omni:asr_request_rtf",
            "ASR HTTP request latency divided by input audio duration.",
            buckets=RTF_BUCKETS,
            registry=registry,
        )
        self.text_ttft = Histogram(
            "sglang_omni:asr_text_ttft_seconds",
            "Time from ASR HTTP request arrival to first transcript delta.",
            buckets=LATENCY_BUCKETS_S,
            registry=registry,
        )
        self.text_interval = Histogram(
            "sglang_omni:asr_text_chunk_interval_seconds",
            "Wall time between consecutive ASR transcript deltas.",
            buckets=FAST_BUCKETS_S,
            registry=registry,
        )
        self.active_requests: dict[str, AsrState] = {}

    def record(
        self,
        event_name: RuntimeMetricEvent,
        request_id: str,
        timestamp_ns: int,
        details: MetricDetails,
    ) -> None:
        if event_name == "asr_request_start":
            if request_id not in self.active_requests:
                self.active_requests[request_id] = AsrState(started_ns=timestamp_ns)
                self.running.inc()
        elif event_name == "asr_text_chunk":
            state = self.active_requests.get(request_id)
            if state is not None:
                if state.first_text_ns is None:
                    state.first_text_ns = timestamp_ns
                elif state.last_text_ns is not None:
                    state.text_intervals_s.append(
                        max(0.0, (timestamp_ns - state.last_text_ns) / 1e9)
                    )
                state.last_text_ns = timestamp_ns
        elif event_name in {"asr_response_done", "asr_response_aborted"}:
            state = self.active_requests.pop(request_id, None)
            if state is not None:
                self.running.dec()
                status = details.get("status")
                if status in TERMINAL_STATUSES:
                    self.requests.labels(status).inc()
                    if status != "rejected":
                        self.requests.labels("accepted").inc()
                    if event_name == "asr_response_done" and status == "completed":
                        self.record_completed(state, timestamp_ns, details)

    def record_completed(
        self, state: AsrState, timestamp_ns: int, details: MetricDetails
    ) -> None:
        elapsed_s = max(0.0, (timestamp_ns - state.started_ns) / 1e9)
        self.request_latency.observe(elapsed_s)
        if state.first_text_ns is not None:
            self.text_ttft.observe(
                max(0.0, (state.first_text_ns - state.started_ns) / 1e9)
            )
            for interval_s in state.text_intervals_s:
                self.text_interval.observe(interval_s)
        duration_s = details.get("audio_duration_s")
        if isinstance(duration_s, (int, float)) and duration_s > 0:
            self.input_duration.observe(duration_s)
            self.request_rtf.observe(elapsed_s / duration_s)
            engine_time_s = details.get("engine_time_s")
            if isinstance(engine_time_s, (int, float)) and engine_time_s >= 0:
                self.engine_latency.observe(engine_time_s)
                self.engine_rtf.observe(engine_time_s / duration_s)

    def prune(self, now_ns: int) -> None:
        expired = [
            request_id
            for request_id, state in self.active_requests.items()
            if now_ns - state.started_ns > ORPHANED_STATE_TTL_NS
        ]
        for request_id in expired:
            self.active_requests.pop(request_id)
        if expired:
            self.running.dec(len(expired))
