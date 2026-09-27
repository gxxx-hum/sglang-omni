# SPDX-License-Identifier: Apache-2.0
"""Prometheus metrics for pipeline requests, stages, and transfers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from sglang_omni.metrics.types import (
    FAST_BUCKETS_S,
    LATENCY_BUCKETS_S,
    ORPHANED_STATE_TTL_NS,
    REQUEST_STATUSES,
    TERMINAL_GRACE_NS,
    MetricDetails,
    RuntimeMetricEvent,
)


@dataclass(kw_only=True)
class TransferState:
    sent: deque[int] = field(default_factory=deque)
    received: deque[int] = field(default_factory=deque)

    @property
    def latest_ns(self) -> int:
        timestamps = [*self.sent, *self.received]
        return max(timestamps) if timestamps else 0


class PipelineMetrics:
    """Aggregate pipeline lifecycle, stage, and transfer events."""

    def __init__(self, registry: CollectorRegistry) -> None:
        self.requests = Counter(
            "sglang_omni:requests_total",
            "Generation requests by outcome.",
            ["status"],
            registry=registry,
        )
        for status in REQUEST_STATUSES:
            self.requests.labels(status).inc(0)
        self.running = Gauge(
            "sglang_omni:requests_running",
            "Accepted generation requests not yet terminal.",
            registry=registry,
        )
        self.running.set(0)
        self.request_e2e = Histogram(
            "sglang_omni:request_e2e_latency_seconds",
            "Pipeline admission to completion latency.",
            buckets=LATENCY_BUCKETS_S,
            registry=registry,
        )
        self.waiting = Gauge(
            "sglang_omni:requests_waiting",
            "Requests waiting in a stage scheduler.",
            ["stage"],
            registry=registry,
        )
        self.stage_latency = Histogram(
            "sglang_omni:stage_latency_seconds",
            "Time spent in a pipeline stage or phase.",
            ["stage", "phase"],
            buckets=LATENCY_BUCKETS_S,
            registry=registry,
        )
        self.transfer_latency = Histogram(
            "sglang_omni:transfer_latency_seconds",
            "Time from a stage send to the next stage receiving payload.",
            ["from_stage", "to_stage"],
            buckets=FAST_BUCKETS_S,
            registry=registry,
        )
        self.execution = Counter(
            "sglang_omni:execution_total",
            "Code2Wav graph hits and eager fallbacks.",
            ["stage", "kind"],
            registry=registry,
        )
        self.admitted_ns: dict[str, int] = {}
        self.terminal_ns: dict[str, int] = {}
        self.waiting_since_ns: dict[tuple[str, str], int] = {}
        self.phase_started_ns: dict[tuple[str, str, str], int] = {}
        self.transfers: dict[tuple[str, str, str], TransferState] = {}

    def record(
        self,
        event_name: RuntimeMetricEvent,
        request_id: str,
        stage: str,
        timestamp_ns: int,
        details: MetricDetails,
    ) -> None:
        if event_name == "request_rejected":
            self.requests.labels("rejected").inc()
        elif event_name == "request_admission":
            if request_id not in self.admitted_ns:
                self.terminal_ns.pop(request_id, None)
                self.admitted_ns[request_id] = timestamp_ns
                self.requests.labels("accepted").inc()
                self.running.inc()
        elif event_name in {"request_completed", "request_failed", "request_cancelled"}:
            self.record_terminal(event_name, request_id, timestamp_ns)
        elif event_name == "scheduler_queue_enter":
            if request_id not in self.terminal_ns:
                key = (request_id, stage)
                if key not in self.waiting_since_ns:
                    self.waiting_since_ns[key] = timestamp_ns
                    self.waiting.labels(stage).inc()
        elif event_name == "scheduler_prefill_start":
            key = (request_id, stage)
            queued_ns = self.waiting_since_ns.pop(key, None)
            if queued_ns is not None:
                self.waiting.labels(stage).dec()
                self.stage_latency.labels(stage, "queue_wait").observe(
                    max(0.0, (timestamp_ns - queued_ns) / 1e9)
                )
            self.phase_started_ns[(request_id, stage, "prefill")] = timestamp_ns
        elif event_name == "scheduler_prefill_end":
            start_ns = self.phase_started_ns.pop((request_id, stage, "prefill"), None)
            if start_ns is not None:
                self.stage_latency.labels(stage, "prefill").observe(
                    max(0.0, (timestamp_ns - start_ns) / 1e9)
                )
            self.phase_started_ns[(request_id, stage, "decode")] = timestamp_ns
        elif event_name == "stage_dispatch":
            self.phase_started_ns[(request_id, stage, "stage")] = timestamp_ns
        elif event_name == "stage_complete":
            self.record_stage_complete(request_id, stage, timestamp_ns)
        elif event_name in {"preprocess_start", "code2wav_decode_start"}:
            phase = "preprocessing" if event_name == "preprocess_start" else "code2wav"
            self.phase_started_ns[(request_id, stage, phase)] = timestamp_ns
        elif event_name in {"preprocess_end", "code2wav_decode_end"}:
            self.record_phase_end(event_name, request_id, stage, timestamp_ns, details)
        elif event_name in {"stage_hop_sent", "stage_input_received"}:
            self.record_transfer(event_name, request_id, stage, timestamp_ns, details)

    def record_terminal(
        self, event_name: RuntimeMetricEvent, request_id: str, timestamp_ns: int
    ) -> None:
        start_ns = self.admitted_ns.pop(request_id, None)
        if start_ns is not None:
            self.terminal_ns[request_id] = timestamp_ns
            if event_name == "request_completed":
                self.request_e2e.observe(max(0.0, (timestamp_ns - start_ns) / 1e9))
            self.running.dec()
            self.requests.labels(event_name.removeprefix("request_")).inc()
            self.clear_request_state(request_id)

    def clear_request_state(self, request_id: str) -> None:
        for key in [key for key in self.waiting_since_ns if key[0] == request_id]:
            self.waiting.labels(key[1]).dec()
            self.waiting_since_ns.pop(key)
        for key in [key for key in self.phase_started_ns if key[0] == request_id]:
            self.phase_started_ns.pop(key)
        for key in [key for key in self.transfers if key[0] == request_id]:
            self.transfers.pop(key)

    def record_stage_complete(
        self, request_id: str, stage: str, timestamp_ns: int
    ) -> None:
        start_ns = self.phase_started_ns.pop((request_id, stage, "stage"), None)
        if start_ns is not None:
            self.stage_latency.labels(stage, "stage").observe(
                max(0.0, (timestamp_ns - start_ns) / 1e9)
            )
        decode_ns = self.phase_started_ns.pop((request_id, stage, "decode"), None)
        if decode_ns is not None:
            self.stage_latency.labels(stage, "decode").observe(
                max(0.0, (timestamp_ns - decode_ns) / 1e9)
            )

    def record_phase_end(
        self,
        event_name: RuntimeMetricEvent,
        request_id: str,
        stage: str,
        timestamp_ns: int,
        details: MetricDetails,
    ) -> None:
        phase = "preprocessing" if event_name == "preprocess_end" else "code2wav"
        start_ns = self.phase_started_ns.pop((request_id, stage, phase), None)
        if start_ns is not None:
            self.stage_latency.labels(stage, phase).observe(
                max(0.0, (timestamp_ns - start_ns) / 1e9)
            )
        if event_name == "code2wav_decode_end":
            if "graph" in str(details.get("execution_mode", "")):
                self.execution.labels(stage, "graph_hit").inc()
            if details.get("fallback_reason") is not None:
                self.execution.labels(stage, "eager_fallback").inc()

    def record_transfer(
        self,
        event_name: RuntimeMetricEvent,
        request_id: str,
        stage: str,
        timestamp_ns: int,
        details: MetricDetails,
    ) -> None:
        if event_name == "stage_hop_sent":
            from_stage = stage
            to_stage = details.get("to_stage")
        else:
            from_stage = details.get("from_stage")
            to_stage = stage
        if (
            isinstance(from_stage, str)
            and isinstance(to_stage, str)
            and from_stage != "coordinator"
        ):
            key = (request_id, from_stage, to_stage)
            transfer = self.transfers.setdefault(key, TransferState())
            if event_name == "stage_hop_sent":
                transfer.sent.append(timestamp_ns)
            else:
                transfer.received.append(timestamp_ns)
            while transfer.sent and transfer.received:
                self.transfer_latency.labels(from_stage, to_stage).observe(
                    max(
                        0.0,
                        (transfer.received.popleft() - transfer.sent.popleft()) / 1e9,
                    )
                )
            if not transfer.sent and not transfer.received:
                self.transfers.pop(key)

    def prune(self, now_ns: int) -> None:
        expired_terminal = {
            request_id
            for request_id, timestamp_ns in self.terminal_ns.items()
            if now_ns - timestamp_ns > TERMINAL_GRACE_NS
        }
        for request_id in expired_terminal:
            self.terminal_ns.pop(request_id)
            self.clear_request_state(request_id)

        expired_requests = [
            request_id
            for request_id, timestamp_ns in self.admitted_ns.items()
            if now_ns - timestamp_ns > ORPHANED_STATE_TTL_NS
        ]
        for request_id in expired_requests:
            self.admitted_ns.pop(request_id)
            self.clear_request_state(request_id)
        if expired_requests:
            self.running.dec(len(expired_requests))

        for key, timestamp_ns in list(self.waiting_since_ns.items()):
            if now_ns - timestamp_ns > ORPHANED_STATE_TTL_NS:
                self.waiting.labels(key[1]).dec()
                self.waiting_since_ns.pop(key)
        for key, timestamp_ns in list(self.phase_started_ns.items()):
            if now_ns - timestamp_ns > ORPHANED_STATE_TTL_NS:
                self.phase_started_ns.pop(key)
        for key, transfer in list(self.transfers.items()):
            if now_ns - transfer.latest_ns > ORPHANED_STATE_TTL_NS:
                self.transfers.pop(key)
