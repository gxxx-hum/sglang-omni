# SPDX-License-Identifier: Apache-2.0
"""Prometheus metrics derived from pipeline lifecycle events."""

from __future__ import annotations

import os
from multiprocessing.queues import Queue
from multiprocessing.sharedctypes import Synchronized
from queue import Empty, Full

from prometheus_client import (
    CollectorRegistry,
    Gauge,
    GCCollector,
    PlatformCollector,
    ProcessCollector,
    generate_latest,
    multiprocess,
    values,
)

from sglang_omni.metrics.asr import AsrMetrics
from sglang_omni.metrics.audio import AudioMetrics
from sglang_omni.metrics.pipeline import PipelineMetrics
from sglang_omni.metrics.types import (
    ASR_EVENT_NAMES,
    AUDIO_EVENT_NAMES,
    MetricDetails,
    RuntimeMetricEvent,
)

METRIC_DRAIN_BATCH_SIZE = 1024


def enqueue_metric(
    events: Queue,
    dropped: Synchronized,
    event_name: RuntimeMetricEvent,
    request_id: str,
    stage: str,
    timestamp_ns: int,
    metadata: MetricDetails,
) -> None:
    """Send a worker metric event without blocking model execution."""
    try:
        events.put_nowait((event_name, request_id, stage, timestamp_ns, metadata))
    except Full:
        with dropped.get_lock():
            dropped.value += 1


class RuntimeMetrics:
    """Aggregate API and stage events without request IDs as metric labels."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        ProcessCollector(registry=self.registry)
        PlatformCollector(registry=self.registry)
        GCCollector(registry=self.registry)
        self.pipeline = PipelineMetrics(self.registry)
        self.audio = AudioMetrics(self.registry)
        self.asr = AsrMetrics(self.registry)
        self.dropped_events = Gauge(
            "sglang_omni:metric_events_dropped",
            "Worker metric events dropped because the aggregation queue was full.",
            registry=self.registry,
        )
        self.dropped_events.set(0)

    def record(
        self,
        event_name: RuntimeMetricEvent,
        request_id: str,
        stage: str,
        timestamp_ns: int,
        metadata: MetricDetails | None = None,
    ) -> None:
        details = metadata or {}
        if event_name in ASR_EVENT_NAMES:
            self.asr.record(event_name, request_id, timestamp_ns, details)
        elif event_name in AUDIO_EVENT_NAMES:
            self.audio.record(event_name, request_id, timestamp_ns, details)
        else:
            self.pipeline.record(event_name, request_id, stage, timestamp_ns, details)

    def render(self) -> bytes:
        metrics_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
        if metrics_dir is None:
            output = generate_latest(self.registry)
        else:
            upstream_registry = CollectorRegistry()
            multiprocess.MultiProcessCollector(upstream_registry, path=metrics_dir)
            upstream_metrics = generate_latest(upstream_registry)
            if values.ValueClass.__name__ == "MmapedValue":
                output = upstream_metrics
            else:
                output = upstream_metrics + generate_latest(self.registry)
        return output

    def prune(self, now_ns: int) -> None:
        self.pipeline.prune(now_ns)
        self.audio.prune(now_ns)
        self.asr.prune(now_ns)


def drain_metric_events(events: Queue, metrics: RuntimeMetrics) -> int:
    """Consume one bounded batch of stage worker events."""
    count = 0
    for _ in range(METRIC_DRAIN_BATCH_SIZE):
        try:
            event = events.get_nowait()
        except Empty:
            break
        metrics.record(*event)
        count += 1
    return count
