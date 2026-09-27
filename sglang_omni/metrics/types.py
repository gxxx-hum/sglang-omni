# SPDX-License-Identifier: Apache-2.0
"""Shared types and constants for runtime metrics."""

from __future__ import annotations

from typing import Literal

MetricValue = str | int | float | bool | None
MetricDetails = dict[str, MetricValue]
RequestStatus = Literal["accepted", "rejected", "completed", "failed", "cancelled"]
TerminalStatus = Literal["rejected", "completed", "failed", "cancelled"]
RuntimeMetricEvent = Literal[
    "asr_request_start",
    "asr_response_aborted",
    "asr_response_done",
    "asr_text_chunk",
    "audio_chunk",
    "audio_response_aborted",
    "audio_response_done",
    "audio_response_start",
    "code2wav_decode_end",
    "code2wav_decode_start",
    "preprocess_end",
    "preprocess_start",
    "request_admission",
    "request_cancelled",
    "request_completed",
    "request_failed",
    "request_rejected",
    "scheduler_prefill_end",
    "scheduler_prefill_start",
    "scheduler_queue_enter",
    "stage_complete",
    "stage_dispatch",
    "stage_hop_sent",
    "stage_input_received",
]

REQUEST_STATUSES: tuple[RequestStatus, ...] = (
    "accepted",
    "rejected",
    "completed",
    "failed",
    "cancelled",
)
TERMINAL_STATUSES: tuple[TerminalStatus, ...] = (
    "rejected",
    "completed",
    "failed",
    "cancelled",
)
ASR_EVENT_NAMES = frozenset(
    {"asr_request_start", "asr_response_aborted", "asr_response_done", "asr_text_chunk"}
)
AUDIO_EVENT_NAMES = frozenset(
    {
        "audio_chunk",
        "audio_response_aborted",
        "audio_response_done",
        "audio_response_start",
    }
)
CONTINUITY_THRESHOLDS_MS = (50, 100, 200)
LATENCY_BUCKETS_S = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300)
FAST_BUCKETS_S = (
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1,
    2.5,
    5,
    10,
    30,
    60,
)
RTF_BUCKETS = (0.1, 0.2, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 5, 10)
TERMINAL_GRACE_NS = 30_000_000_000
ORPHANED_STATE_TTL_NS = 3_600_000_000_000
