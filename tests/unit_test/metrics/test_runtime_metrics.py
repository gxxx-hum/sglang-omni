# SPDX-License-Identifier: Apache-2.0
"""Contract tests for Prometheus runtime metrics."""

import os
import subprocess
import sys
import tempfile
import unittest
from multiprocessing import get_context
from queue import Empty, Queue
from unittest.mock import patch

from sglang_omni.metrics.runtime import (
    RuntimeMetrics,
    drain_metric_events,
    enqueue_metric,
)
from sglang_omni.metrics.types import ORPHANED_STATE_TTL_NS


class RuntimeMetricsTest(unittest.TestCase):
    def test_runtime_owns_only_omni_metric_names(self) -> None:
        metrics = RuntimeMetrics()

        metric_names = {
            metric.name
            for collector in metrics.registry.collect()
            for metric in [collector]
        }

        self.assertTrue(any(name.startswith("sglang_omni:") for name in metric_names))
        self.assertFalse(any(name.startswith("sglang:") for name in metric_names))

    def test_render_combines_upstream_multiprocess_and_omni_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as metrics_dir:
            env = dict(os.environ)
            env["PROMETHEUS_MULTIPROC_DIR"] = metrics_dir
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from prometheus_client import Counter; "
                        "Counter('sglang:test_upstream_total', 'test').inc(2)"
                    ),
                ],
                check=True,
                env=env,
            )
            with patch.dict(
                os.environ,
                {"PROMETHEUS_MULTIPROC_DIR": metrics_dir},
            ):
                snapshot = RuntimeMetrics().render().decode()

        self.assertIn("sglang:test_upstream_total 2.0", snapshot)
        self.assertIn("sglang_omni:metric_events_dropped 0.0", snapshot)

    def test_asr_completed_request_records_domain_metrics(self) -> None:
        metrics = RuntimeMetrics()
        metrics.record("asr_request_start", "asr-1", "api", 1_000_000_000)
        metrics.record("asr_text_chunk", "asr-1", "api", 1_200_000_000)
        metrics.record("asr_text_chunk", "asr-1", "api", 1_300_000_000)
        metrics.record(
            "asr_response_done",
            "asr-1",
            "api",
            1_500_000_000,
            {
                "status": "completed",
                "audio_duration_s": 2.0,
                "engine_time_s": 0.5,
                "prompt_tokens": 20,
                "completion_tokens": 4,
            },
        )

        snapshot = metrics.render().decode()
        self.assertIn('sglang_omni:asr_requests_total{status="accepted"} 1.0', snapshot)
        self.assertIn(
            'sglang_omni:asr_requests_total{status="completed"} 1.0', snapshot
        )
        self.assertIn("sglang_omni:asr_requests_running 0.0", snapshot)
        self.assertIn("sglang_omni:asr_request_latency_seconds_sum 0.5", snapshot)
        self.assertIn("sglang_omni:asr_input_duration_seconds_sum 2.0", snapshot)
        self.assertIn("sglang_omni:asr_engine_latency_seconds_sum 0.5", snapshot)
        self.assertIn("sglang_omni:asr_engine_rtf_sum 0.25", snapshot)
        self.assertIn("sglang_omni:asr_request_rtf_sum 0.25", snapshot)
        self.assertIn("sglang_omni:asr_text_ttft_seconds_sum 0.2", snapshot)
        self.assertIn("sglang_omni:asr_text_chunk_interval_seconds_sum 0.1", snapshot)
        self.assertNotIn("sglang_omni:asr_prompt_tokens_total", snapshot)

    def test_asr_rejection_does_not_emit_performance_samples(self) -> None:
        metrics = RuntimeMetrics()
        metrics.record("asr_request_start", "asr-1", "api", 1_000_000_000)
        metrics.record(
            "asr_response_done",
            "asr-1",
            "api",
            1_100_000_000,
            {"status": "rejected"},
        )

        snapshot = metrics.render().decode()
        self.assertIn('sglang_omni:asr_requests_total{status="rejected"} 1.0', snapshot)
        self.assertIn('sglang_omni:asr_requests_total{status="accepted"} 0.0', snapshot)
        self.assertIn("sglang_omni:asr_request_latency_seconds_count 0.0", snapshot)
        self.assertIn("sglang_omni:asr_input_duration_seconds_count 0.0", snapshot)

    def test_failed_asr_stream_discards_partial_timing(self) -> None:
        metrics = RuntimeMetrics()
        metrics.record("asr_request_start", "asr-1", "api", 1_000_000_000)
        metrics.record("asr_text_chunk", "asr-1", "api", 1_200_000_000)
        metrics.record(
            "asr_response_done",
            "asr-1",
            "api",
            1_300_000_000,
            {"status": "failed", "audio_duration_s": 1.0},
        )

        snapshot = metrics.render().decode()
        self.assertIn('sglang_omni:asr_requests_total{status="failed"} 1.0', snapshot)
        self.assertIn("sglang_omni:asr_text_ttft_seconds_count 0.0", snapshot)
        self.assertIn("sglang_omni:asr_text_chunk_interval_seconds_count 0.0", snapshot)

    def test_prune_repairs_orphaned_running_and_waiting_gauges(self) -> None:
        metrics = RuntimeMetrics()
        metrics.record("request_admission", "request-1", "coordinator", 1)
        metrics.record("scheduler_queue_enter", "request-1", "talker", 2)
        metrics.record("asr_request_start", "asr-1", "api", 1)
        metrics.record("audio_response_start", "audio-1", "api", 1)
        metrics.record("audio_chunk", "audio-1", "api", 2, {"duration_s": 0.1})

        metrics.prune(ORPHANED_STATE_TTL_NS + 3)

        snapshot = metrics.render().decode()
        self.assertIn("sglang_omni:requests_running 0.0", snapshot)
        self.assertIn('sglang_omni:requests_waiting{stage="talker"} 0.0', snapshot)
        self.assertIn("sglang_omni:asr_requests_running 0.0", snapshot)
        metrics.record(
            "audio_response_done", "audio-1", "api", ORPHANED_STATE_TTL_NS + 4
        )
        self.assertIn(
            "sglang_omni:audio_output_duration_seconds_count 0.0",
            metrics.render().decode(),
        )

    def test_drain_worker_events_into_exporter(self) -> None:
        events: Queue[tuple[str, str, str, int, dict[str, object]]] = Queue()
        events.put(("request_admission", "request-1", "coordinator", 1, {}))
        metrics = RuntimeMetrics()

        self.assertEqual(drain_metric_events(events, metrics), 1)
        self.assertIn(
            'sglang_omni:requests_total{status="accepted"} 1.0',
            metrics.render().decode(),
        )

    def test_worker_events_use_bounded_queue_and_count_drops(self) -> None:
        context = get_context("spawn")
        events = context.Queue(maxsize=1)
        dropped = context.Value("Q", 0)
        try:
            enqueue_metric(
                events, dropped, "request_admission", "one", "coordinator", 1, {}
            )
            enqueue_metric(
                events, dropped, "request_admission", "two", "coordinator", 2, {}
            )
            self.assertEqual(dropped.value, 1)
            self.assertEqual(events.get(timeout=1)[1], "one")
            with self.assertRaises(Empty):
                events.get_nowait()
        finally:
            events.close()
            events.join_thread()

    def test_request_lifecycle_and_queue_state(self) -> None:
        metrics = RuntimeMetrics()
        metrics.record("request_rejected", "rejected", "coordinator", 0)
        metrics.record("request_admission", "request-1", "coordinator", 1_000_000_000)
        metrics.record("scheduler_queue_enter", "request-1", "talker", 1_100_000_000)

        snapshot = metrics.render().decode()
        self.assertIn('sglang_omni:requests_total{status="rejected"} 1.0', snapshot)
        self.assertIn('sglang_omni:requests_total{status="accepted"} 1.0', snapshot)
        self.assertIn("sglang_omni:requests_running 1.0", snapshot)
        self.assertIn('sglang_omni:requests_waiting{stage="talker"} 1.0', snapshot)

        metrics.record("scheduler_prefill_start", "request-1", "talker", 1_200_000_000)
        metrics.record("request_completed", "request-1", "coordinator", 2_000_000_000)
        metrics.record("request_completed", "request-1", "coordinator", 2_000_000_000)
        snapshot = metrics.render().decode()
        self.assertIn('sglang_omni:requests_total{status="completed"} 1.0', snapshot)
        self.assertIn("sglang_omni:requests_running 0.0", snapshot)
        self.assertIn('sglang_omni:requests_waiting{stage="talker"} 0.0', snapshot)
        self.assertIn("sglang_omni:request_e2e_latency_seconds_sum 1.0", snapshot)
        metrics.record("scheduler_queue_enter", "request-1", "talker", 1_100_000_000)
        self.assertIn(
            'sglang_omni:requests_waiting{stage="talker"} 0.0',
            metrics.render().decode(),
        )

    def test_audio_timing_and_continuity(self) -> None:
        metrics = RuntimeMetrics()
        metrics.record("audio_response_start", "request-1", "api", 1_000_000_000)
        metrics.record(
            "audio_chunk",
            "request-1",
            "api",
            1_200_000_000,
            {"duration_s": 0.1},
        )
        metrics.record(
            "audio_chunk",
            "request-1",
            "api",
            1_350_000_000,
            {"duration_s": 0.1},
        )
        metrics.record("request_admission", "request-1", "coordinator", 1_050_000_000)
        metrics.record("request_completed", "request-1", "coordinator", 1_300_000_000)
        metrics.record(
            "audio_response_done",
            "request-1",
            "api",
            1_400_000_000,
            {"audio_generation_s": 0.25},
        )
        snapshot = metrics.render().decode()
        self.assertIn("sglang_omni:audio_ttfp_seconds_sum 0.2", snapshot)
        self.assertIn("sglang_omni:audio_output_duration_seconds_sum 0.2", snapshot)
        self.assertIn("sglang_omni:audio_chunk_interval_seconds_count 1.0", snapshot)
        self.assertIn("sglang_omni:audio_underrun_seconds_count 1.0", snapshot)
        self.assertIn("sglang_omni:audio_rtf_count 1.0", snapshot)
        self.assertIn(
            'sglang_omni:audio_continuity_ok_total{threshold_ms="100"} 1.0',
            snapshot,
        )

    def test_aborted_audio_does_not_count_as_completed(self) -> None:
        metrics = RuntimeMetrics()
        metrics.record("audio_response_start", "request-1", "api", 1_000_000_000)
        metrics.record(
            "audio_chunk",
            "request-1",
            "api",
            1_100_000_000,
            {"duration_s": 0.1},
        )
        metrics.record("audio_response_aborted", "request-1", "api", 1_200_000_000)
        metrics.record("audio_response_done", "request-1", "api", 1_300_000_000)
        snapshot = metrics.render().decode()
        self.assertIn("sglang_omni:audio_output_duration_seconds_count 0.0", snapshot)
        self.assertIn("sglang_omni:audio_ttfp_seconds_count 1.0", snapshot)

    def test_stage_transfer_and_graph_metrics(self) -> None:
        metrics = RuntimeMetrics()
        metrics.record("stage_dispatch", "request-1", "talker", 1_000_000_000)
        metrics.record("scheduler_prefill_start", "request-1", "talker", 1_100_000_000)
        metrics.record("scheduler_prefill_end", "request-1", "talker", 1_200_000_000)
        metrics.record("stage_complete", "request-1", "talker", 1_500_000_000)
        metrics.record(
            "stage_hop_sent",
            "request-1",
            "talker",
            1_510_000_000,
            {"to_stage": "code2wav"},
        )
        metrics.record(
            "stage_input_received",
            "request-1",
            "code2wav",
            1_530_000_000,
            {"from_stage": "talker"},
        )
        metrics.record("code2wav_decode_start", "request-1", "code2wav", 1_550_000_000)
        metrics.record(
            "code2wav_decode_end",
            "request-1",
            "code2wav",
            1_600_000_000,
            {"execution_mode": "cuda_graph"},
        )
        snapshot = metrics.render().decode()
        self.assertIn(
            'sglang_omni:stage_latency_seconds_count{phase="prefill",stage="talker"} 1.0',
            snapshot,
        )
        self.assertIn(
            'sglang_omni:stage_latency_seconds_count{phase="decode",stage="talker"} 1.0',
            snapshot,
        )
        self.assertIn(
            'sglang_omni:stage_latency_seconds_count{phase="code2wav",stage="code2wav"} 1.0',
            snapshot,
        )
        self.assertIn(
            'sglang_omni:transfer_latency_seconds_count{from_stage="talker",to_stage="code2wav"} 1.0',
            snapshot,
        )
        self.assertIn(
            'sglang_omni:execution_total{kind="graph_hit",stage="code2wav"} 1.0',
            snapshot,
        )

    def test_each_stage_hop_is_observed(self) -> None:
        metrics = RuntimeMetrics()
        metrics.record(
            "stage_hop_sent",
            "request-1",
            "talker",
            1_000_000_000,
            {"to_stage": "code2wav"},
        )
        metrics.record(
            "stage_hop_sent",
            "request-1",
            "talker",
            1_100_000_000,
            {"to_stage": "code2wav"},
        )
        metrics.record(
            "stage_input_received",
            "request-1",
            "code2wav",
            1_200_000_000,
            {"from_stage": "talker"},
        )
        metrics.record(
            "stage_input_received",
            "request-1",
            "code2wav",
            1_300_000_000,
            {"from_stage": "talker"},
        )
        self.assertIn(
            'sglang_omni:transfer_latency_seconds_count{from_stage="talker",to_stage="code2wav"} 2.0',
            metrics.render().decode(),
        )

    def test_transfer_events_can_arrive_from_workers_out_of_order(self) -> None:
        metrics = RuntimeMetrics()
        metrics.record(
            "stage_input_received",
            "request-1",
            "code2wav",
            1_200_000_000,
            {"from_stage": "talker"},
        )
        metrics.record(
            "stage_hop_sent",
            "request-1",
            "talker",
            1_000_000_000,
            {"to_stage": "code2wav"},
        )
        self.assertIn(
            'sglang_omni:transfer_latency_seconds_count{from_stage="talker",to_stage="code2wav"} 1.0',
            metrics.render().decode(),
        )


if __name__ == "__main__":
    unittest.main()
