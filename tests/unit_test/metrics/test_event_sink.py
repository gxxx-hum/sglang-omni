# SPDX-License-Identifier: Apache-2.0
"""Metric event forwarding tests."""

import unittest

from sglang_omni.profiler.event_recorder import emit, set_metrics_sink


class MetricEventSinkTest(unittest.TestCase):
    def test_forwards_selected_events_without_enabling_profiler(self) -> None:
        seen: list[tuple[str, str, str, int, dict[str, object]]] = []
        set_metrics_sink(lambda *args: seen.append(args))
        try:
            emit(
                request_id="request-1",
                stage="talker",
                event_name="stage_hop_sent",
                timestamp_ns=123,
                metadata={"to_stage": "code2wav", "unrelated": object()},
            )
            emit(
                request_id="request-1",
                stage="talker",
                event_name="unrelated",
                timestamp_ns=124,
            )
        finally:
            set_metrics_sink(None)

        self.assertEqual(
            seen,
            [
                (
                    "stage_hop_sent",
                    "request-1",
                    "talker",
                    123,
                    {"to_stage": "code2wav"},
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
