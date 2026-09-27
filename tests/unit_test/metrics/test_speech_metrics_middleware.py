# SPDX-License-Identifier: Apache-2.0
"""Speech HTTP audio measurement tests."""

import asyncio
import unittest

from sglang_omni.metrics.runtime import RuntimeMetrics
from sglang_omni.serve.speech_metrics import SpeechMetricsMiddleware


class SpeechMetricsMiddlewareTest(unittest.TestCase):
    def test_transcription_response_records_asr_metrics(self) -> None:
        async def app(scope, receive, send) -> None:
            scope["state"]["asr_audio_duration_s"] = 2.0
            scope["state"]["asr_engine_time_s"] = 0.5
            scope["state"]["asr_prompt_tokens"] = 20
            scope["state"]["asr_completion_tokens"] = 4
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": b"{}"})

        async def send(message: dict[str, object]) -> None:
            return None

        async def receive() -> dict[str, object]:
            return {"type": "http.request", "body": b""}

        metrics = RuntimeMetrics()
        asyncio.run(
            SpeechMetricsMiddleware(app, metrics)(
                {
                    "type": "http",
                    "path": "/v1/audio/transcriptions",
                    "method": "POST",
                },
                receive,
                send,
            )
        )
        snapshot = metrics.render().decode()
        self.assertIn(
            'sglang_omni:asr_requests_total{status="completed"} 1.0', snapshot
        )
        self.assertIn("sglang_omni:asr_input_duration_seconds_sum 2.0", snapshot)
        self.assertIn("sglang_omni:asr_engine_latency_seconds_sum 0.5", snapshot)

    def test_translation_response_records_asr_metrics(self) -> None:
        async def app(scope, receive, send) -> None:
            scope["state"]["asr_audio_duration_s"] = 2.0
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"{}"})

        async def send(message: dict[str, object]) -> None:
            return None

        async def receive() -> dict[str, object]:
            return {"type": "http.request", "body": b""}

        metrics = RuntimeMetrics()
        asyncio.run(
            SpeechMetricsMiddleware(app, metrics)(
                {"type": "http", "path": "/v1/audio/translations", "method": "POST"},
                receive,
                send,
            )
        )
        snapshot = metrics.render().decode()
        self.assertIn(
            'sglang_omni:asr_requests_total{status="completed"} 1.0', snapshot
        )
        self.assertIn("sglang_omni:asr_input_duration_seconds_sum 2.0", snapshot)

    def test_asr_disconnect_records_aborted_without_performance_samples(self) -> None:
        async def app(scope, receive, send) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            raise ConnectionError("client disconnected")

        async def send(message: dict[str, object]) -> None:
            return None

        async def receive() -> dict[str, object]:
            return {"type": "http.request", "body": b""}

        metrics = RuntimeMetrics()
        with self.assertRaises(ConnectionError):
            asyncio.run(
                SpeechMetricsMiddleware(app, metrics)(
                    {
                        "type": "http",
                        "path": "/v1/audio/transcriptions",
                        "method": "POST",
                    },
                    receive,
                    send,
                )
            )
        snapshot = metrics.render().decode()
        self.assertIn(
            'sglang_omni:asr_requests_total{status="cancelled"} 1.0', snapshot
        )
        self.assertIn("sglang_omni:asr_requests_running 0.0", snapshot)
        self.assertIn("sglang_omni:asr_request_latency_seconds_count 0.0", snapshot)

    def test_transcription_stream_records_text_timing(self) -> None:
        async def app(scope, receive, send) -> None:
            scope["state"]["asr_audio_duration_s"] = 1.0
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"text/event-stream")],
                }
            )
            for delta in ("hello", " world"):
                body = (
                    'data: {"type":"transcript.text.delta","delta":' f'"{delta}"}}\n\n'
                ).encode()
                await send(
                    {"type": "http.response.body", "body": body, "more_body": True}
                )
            await send({"type": "http.response.body", "body": b""})

        async def send(message: dict[str, object]) -> None:
            return None

        async def receive() -> dict[str, object]:
            return {"type": "http.request", "body": b""}

        metrics = RuntimeMetrics()
        asyncio.run(
            SpeechMetricsMiddleware(app, metrics)(
                {
                    "type": "http",
                    "path": "/v1/audio/transcriptions",
                    "method": "POST",
                },
                receive,
                send,
            )
        )
        snapshot = metrics.render().decode()
        self.assertIn("sglang_omni:asr_text_ttft_seconds_count 1.0", snapshot)
        self.assertIn("sglang_omni:asr_text_chunk_interval_seconds_count 1.0", snapshot)

    def test_transcription_stream_failure_discards_partial_timing(self) -> None:
        async def app(scope, receive, send) -> None:
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"text/event-stream")],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b'data: {"type":"transcript.text.delta","delta":"hi"}\n\n',
                    "more_body": True,
                }
            )
            scope["state"]["asr_stream_failed"] = True
            await send({"type": "http.response.body", "body": b""})

        async def send(message: dict[str, object]) -> None:
            return None

        async def receive() -> dict[str, object]:
            return {"type": "http.request", "body": b""}

        metrics = RuntimeMetrics()
        asyncio.run(
            SpeechMetricsMiddleware(app, metrics)(
                {
                    "type": "http",
                    "path": "/v1/audio/transcriptions",
                    "method": "POST",
                },
                receive,
                send,
            )
        )
        snapshot = metrics.render().decode()
        self.assertIn('sglang_omni:asr_requests_total{status="failed"} 1.0', snapshot)
        self.assertIn(
            'sglang_omni:asr_requests_total{status="completed"} 0.0', snapshot
        )
        self.assertIn("sglang_omni:asr_text_ttft_seconds_count 0.0", snapshot)
        self.assertIn("sglang_omni:asr_text_chunk_interval_seconds_count 0.0", snapshot)

    def test_pcm_chunks_are_measured_through_response_completion(self) -> None:
        async def app(scope, receive, send) -> None:
            scope["state"]["audio_streaming"] = True
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", b"audio/pcm"),
                        (b"x-sample-rate", b"24000"),
                        (b"x-channels", b"1"),
                        (b"x-bit-depth", b"16"),
                    ],
                }
            )
            await send(
                {"type": "http.response.body", "body": bytes(4800), "more_body": True}
            )
            await send({"type": "http.response.body", "body": bytes(4800)})

        metrics = RuntimeMetrics()
        sent: list[dict[str, object]] = []

        async def send(message: dict[str, object]) -> None:
            sent.append(message)

        async def receive() -> dict[str, object]:
            return {"type": "http.request", "body": b""}

        asyncio.run(
            SpeechMetricsMiddleware(app, metrics)(
                {"type": "http", "path": "/v1/audio/speech", "method": "POST"},
                receive,
                send,
            )
        )
        snapshot = metrics.render().decode()
        self.assertEqual(len(sent), 3)
        self.assertIn("sglang_omni:audio_output_duration_seconds_sum 0.2", snapshot)
        self.assertIn("sglang_omni:audio_ttfp_seconds_count 1.0", snapshot)
        self.assertIn("sglang_omni:audio_chunk_interval_seconds_count 1.0", snapshot)

    def test_failed_send_does_not_record_completed_stream(self) -> None:
        async def app(scope, receive, send) -> None:
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", b"audio/pcm"),
                        (b"x-sample-rate", b"24000"),
                        (b"x-channels", b"1"),
                        (b"x-bit-depth", b"16"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": bytes(4800)})

        async def send(message: dict[str, object]) -> None:
            if message["type"] == "http.response.body":
                raise ConnectionError("client disconnected")

        async def receive() -> dict[str, object]:
            return {"type": "http.request", "body": b""}

        metrics = RuntimeMetrics()
        with self.assertRaises(ConnectionError):
            asyncio.run(
                SpeechMetricsMiddleware(app, metrics)(
                    {"type": "http", "path": "/v1/audio/speech", "method": "POST"},
                    receive,
                    send,
                )
            )
        snapshot = metrics.render().decode()
        self.assertIn("sglang_omni:audio_output_duration_seconds_count 0.0", snapshot)

    def test_nonstream_audio_reports_duration_and_engine_rtf(self) -> None:
        async def app(scope, receive, send) -> None:
            scope["state"]["audio_duration_s"] = 0.5
            scope["state"]["audio_generation_s"] = 0.25
            scope["state"]["prompt_tokens"] = 12
            scope["state"]["completion_tokens"] = 8
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"audio/wav")],
                }
            )
            await send({"type": "http.response.body", "body": bytes(10)})

        async def send(message: dict[str, object]) -> None:
            return None

        async def receive() -> dict[str, object]:
            return {"type": "http.request", "body": b""}

        metrics = RuntimeMetrics()
        asyncio.run(
            SpeechMetricsMiddleware(app, metrics)(
                {"type": "http", "path": "/v1/audio/speech", "method": "POST"},
                receive,
                send,
            )
        )
        snapshot = metrics.render().decode()
        self.assertIn("sglang_omni:audio_output_duration_seconds_sum 0.5", snapshot)
        self.assertIn("sglang_omni:audio_rtf_sum 0.5", snapshot)
        self.assertIn("sglang_omni:audio_ttfp_seconds_count 0.0", snapshot)
        self.assertNotIn("sglang_omni:tts_prompt_tokens_total", snapshot)

    def test_failed_audio_does_not_record_tts_tokens(self) -> None:
        async def app(scope, receive, send) -> None:
            scope["state"]["prompt_tokens"] = 12
            scope["state"]["completion_tokens"] = 8
            await send(
                {
                    "type": "http.response.start",
                    "status": 500,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": b"{}"})

        async def send(message: dict[str, object]) -> None:
            return None

        async def receive() -> dict[str, object]:
            return {"type": "http.request", "body": b""}

        metrics = RuntimeMetrics()
        asyncio.run(
            SpeechMetricsMiddleware(app, metrics)(
                {"type": "http", "path": "/v1/audio/speech", "method": "POST"},
                receive,
                send,
            )
        )
        snapshot = metrics.render().decode()
        self.assertNotIn("sglang:prompt_tokens_total{", snapshot)
        self.assertNotIn("sglang:generation_tokens_total{", snapshot)

    def test_nonstream_pcm_does_not_emit_ttfp_or_continuity(self) -> None:
        async def app(scope, receive, send) -> None:
            scope["state"]["audio_duration_s"] = 0.1
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", b"audio/pcm"),
                        (b"x-sample-rate", b"24000"),
                        (b"x-channels", b"1"),
                        (b"x-bit-depth", b"16"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": bytes(4800)})

        async def send(message: dict[str, object]) -> None:
            return None

        async def receive() -> dict[str, object]:
            return {"type": "http.request", "body": b""}

        metrics = RuntimeMetrics()
        asyncio.run(
            SpeechMetricsMiddleware(app, metrics)(
                {"type": "http", "path": "/v1/audio/speech", "method": "POST"},
                receive,
                send,
            )
        )
        snapshot = metrics.render().decode()
        self.assertIn("sglang_omni:audio_output_duration_seconds_sum 0.1", snapshot)
        self.assertIn("sglang_omni:audio_ttfp_seconds_count 0.0", snapshot)
        self.assertIn("sglang_omni:audio_underrun_seconds_count 0.0", snapshot)


if __name__ == "__main__":
    unittest.main()
