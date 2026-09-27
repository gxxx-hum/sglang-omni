# SPDX-License-Identifier: Apache-2.0
"""Measure speech audio as it leaves the HTTP server."""

import asyncio
import time
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from sglang_omni.metrics.runtime import RuntimeMetrics
from sglang_omni.metrics.types import MetricDetails, RuntimeMetricEvent, TerminalStatus

ASR_ENDPOINTS = frozenset({"/v1/audio/transcriptions", "/v1/audio/translations"})


class SpeechMetricsMiddleware:
    """Record audio timing from successful speech responses."""

    def __init__(self, app: ASGIApp, metrics: RuntimeMetrics) -> None:
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] == "http"
            and scope["path"] in ASR_ENDPOINTS
            and scope["method"] == "POST"
        ):
            request_id = str(uuid.uuid4())
            scope.setdefault("state", {})
            self.metrics.record("asr_request_start", request_id, "api", time.time_ns())
            response_status: int | None = None
            finished = False
            is_streaming = False

            async def send_transcription(message: Message) -> None:
                nonlocal response_status, finished, is_streaming
                if message["type"] == "http.response.start":
                    response_status = message["status"]
                    headers = dict(message.get("headers", []))
                    is_streaming = headers.get(b"content-type", b"").startswith(
                        b"text/event-stream"
                    )
                elif message["type"] == "http.response.body":
                    body = message.get("body", b"")
                    if b'"type":"transcript.text.delta"' in body:
                        self.metrics.record(
                            "asr_text_chunk", request_id, "api", time.time_ns()
                        )
                await send(message)
                if message["type"] == "http.response.body" and not message.get(
                    "more_body", False
                ):
                    finished = True

            def record_terminal(
                event_name: RuntimeMetricEvent,
                status: TerminalStatus,
                details: MetricDetails | None = None,
            ) -> None:
                terminal_details = details or {}
                terminal_details["status"] = status
                self.metrics.record(
                    event_name,
                    request_id,
                    "api",
                    time.time_ns(),
                    terminal_details,
                )

            try:
                await self.app(scope, receive, send_transcription)
            except (asyncio.CancelledError, ConnectionError):
                record_terminal("asr_response_aborted", "cancelled")
                raise
            except Exception:
                record_terminal("asr_response_aborted", "failed")
                raise
            if not finished:
                record_terminal("asr_response_aborted", "cancelled")
                return

            state = scope["state"]
            if state.get("asr_stream_failed", False):
                status: TerminalStatus = "failed"
            elif response_status is not None and response_status < 400:
                status = "completed"
            elif response_status is not None and response_status < 500:
                status = "rejected"
            else:
                status = "failed"
            details: MetricDetails = {"is_streaming": is_streaming}
            for name in (
                "asr_audio_duration_s",
                "asr_engine_time_s",
                "asr_prompt_tokens",
                "asr_completion_tokens",
            ):
                value = state.get(name)
                if isinstance(value, (str, int, float, bool)) or value is None:
                    details[name.removeprefix("asr_")] = value
            record_terminal("asr_response_done", status, details)
            return

        if (
            scope["type"] != "http"
            or scope["path"] != "/v1/audio/speech"
            or scope["method"] != "POST"
        ):
            await self.app(scope, receive, send)
            return

        request_id = str(uuid.uuid4())
        scope.setdefault("state", {})
        self.metrics.record("audio_response_start", request_id, "api", time.time_ns())
        bytes_per_second = 0
        is_audio_response = False
        succeeded = False

        async def send_audio(message: Message) -> None:
            nonlocal bytes_per_second, is_audio_response, succeeded
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                content_type = headers.get(b"content-type", b"")
                is_audio_response = message["status"] < 400 and content_type.startswith(
                    b"audio/"
                )
                if (
                    is_audio_response
                    and scope["state"].get("audio_streaming", False)
                    and content_type.startswith(b"audio/pcm")
                ):
                    sample_rate = int(headers[b"x-sample-rate"])
                    channels = int(headers[b"x-channels"])
                    bit_depth = int(headers[b"x-bit-depth"])
                    bytes_per_second = sample_rate * channels * bit_depth // 8
            elif message["type"] == "http.response.body" and bytes_per_second:
                body = message.get("body", b"")
                if body:
                    self.metrics.record(
                        "audio_chunk",
                        request_id,
                        "api",
                        time.time_ns(),
                        {"duration_s": len(body) / bytes_per_second},
                    )
            await send(message)
            if message["type"] == "http.response.body" and is_audio_response:
                if not message.get("more_body", False):
                    succeeded = True

        try:
            await self.app(scope, receive, send_audio)
        finally:
            if succeeded:
                state = scope["state"]
                details = {
                    name: state[name]
                    for name in (
                        "audio_duration_s",
                        "audio_generation_s",
                        "prompt_tokens",
                        "completion_tokens",
                    )
                    if name in state
                }
                details["is_streaming"] = state.get("audio_streaming", False)
                self.metrics.record(
                    "audio_response_done", request_id, "api", time.time_ns(), details
                )
            else:
                self.metrics.record(
                    "audio_response_aborted", request_id, "api", time.time_ns()
                )
