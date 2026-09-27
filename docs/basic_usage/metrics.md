# Prometheus metrics

Pass `--enable-metrics` when starting the server to enable collection and expose
Prometheus text at `GET /metrics`:

```bash
sgl-omni serve --model-path MODEL_ID --enable-metrics
curl http://localhost:8000/metrics
```

Metrics are disabled by default, matching SGLang server behavior.

## Request metrics

| Metric | Type | Description |
| --- | --- | --- |
| `sglang_omni:requests_total{status}` | Counter | Accepted, rejected, completed, failed, and cancelled pipeline requests |
| `sglang_omni:requests_running` | Gauge | Accepted requests without a terminal event |
| `sglang_omni:requests_waiting{stage}` | Gauge | Requests waiting in a stage scheduler |
| `sglang_omni:request_e2e_latency_seconds` | Histogram | Pipeline admission to completion latency |

## Inference metrics

Engine stages use SGLang's own metrics collector. SGLang-Omni enables that
collector for each engine stage and exposes its multiprocess output without
recomputing token counts or latency. Every SGLang metric includes a `stage`
label; replicated stages also include `replica`. Existing SGLang labels such as
`model_name`, `engine_type`, `tp_rank`, and `pp_rank` remain unchanged.

Metric names, buckets, and calculations follow the installed SGLang version.
See the [SGLang observability documentation](https://docs.sglang.ai/advanced_features/observability.html)
for the available inference metrics. The SGLang-Omni sections below document
only the additional pipeline and modality metrics.

## Generated audio metrics

Audio output from `/v1/audio/speech` and `/v1/chat/completions` populates these
metrics:

| Metric | Type | Description |
| --- | --- | --- |
| `sglang_omni:audio_ttfp_seconds` | Histogram | Time to first audio payload for streaming responses |
| `sglang_omni:audio_rtf` | Histogram | Engine generation time divided by output audio duration |
| `sglang_omni:audio_e2e_latency_seconds` | Histogram | HTTP request arrival to final audio response |
| `sglang_omni:audio_output_duration_seconds` | Histogram | Generated audio duration |
| `sglang_omni:audio_chunk_interval_seconds` | Histogram | Wall time between audio chunks |
| `sglang_omni:audio_underrun_seconds` | Histogram | Largest playback buffer underrun in a completed stream |
| `sglang_omni:audio_continuity_ok_total{threshold_ms}` | Counter | Completed streams within each underrun threshold |

Non-streaming responses do not contribute TTFP, chunk interval, underrun, or
continuity samples. Interrupted streams can contribute TTFP and chunk intervals,
but do not contribute output duration, E2E latency, RTF, underrun, or continuity.

## ASR metrics

Requests to `/v1/audio/transcriptions` and `/v1/audio/translations` populate
these metrics:

| Metric | Type | Description |
| --- | --- | --- |
| `sglang_omni:asr_requests_total{status}` | Counter | Accepted, rejected, completed, failed, and cancelled requests |
| `sglang_omni:asr_requests_running` | Gauge | ASR HTTP requests currently in progress |
| `sglang_omni:asr_request_latency_seconds` | Histogram | HTTP request latency for completed transcriptions |
| `sglang_omni:asr_input_duration_seconds` | Histogram | Input audio duration |
| `sglang_omni:asr_engine_latency_seconds` | Histogram | Summed engine time, including all long-audio chunks |
| `sglang_omni:asr_engine_rtf` | Histogram | Engine time divided by input audio duration |
| `sglang_omni:asr_request_rtf` | Histogram | Server request latency divided by input audio duration |
| `sglang_omni:asr_text_ttft_seconds` | Histogram | Time to first transcript delta for streaming requests |
| `sglang_omni:asr_text_chunk_interval_seconds` | Histogram | Time between streaming transcript deltas |

Queue, cache, token, and inference latency metrics come from SGLang engine
stages. Filter them with the configured `stage` label. ASR HTTP timing remains
under the `sglang_omni:asr_*` families because SGLang does not own that boundary.

## Stage and execution metrics

| Metric | Type | Description |
| --- | --- | --- |
| `sglang_omni:stage_latency_seconds{stage,phase}` | Histogram | Queue, preprocessing, prefill, decode, Code2Wav, and stage latency |
| `sglang_omni:transfer_latency_seconds{from_stage,to_stage}` | Histogram | Inter-process stage transfer latency |
| `sglang_omni:execution_total{stage,kind}` | Counter | Code2Wav graph hits and eager fallbacks |
| `sglang_omni:metric_events_dropped` | Gauge | Worker metric events dropped because the aggregation queue was full |

The endpoint also includes the standard Python process, platform, and garbage
collector metric families from `prometheus_client`.

Prometheus derives benchmark summaries from counters and histograms. For example:

```promql
# Completed ASR requests per second
sum(rate(sglang_omni:asr_requests_total{status="completed"}[5m]))

# Input audio seconds processed per wall-clock second (RTFx)
sum(rate(sglang_omni:asr_input_duration_seconds_sum[5m]))

# P95 engine RTF
histogram_quantile(
  0.95,
  sum by (le) (rate(sglang_omni:asr_engine_rtf_bucket[5m]))
)
```
