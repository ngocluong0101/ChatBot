"""Tracing (OpenTelemetry -> Jaeger) và metrics (Prometheus, endpoint /metrics) cho pipeline.

Mọi thứ đều TẮT được và không làm gì khi tắt:
  * tracing: `RAG_TRACING_ENABLED=true` + `RAG_OTLP_ENDPOINT=http://jaeger:4318`; tắt thì span là no-op.
  * metrics: `RAG_METRICS_ENABLED=false` để không tạo bộ đếm nào.

Mỗi bước của pipeline (viết lại câu hỏi, truy hồi, từng nhánh trong truy hồi, LLM) đi qua `Telemetry.stage`:
một span + một mẫu vào histogram `ragbot_stage_seconds{stage=...}`. Span KHÔNG dùng context "hiện hành" của
OpenTelemetry (không `attach`): generator của /chat/stream được Starlette chạy rải rác trên nhiều thread,
gắn/gỡ context giữa các thread sẽ lỗi, nên span cha được truyền tường minh.
"""

import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ragbot.config import Settings

logger = logging.getLogger(__name__)

# Histogram theo giây: bước nhanh (BM25, vài ms) tới LLM chậm (vài chục giây khi hàng đợi dài).
BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 40, 80, 160)


class Stage:
    """Một bước đang chạy: ghi thuộc tính vào span (nếu có) và là cha của các bước con."""

    def __init__(self, span: Any = None) -> None:
        self._span = span
        self.started = time.perf_counter()

    def set(self, key: str, value: Any) -> None:
        if self._span is not None and value is not None:
            self._span.set_attribute(key, value)

    @property
    def span(self) -> Any:
        return self._span

    def elapsed(self) -> float:
        return time.perf_counter() - self.started


class UsageRecorder:
    """Callback LangChain gắn vào ChatOllama: giữ số token của lượt sinh vừa xong, tách theo thread.

    `StrOutputParser` ở cuối chuỗi làm mất `usage_metadata`, nên đọc nó ở đây. Lời gọi sync chạy
    callback trên chính thread gọi `.invoke()` / `next()`, nên `pop()` ngay sau lời gọi là an toàn.
    """

    def __init__(self) -> None:
        self._local = threading.local()

    def on_llm_end(self, response: Any, **_: Any) -> None:
        try:
            message = response.generations[0][0].message
            usage = message.usage_metadata or {}
            info = response.generations[0][0].generation_info or {}
        except (AttributeError, IndexError):
            return
        record = {
            "prompt_tokens": usage.get("input_tokens"),
            "completion_tokens": usage.get("output_tokens"),
        }
        eval_ns, eval_count = info.get("eval_duration"), info.get("eval_count")
        if eval_ns and eval_count:
            record["tokens_per_second"] = round(eval_count / (eval_ns / 1e9), 2)
        self._local.last = record

    def pop(self) -> dict[str, Any] | None:
        record = getattr(self._local, "last", None)
        self._local.last = None
        return record


def make_usage_handler(recorder: UsageRecorder) -> Any:
    """Gói recorder thành BaseCallbackHandler (import muộn để module này nhẹ khi không cần)."""
    from langchain_core.callbacks import BaseCallbackHandler

    class Handler(BaseCallbackHandler):
        def on_llm_end(self, response: Any, **kwargs: Any) -> None:
            recorder.on_llm_end(response)

    return Handler()


def setup_tracing(settings: Settings) -> Any:
    """Cài TracerProvider gửi span qua OTLP/HTTP tới `otlp_endpoint`. Trả về tracer (no-op nếu tắt)."""
    from opentelemetry import trace

    if not settings.tracing_enabled:
        return trace.get_tracer("ragbot")
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": settings.service_name}))
    endpoint = settings.otlp_endpoint.rstrip("/") + "/v1/traces"
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    # Provider toàn cục chỉ đặt được một lần: lấy tracer thẳng từ provider riêng để các test không đụng nhau.
    logger.info("Tracing bật: gửi span tới %s", endpoint)
    return provider.get_tracer("ragbot")


class Telemetry:
    def __init__(self, settings: Settings | None = None, registry: Any = None, tracer: Any = None) -> None:
        """`tracer` truyền vào để test (bộ ghi span trong bộ nhớ); mặc định tạo theo `settings`."""
        self._tracer: Any = tracer
        self.metrics = False
        if tracer is None and settings is not None and settings.tracing_enabled:
            self._tracer = setup_tracing(settings)
        if settings is not None and settings.metrics_enabled:
            self._init_metrics(registry)

    # ------------------------------------------------------------------ metrics

    def _init_metrics(self, registry: Any) -> None:
        from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

        self.registry = registry or CollectorRegistry()
        self.metrics = True
        self._stage = Histogram(
            "ragbot_stage_seconds",
            "Thời gian từng bước của pipeline",
            ["stage"],
            buckets=BUCKETS,
            registry=self.registry,
        )
        self._requests = Counter(
            "ragbot_requests_total",
            "Số yêu cầu theo endpoint và kết quả (ok, refused, error, busy)",
            ["endpoint", "outcome"],
            registry=self.registry,
        )
        self._request_seconds = Histogram(
            "ragbot_request_seconds",
            "Thời gian xử lý trọn một yêu cầu (không gồm thời gian chờ vào hàng đợi)",
            ["endpoint"],
            buckets=BUCKETS,
            registry=self.registry,
        )
        self._queue_wait = Histogram(
            "ragbot_queue_wait_seconds",
            "Thời gian chờ được nhận xử lý",
            buckets=BUCKETS,
            registry=self.registry,
        )
        self._in_flight = Gauge(
            "ragbot_in_flight_requests", "Yêu cầu đang được xử lý", registry=self.registry
        )
        self._waiting = Gauge("ragbot_waiting_requests", "Yêu cầu đang chờ", registry=self.registry)
        self._tokens = Counter(
            "ragbot_llm_tokens_total",
            "Token LLM theo loại (prompt, completion)",
            ["kind"],
            registry=self.registry,
        )

    def render_metrics(self) -> bytes:
        from prometheus_client import generate_latest

        return generate_latest(self.registry)

    def observe_request(self, endpoint: str, outcome: str, seconds: float | None = None) -> None:
        if not self.metrics:
            return
        self._requests.labels(endpoint, outcome).inc()
        if seconds is not None:
            self._request_seconds.labels(endpoint).observe(seconds)

    def observe_queue_wait(self, seconds: float) -> None:
        if self.metrics:
            self._queue_wait.observe(seconds)

    def observe_tokens(self, usage: dict[str, Any] | None) -> None:
        if self.metrics and usage:
            for kind in ("prompt", "completion"):
                if usage.get(f"{kind}_tokens"):
                    self._tokens.labels(kind).inc(usage[f"{kind}_tokens"])

    @contextmanager
    def waiting(self) -> Iterator[None]:
        if self.metrics:
            self._waiting.inc()
        try:
            yield
        finally:
            if self.metrics:
                self._waiting.dec()

    @contextmanager
    def in_flight(self) -> Iterator[None]:
        if self.metrics:
            self._in_flight.inc()
        try:
            yield
        finally:
            if self.metrics:
                self._in_flight.dec()

    # ------------------------------------------------------------------ tracing

    @contextmanager
    def stage(self, name: str, parent: Stage | None = None, **attributes: Any) -> Iterator[Stage]:
        span = None
        if self._tracer is not None:
            from opentelemetry import trace

            context = trace.set_span_in_context(parent.span) if parent and parent.span else None
            span = self._tracer.start_span(name, context=context)
        current = Stage(span)
        for key, value in attributes.items():
            current.set(key, value)
        try:
            yield current
        except BaseException as exc:
            # GeneratorExit (client ngắt SSE giữa chừng) không phải lỗi của bước này.
            if span is not None and not isinstance(exc, GeneratorExit):
                span.record_exception(exc)
                span.set_attribute("error", True)
            raise
        finally:
            if self.metrics:
                self._stage.labels(name).observe(current.elapsed())
            if span is not None:
                span.end()


NOOP = Telemetry()
