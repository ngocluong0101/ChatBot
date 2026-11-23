"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>

API HTTP (FastAPI): /chat, /chat/stream (SSE), /health và giao diện web ở /.

Chạy:  uvicorn ragbot.api:app --host 0.0.0.0 --port 8000

Mỗi lượt hỏi giành một slot semaphore (`RAG_MAX_CONCURRENT_REQUESTS`, nên bằng OLLAMA_NUM_PARALLEL);
chờ quá `RAG_QUEUE_TIMEOUT_SECONDS` thì trả 503 + Retry-After. Truy hồi xếp hàng qua một khóa trong
`Retriever`, còn LLM chạy chồng lên nhau. Quan sát: /metrics (Prometheus), tracing OpenTelemetry
(ragbot/telemetry.py).
"""

import json
import logging
import secrets
import threading
import time
import weakref
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from ragbot.config import Settings, get_settings
from ragbot.errors import RagError
from ragbot.pipeline import Answer, RagPipeline, Source, Turn, build_pipeline, format_citation
from ragbot.telemetry import Telemetry

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(max_length=1000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=40)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question không được để trống")
        return value


class SourceOut(BaseModel):
    file: str
    page: int | None
    section: str | None
    score: float
    # Chỉ xuất hiện với văn bản pháp luật (các trường không đặt bị bỏ khỏi JSON, xem to_answer_out).
    title: str | None = None
    dieu: str | None = None
    khoan: str | None = None
    phu_luc: str | None = None
    kind: str | None = None
    citation: str | None = (
        None  # chuỗi trích dẫn đầy đủ, ví dụ "Bộ luật Lao động 2019, Điều 113, Khoản 1 (trang 41)"
    )


class AnswerOut(BaseModel):
    text: str
    refused: bool
    question: str  # câu hỏi độc lập thực sự dùng để truy hồi (có thể đã được viết lại)
    sources: list[SourceOut]


def to_source_out(source: Source) -> SourceOut:
    """Nguồn cũ giữ nguyên dạng {file, page, section, score}; trường pháp luật chỉ thêm khi có giá trị."""
    optional = {
        "title": source.title,
        "dieu": source.dieu,
        "khoan": source.khoan,
        "phu_luc": source.phu_luc,
        "kind": source.kind,
    }
    extra = {k: v for k, v in optional.items() if v is not None}
    if source.title:
        extra["citation"] = format_citation(source)
    return SourceOut(file=source.file, page=source.page, section=source.section, score=source.score, **extra)


def to_answer_out(answer: Answer) -> AnswerOut:
    return AnswerOut(
        text=answer.text,
        refused=answer.refused,
        question=answer.question,
        sources=[to_source_out(s) for s in answer.sources],
    )


def to_turns(history: list[ChatMessage]) -> list[Turn]:
    return [Turn(m.role, m.content) for m in history]


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class Slot:
    """Một slot xử lý đã giành được; `release()` an toàn khi gọi nhiều lần."""

    def __init__(self, semaphore: threading.BoundedSemaphore, telemetry: Telemetry) -> None:
        self._semaphore = semaphore
        self._telemetry = telemetry
        self._lock = threading.Lock()
        self._held = True
        self.started = time.perf_counter()
        self._flight = telemetry.in_flight()
        self._flight.__enter__()

    def release(self) -> None:
        with self._lock:
            if not self._held:
                return
            self._held = False
        self._flight.__exit__(None, None, None)
        self._semaphore.release()


def create_app(pipeline: RagPipeline | None = None, settings: Settings | None = None) -> FastAPI:
    """`pipeline` truyền vào để test; mặc định dựng pipeline thật khi khởi động."""
    settings = settings or get_settings()
    semaphore = threading.BoundedSemaphore(settings.max_concurrent_requests)
    telemetry = Telemetry(settings)

    def acquire_slot(endpoint: str) -> Slot:
        """Chờ tối đa `queue_timeout_seconds` để được xử lý; quá hạn thì 503 (backpressure)."""
        waited = time.perf_counter()
        with telemetry.waiting():
            got = semaphore.acquire(timeout=settings.queue_timeout_seconds)
        telemetry.observe_queue_wait(time.perf_counter() - waited)
        if not got:
            telemetry.observe_request(endpoint, "busy")
            raise HTTPException(
                status_code=503,
                detail="Hệ thống đang quá tải, hãy thử lại sau",
                headers={"Retry-After": str(max(1, int(settings.queue_timeout_seconds)))},
            )
        return Slot(semaphore, telemetry)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if pipeline is not None:
            app.state.pipeline = pipeline
        else:
            try:
                app.state.pipeline = await run_in_threadpool(build_pipeline, settings)
            except RagError as exc:
                logger.error("Không khởi động được pipeline: %s", exc)
                raise
        if isinstance(app.state.pipeline, RagPipeline):
            app.state.pipeline.telemetry = telemetry
        yield

    app = FastAPI(title="RAG Chatbot", lifespan=lifespan)

    def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        if not settings.api_key:
            return
        supplied = (x_api_key or "").encode()
        if not secrets.compare_digest(supplied, settings.api_key.encode()):
            raise HTTPException(status_code=401, detail="API key không hợp lệ")

    auth = [Depends(require_api_key)]

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        return {
            "status": "ok",
            "llm_model": settings.llm_model,
            "rerank_model": settings.rerank_model or None,
            "chunks": len(request.app.state.pipeline.chunks()),
            "auth_required": bool(settings.api_key),
        }

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        if not telemetry.metrics:
            raise HTTPException(status_code=404, detail="metrics đang tắt (RAG_METRICS_ENABLED=false)")
        from prometheus_client import CONTENT_TYPE_LATEST

        return Response(telemetry.render_metrics(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/chat", response_model=AnswerOut, response_model_exclude_unset=True, dependencies=auth)
    def chat(body: ChatRequest, request: Request) -> AnswerOut:
        slot = acquire_slot("chat")
        outcome = "error"
        try:
            try:
                answer = request.app.state.pipeline.ask(body.question, to_turns(body.history))
            except RagError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            outcome = "refused" if answer.refused else "ok"
        finally:
            telemetry.observe_request("chat", outcome, time.perf_counter() - slot.started)
            slot.release()
        return to_answer_out(answer)

    @app.post("/chat/stream", dependencies=auth)
    def chat_stream(body: ChatRequest, request: Request) -> StreamingResponse:
        active_pipeline: RagPipeline = request.app.state.pipeline
        # Giành slot TRƯỚC khi trả response: quá tải còn trả được 503 (đã gửi header 200 thì không đổi được).
        slot = acquire_slot("chat_stream")

        def generate() -> Iterator[str]:
            outcome = "error"
            try:
                for event in active_pipeline.ask_stream(body.question, to_turns(body.history)):
                    if event.kind == "done" and event.answer is not None:
                        outcome = "refused" if event.answer.refused else "ok"
                        yield sse("done", to_answer_out(event.answer).model_dump(exclude_unset=True))
                    else:
                        yield sse(event.kind, {"text": event.text})
            except RagError as exc:
                # Header 200 đã gửi đi, nên lỗi giữa chừng phải báo bằng một sự kiện SSE.
                yield sse("error", {"message": str(exc)})
            finally:
                telemetry.observe_request("chat_stream", outcome, time.perf_counter() - slot.started)
                slot.release()

        stream = generate()
        # Client ngắt trước khi generator chạy thì `finally` không chạy: chốt thêm bằng finalizer.
        weakref.finalize(stream, slot.release)
        return StreamingResponse(
            stream,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
