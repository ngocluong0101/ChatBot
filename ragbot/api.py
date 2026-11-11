"""API HTTP (FastAPI): /chat, /chat/stream (SSE), /health và giao diện web ở /.

Chạy:  uvicorn ragbot.api:app --host 0.0.0.0 --port 8000

Các model (embedding, reranker, LLM) dùng chung một GPU/CPU và không đảm bảo thread-safe, nên mọi
lượt hỏi được xếp hàng qua một khóa: phù hợp triển khai nội bộ một nhóm nhỏ, không phải dịch vụ
đồng thời cao. Muốn mở rộng, chạy nhiều tiến trình hoặc thay lớp suy luận bằng một server riêng.
"""

import json
import logging
import secrets
import threading
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from ragbot.config import Settings, get_settings
from ragbot.errors import RagError
from ragbot.pipeline import Answer, RagPipeline, Turn, build_pipeline

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


class AnswerOut(BaseModel):
    text: str
    refused: bool
    question: str  # câu hỏi độc lập thực sự dùng để truy hồi (có thể đã được viết lại)
    sources: list[SourceOut]


def to_answer_out(answer: Answer) -> AnswerOut:
    return AnswerOut(
        text=answer.text,
        refused=answer.refused,
        question=answer.question,
        sources=[
            SourceOut(file=s.file, page=s.page, section=s.section, score=s.score) for s in answer.sources
        ],
    )


def to_turns(history: list[ChatMessage]) -> list[Turn]:
    return [Turn(m.role, m.content) for m in history]


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(pipeline: RagPipeline | None = None, settings: Settings | None = None) -> FastAPI:
    """`pipeline` truyền vào để test; mặc định dựng pipeline thật khi khởi động."""
    settings = settings or get_settings()
    lock = threading.Lock()

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

    @app.post("/chat", response_model=AnswerOut, dependencies=auth)
    def chat(body: ChatRequest, request: Request) -> AnswerOut:
        with lock:
            try:
                answer = request.app.state.pipeline.ask(body.question, to_turns(body.history))
            except RagError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        return to_answer_out(answer)

    @app.post("/chat/stream", dependencies=auth)
    def chat_stream(body: ChatRequest, request: Request) -> StreamingResponse:
        active_pipeline: RagPipeline = request.app.state.pipeline

        def generate() -> Iterator[str]:
            with lock:
                try:
                    for event in active_pipeline.ask_stream(body.question, to_turns(body.history)):
                        if event.kind == "done" and event.answer is not None:
                            yield sse("done", to_answer_out(event.answer).model_dump())
                        else:
                            yield sse(event.kind, {"text": event.text})
                except RagError as exc:
                    # Header 200 đã gửi đi, nên lỗi giữa chừng phải báo bằng một sự kiện SSE.
                    yield sse("error", {"message": str(exc)})

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
