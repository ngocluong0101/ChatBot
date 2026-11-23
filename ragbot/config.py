"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>

Cấu hình ứng dụng, đọc từ biến môi trường / file `.env` (tiền tố `RAG_`).

Ví dụ trong `.env`:
    RAG_LLM_MODEL=qwen2.5:7b-instruct
    RAG_TOP_K=4
"""

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        validate_default=True,
    )

    # Đường dẫn tương đối tính từ thư mục gốc dự án. PDF không nằm trong repo (xem data/corpus/README.md).
    docs_dir: Path = Path("data/corpus")
    index_dir: Path = Path(".index")
    # Bảng tên/số hiệu văn bản (data/SOURCES.md); không có file thì lấy tên từ tên file.
    sources_file: Path = Path("data/SOURCES.md")
    # Kho vector: "qdrant" (cập nhật từng file) hoặc "faiss" (nhanh, pickle, dựng lại toàn bộ khi đổi file).
    vector_store: Literal["qdrant", "faiss"] = "qdrant"
    # Qdrant: "" = nhúng (<index_dir>/qdrant, một tiến trình), URL = server, ":memory:" = test.
    qdrant_url: str = ""
    qdrant_collection: str = "ragbot"
    # "pdfplumber": giữ thứ tự dòng, nhận diện bảng (cần cho "legal"); "unstructured": đọc theo trang.
    pdf_loader: Literal["unstructured", "pdfplumber"] = "pdfplumber"

    # Model
    embedding_model: str = "BAAI/bge-m3"
    # Đo trên bộ eval (3 lần chạy): qwen2.5:7b ổn định 0.962, llama3.2:3b dao động 0.943-0.962.
    llm_model: str = "qwen2.5:7b-instruct"
    ollama_base_url: str = "http://localhost:11434"
    # Ollama âm thầm cắt đầu prompt khi vượt context window, nên phải đặt tường minh.
    num_ctx: int = 4096
    temperature: float = 0.0

    # Chunking: "legal" (Chương/Mục/Điều/Khoản, bảng riêng; không có "Điều" thì rơi về "section" rồi "fixed"),
    # "section" (mục "1., 2."), "fixed" (theo độ dài). chunk_size/overlap áp cho "fixed" và đơn vị quá dài.
    chunking: Literal["fixed", "section", "legal"] = "legal"
    chunk_size: int = 1200
    chunk_overlap: int = 200

    # Retrieval. "hybrid" = vector + BM25 (bỏ dấu) gộp bằng RRF; mỗi nhánh lấy fetch_k ứng viên.
    retrieval_mode: Literal["dense", "hybrid"] = "hybrid"
    fetch_k: int = 10
    # Reranker cross-encoder (rỗng = tắt); khi bật, rerank_threshold thay score_threshold.
    # 0.20 nằm giữa câu không-đáp-án cao nhất (0.154) và câu hợp lệ thấp nhất (0.257): evals/after-step4.json.
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_threshold: float = 0.20
    top_k: int = 5
    # Ngưỡng cosine: dưới ngưỡng coi như tài liệu không có thông tin (chỉnh tay trên 12 câu hỏi).
    score_threshold: float = 0.50
    # Chỉ hiển thị nguồn của chunk có điểm >= top-1 trừ biên độ này.
    source_score_margin: float = 0.08

    # Hội thoại: viết lại câu hỏi nối tiếp/không dấu thành câu độc lập trước khi truy hồi (thêm một lượt LLM).
    rewrite_queries: bool = True
    max_history_turns: int = 6  # số lượt hỏi-đáp gần nhất được giữ làm ngữ cảnh

    # API: để trống = không cần khóa; đặt giá trị thì endpoint (trừ /health, trang chủ) cần header X-API-Key.
    api_key: str = ""

    # Đồng thời: số lượt hỏi xử lý cùng lúc (semaphore), nên bằng OLLAMA_NUM_PARALLEL của Ollama.
    # Chờ quá queue_timeout_seconds thì trả 503 + Retry-After.
    max_concurrent_requests: int = 1
    queue_timeout_seconds: float = 120.0
    # Thiết bị embedding/reranker: "auto" (GPU nếu có CUDA), "cpu" hoặc "cuda". Embedding trên CPU nhường
    # ~2,2 GB VRAM cho Ollama; float16 giảm nửa VRAM nhưng điểm cosine lệch nhẹ so với chỉ mục float32.
    embedding_device: Literal["auto", "cpu", "cuda"] = "auto"
    rerank_device: Literal["auto", "cpu", "cuda"] = "auto"
    embedding_dtype: Literal["float32", "float16"] = "float32"

    # Quan sát. Tracing OpenTelemetry gửi tới Jaeger (OTLP/HTTP, cổng 4318); /metrics định dạng Prometheus.
    tracing_enabled: bool = False
    otlp_endpoint: str = "http://localhost:4318"
    service_name: str = "ragbot"
    metrics_enabled: bool = True

    log_level: str = "INFO"

    @field_validator("docs_dir", "index_dir", "sources_file")
    @classmethod
    def _resolve_relative(cls, value: Path) -> Path:
        return value if value.is_absolute() else PROJECT_ROOT / value

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        level = value.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"log_level không hợp lệ: {value!r}")
        return level

    @property
    def active_threshold(self) -> float:
        """Ngưỡng đang áp dụng cho `score` của retrieval (xem ragbot/retrieval.py)."""
        return self.rerank_threshold if self.rerank_model else self.score_threshold

    @model_validator(mode="after")
    def _check_ranges(self) -> "Settings":
        if self.chunk_size <= 0 or self.top_k <= 0 or self.num_ctx <= 0 or self.fetch_k <= 0:
            raise ValueError("chunk_size, top_k, fetch_k và num_ctx phải > 0")
        if self.max_concurrent_requests < 1 or self.queue_timeout_seconds <= 0:
            raise ValueError("max_concurrent_requests phải >= 1 và queue_timeout_seconds phải > 0")
        if self.max_history_turns < 0:
            raise ValueError("max_history_turns phải >= 0")
        if not 0.0 <= self.rerank_threshold <= 1.0:
            raise ValueError("rerank_threshold là xác suất nên phải nằm trong [0, 1]")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap phải nằm trong [0, chunk_size)")
        if not -1.0 <= self.score_threshold <= 1.0:
            raise ValueError("score_threshold là cosine similarity nên phải nằm trong [-1, 1]")
        return self


def parse_overrides(pairs: Sequence[str]) -> dict[str, str]:
    """Đọc các tham số dòng lệnh dạng `--set khóa=giá_trị` thành dict để truyền vào Settings(**...)."""
    overrides: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or key not in Settings.model_fields:
            raise SystemExit(f"--set không hợp lệ: {pair!r}. Khóa hợp lệ: {', '.join(Settings.model_fields)}")
        overrides[key] = value
    return overrides


@lru_cache
def get_settings() -> Settings:
    return Settings()
