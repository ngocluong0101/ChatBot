"""Cấu hình ứng dụng, đọc từ biến môi trường / file `.env` (tiền tố `RAG_`).

Ví dụ trong `.env`:
    RAG_LLM_MODEL=qwen2.5:7b-instruct
    RAG_TOP_K=4
"""

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

    # Dữ liệu. Đường dẫn tương đối được hiểu là tương đối so với thư mục gốc dự án,
    # để chạy từ thư mục nào cũng được.
    papers_dir: Path = Path("papers")
    index_dir: Path = Path(".index")

    # Model
    embedding_model: str = "BAAI/bge-m3"
    # Đo trên bộ eval (3 lần chạy): qwen2.5:7b ổn định 0.962, llama3.2:3b dao động 0.943-0.962.
    llm_model: str = "qwen2.5:7b-instruct"
    ollama_base_url: str = "http://localhost:11434"
    # Ollama âm thầm cắt đầu prompt khi vượt context window, nên phải đặt tường minh.
    num_ctx: int = 4096
    temperature: float = 0.0

    # Chunking. "section": tách theo mục đánh số "1., 2., ..." (kèm tiêu đề tài liệu);
    # "fixed": cắt theo độ dài. chunk_size/overlap dùng cho "fixed" và cho mục dài quá chunk_size.
    chunking: Literal["fixed", "section"] = "section"
    chunk_size: int = 1200
    chunk_overlap: int = 200

    # Retrieval. "hybrid" = vector + BM25 (bỏ dấu) gộp bằng RRF; mỗi nhánh lấy fetch_k ứng viên.
    retrieval_mode: Literal["dense", "hybrid"] = "hybrid"
    fetch_k: int = 10
    # Reranker cross-encoder (rỗng = tắt). Khi bật, rerank_threshold thay cho score_threshold.
    # 0.20 nằm giữa câu không-có-đáp-án cao nhất bị chặn (0.154) và câu hợp lệ thấp nhất (không tính
    # câu không dấu) là 0.257 trên bộ eval; xem evals/after-step4.json.
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_threshold: float = 0.20
    top_k: int = 5
    # Ngưỡng cosine similarity. Dưới ngưỡng => coi như tài liệu không có thông tin.
    # Hiệu chỉnh thủ công trên 12 câu hỏi: câu trong phạm vi có top-1 >= 0.57, câu ngoài chủ đề
    # <= 0.37, câu "gần chủ đề nhưng tài liệu không có" tới ~0.50.
    score_threshold: float = 0.50
    # Chỉ hiển thị nguồn của chunk có điểm >= top-1 trừ biên độ này.
    source_score_margin: float = 0.08

    # Hội thoại. Viết lại câu hỏi nối tiếp ("còn trường hợp nghỉ ốm?") và câu hỏi gõ không dấu
    # thành câu độc lập, có dấu trước khi truy hồi. Chỉ tốn thêm một lượt gọi LLM khi cần.
    rewrite_queries: bool = True
    max_history_turns: int = 6  # số lượt hỏi-đáp gần nhất được giữ làm ngữ cảnh

    # API. Để trống = không yêu cầu khóa; đặt giá trị thì mọi endpoint (trừ /health và trang chủ)
    # yêu cầu header X-API-Key khớp.
    api_key: str = ""

    log_level: str = "INFO"

    @field_validator("papers_dir", "index_dir")
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
        if self.max_history_turns < 0:
            raise ValueError("max_history_turns phải >= 0")
        if not 0.0 <= self.rerank_threshold <= 1.0:
            raise ValueError("rerank_threshold là xác suất nên phải nằm trong [0, 1]")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap phải nằm trong [0, chunk_size)")
        if not -1.0 <= self.score_threshold <= 1.0:
            raise ValueError("score_threshold là cosine similarity nên phải nằm trong [-1, 1]")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
