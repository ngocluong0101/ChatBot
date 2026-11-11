"""Reranker cross-encoder: chấm lại độ liên quan của (câu hỏi, chunk) chính xác hơn bi-encoder."""

import logging
from collections.abc import Sequence
from typing import Protocol

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    def score(self, query: str, docs: Sequence[Document]) -> list[float]: ...


class CrossEncoderReranker:
    """Bọc sentence_transformers.CrossEncoder; điểm trả về nằm trong [0, 1] (sigmoid)."""

    def __init__(self, model_name: str) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        use_cuda = torch.cuda.is_available()
        logger.info("Nạp reranker %s (%s)", model_name, "cuda/fp16" if use_cuda else "cpu")
        self._model = CrossEncoder(
            model_name,
            max_length=512,
            device="cuda" if use_cuda else "cpu",
            model_kwargs={"torch_dtype": torch.float16} if use_cuda else {},
        )

    def score(self, query: str, docs: Sequence[Document]) -> list[float]:
        if not docs:
            return []
        pairs = [(query, d.page_content) for d in docs]
        return [float(s) for s in self._model.predict(pairs, show_progress_bar=False)]
