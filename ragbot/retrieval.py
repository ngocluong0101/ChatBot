"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>

Truy hồi chunk: dense (vector) | hybrid (vector + BM25, gộp bằng RRF), tùy chọn rerank.

`score` dùng để áp ngưỡng: xác suất của cross-encoder (thang [0, 1]) nếu có reranker,
ngược lại là cosine similarity.
"""

import logging
import threading
from typing import Any

from langchain_core.documents import Document

from ragbot.config import Settings
from ragbot.lexical import BM25Index
from ragbot.rerank import Reranker
from ragbot.telemetry import NOOP, Stage, Telemetry
from ragbot.vectorstore import all_chunks

logger = logging.getLogger(__name__)

RRF_K = 60  # hằng số chuẩn của Reciprocal Rank Fusion
# Với corpus nhỏ, lấy điểm vector của MỌI chunk để chunk chỉ do BM25 tìm ra vẫn có điểm cosine.
MAX_DENSE_POOL = 1000

DocKey = tuple[str, int | None, str]


def _key(doc: Document) -> DocKey:
    return (doc.metadata.get("source", "?"), doc.metadata.get("page"), doc.page_content)


def reciprocal_rank_fusion(rankings: list[list[DocKey]]) -> dict[DocKey, float]:
    fused: dict[DocKey, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            fused[key] = fused.get(key, 0.0) + 1.0 / (RRF_K + rank)
    return fused


class Retriever:
    """Dùng chung cho mọi luồng của API.

    Embedding, cross-encoder và Qdrant nhúng không an toàn khi gọi đồng thời, nên phần tính toán của `search`
    đi qua một khóa; LLM nằm ngoài khóa nên các lượt vẫn chạy chồng nhau (`Settings.max_concurrent_requests`).
    """

    def __init__(
        self,
        settings: Settings,
        vectorstore: Any,
        reranker: Reranker | None = None,
        telemetry: Telemetry = NOOP,
    ) -> None:
        self._settings = settings
        self._vectorstore = vectorstore
        self._reranker = reranker
        self.telemetry = telemetry
        self._lock = threading.Lock()
        self._bm25: BM25Index | None = None
        self._chunks: list[Document] = []
        if settings.retrieval_mode == "hybrid":
            self._chunks = all_chunks(vectorstore)
            self._bm25 = BM25Index(self._chunks)

    def _dense(self, question: str, k: int, source: str | None) -> list[tuple[Document, float]]:
        # `source` (lọc theo văn bản) chỉ truyền khi được yêu cầu, để các kho không hỗ trợ vẫn dùng như cũ.
        kwargs = {"source": source} if source else {}
        results = self._vectorstore.similarity_search_with_score(question, k=k, **kwargs)
        return [(d, float(sc)) for d, sc in results]

    def search(
        self, question: str, source: str | None = None, parent: Stage | None = None
    ) -> list[tuple[Document, float]]:
        """Top-k chunk; `source` giới hạn trong một văn bản (tên file PDF). `parent`: span cha để tracing."""
        t = self.telemetry
        with t.stage("retrieve_lock_wait", parent):
            self._lock.acquire()
        try:
            return self._search(question, source, parent)
        finally:
            self._lock.release()

    def _search(
        self, question: str, source: str | None, parent: Stage | None
    ) -> list[tuple[Document, float]]:
        s, t = self._settings, self.telemetry
        if self._bm25 is None:
            with t.stage("dense", parent):
                candidates = self._dense(question, s.fetch_k if self._reranker else s.top_k, source)
        else:
            candidates = self._hybrid_candidates(question, source, parent)

        if self._reranker is None:
            return candidates[: s.top_k]

        with t.stage("rerank", parent, **{"rerank.pairs": len(candidates)}):
            scores = self._reranker.score(question, [d for d, _ in candidates])
        reranked = sorted(
            zip((d for d, _ in candidates), scores, strict=True), key=lambda x: x[1], reverse=True
        )
        return [(d, float(sc)) for d, sc in reranked[: s.top_k]]

    def _lexical(self, question: str, source: str | None) -> list[Document]:
        assert self._bm25 is not None
        if source is None:
            return [self._chunks[i] for i, _ in self._bm25.top(question, self._settings.fetch_k)]
        ranked = (self._chunks[i] for i, _ in self._bm25.top(question, len(self._chunks)))
        return [c for c in ranked if c.metadata.get("source") == source][: self._settings.fetch_k]

    def _hybrid_candidates(
        self, question: str, source: str | None = None, parent: Stage | None = None
    ) -> list[tuple[Document, float]]:
        assert self._bm25 is not None
        s, t = self._settings, self.telemetry
        pool = min(len(self._chunks), MAX_DENSE_POOL)
        with t.stage("dense", parent):
            dense = self._dense(question, pool, source)
        dense_score = {_key(d): sc for d, sc in dense}
        with t.stage("bm25", parent):
            lexical = self._lexical(question, source)

        fused = reciprocal_rank_fusion([[_key(d) for d, _ in dense[: s.fetch_k]], [_key(d) for d in lexical]])
        docs_by_key = {_key(d): d for d, _ in dense[: s.fetch_k]} | {_key(d): d for d in lexical}
        ordered = sorted(fused, key=lambda key: fused[key], reverse=True)
        # Chunk ngoài top dense vẫn mang điểm cosine thật (nếu nằm trong pool), không thì 0.
        return [(docs_by_key[key], dense_score.get(key, 0.0)) for key in ordered]
