"""Truy hồi chunk: dense (vector) | hybrid (vector + BM25, gộp bằng RRF), tùy chọn rerank.

Điểm trả về (`score`) là điểm dùng để áp ngưỡng "có đủ liên quan không":
  * có reranker  -> xác suất do cross-encoder chấm (thang [0, 1])
  * không có     -> cosine similarity của vector
"""

import logging

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from ragbot.config import Settings
from ragbot.lexical import BM25Index
from ragbot.rerank import Reranker

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
    def __init__(self, settings: Settings, vectorstore: FAISS, reranker: Reranker | None = None) -> None:
        self._settings = settings
        self._vectorstore = vectorstore
        self._reranker = reranker
        self._bm25: BM25Index | None = None
        self._chunks: list[Document] = []
        if settings.retrieval_mode == "hybrid":
            self._chunks = list(vectorstore.docstore._dict.values())  # chưa có API công khai
            self._bm25 = BM25Index(self._chunks)

    def search(self, question: str) -> list[tuple[Document, float]]:
        s = self._settings
        if self._bm25 is None:
            k = s.fetch_k if self._reranker else s.top_k
            results = self._vectorstore.similarity_search_with_score(question, k=k)
            candidates = [(d, float(sc)) for d, sc in results]
        else:
            candidates = self._hybrid_candidates(question)

        if self._reranker is None:
            return candidates[: s.top_k]

        scores = self._reranker.score(question, [d for d, _ in candidates])
        reranked = sorted(
            zip((d for d, _ in candidates), scores, strict=True), key=lambda x: x[1], reverse=True
        )
        return [(d, float(sc)) for d, sc in reranked[: s.top_k]]

    def _hybrid_candidates(self, question: str) -> list[tuple[Document, float]]:
        assert self._bm25 is not None
        s = self._settings
        pool = min(len(self._chunks), MAX_DENSE_POOL)
        results = self._vectorstore.similarity_search_with_score(question, k=pool)
        dense = [(d, float(sc)) for d, sc in results]
        dense_score = {_key(d): sc for d, sc in dense}
        lexical = [self._chunks[i] for i, _ in self._bm25.top(question, s.fetch_k)]

        fused = reciprocal_rank_fusion([[_key(d) for d, _ in dense[: s.fetch_k]], [_key(d) for d in lexical]])
        docs_by_key = {_key(d): d for d, _ in dense[: s.fetch_k]} | {_key(d): d for d in lexical}
        ordered = sorted(fused, key=lambda key: fused[key], reverse=True)
        # Chunk ngoài top dense vẫn mang điểm cosine thật (nếu nằm trong pool), không thì 0.
        return [(docs_by_key[key], dense_score.get(key, 0.0)) for key in ordered]
