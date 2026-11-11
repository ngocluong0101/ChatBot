from types import SimpleNamespace

from langchain_core.documents import Document

from ragbot.config import Settings
from ragbot.retrieval import Retriever, _key, reciprocal_rank_fusion


def doc(name: str, text: str) -> Document:
    return Document(page_content=text, metadata={"source": name, "page": 1})


class StubVectorStore:
    """Giả FAISS: trả các tài liệu theo thứ tự cho trước, ghi lại k được yêu cầu."""

    def __init__(self, scored: list[tuple[Document, float]]):
        self._scored = scored
        self.docstore = SimpleNamespace(_dict={i: d for i, (d, _) in enumerate(scored)})
        self.requested_k: list[int] = []

    def similarity_search_with_score(self, _question: str, k: int):
        self.requested_k.append(k)
        return self._scored[:k]


class StubReranker:
    def __init__(self, scores_by_source: dict[str, float]):
        self._scores = scores_by_source

    def score(self, _query, docs):
        return [self._scores[d.metadata["source"]] for d in docs]


def settings(**kwargs) -> Settings:
    kwargs.setdefault("rerank_model", "")
    kwargs.setdefault("retrieval_mode", "dense")
    return Settings(**kwargs)


A = doc("a.pdf", "Mật khẩu phải có ít nhất 12 ký tự")
B = doc("b.pdf", "Nghỉ phép năm 12 ngày")
C = doc("c.pdf", "Khung giờ cốt lõi 09:30")


def test_rrf_prefers_items_found_by_both_rankers():
    fused = reciprocal_rank_fusion([[_key(A), _key(B)], [_key(C), _key(B)]])
    assert fused[_key(B)] > fused[_key(A)] and fused[_key(B)] > fused[_key(C)]


def test_dense_without_reranker_returns_top_k_with_cosine_scores():
    vs = StubVectorStore([(A, 0.9), (B, 0.6), (C, 0.4)])
    result = Retriever(settings(top_k=2), vs).search("câu hỏi")
    assert [(d.metadata["source"], s) for d, s in result] == [("a.pdf", 0.9), ("b.pdf", 0.6)]
    assert vs.requested_k == [2]


def test_reranker_reorders_candidates_and_replaces_scores():
    vs = StubVectorStore([(A, 0.9), (B, 0.6), (C, 0.4)])
    reranker = StubReranker({"a.pdf": 0.1, "b.pdf": 0.95, "c.pdf": 0.5})
    result = Retriever(settings(top_k=2, fetch_k=3), vs, reranker).search("câu hỏi")
    assert [(d.metadata["source"], s) for d, s in result] == [("b.pdf", 0.95), ("c.pdf", 0.5)]
    assert vs.requested_k == [3], "phải lấy fetch_k ứng viên để rerank, không phải top_k"


def test_hybrid_adds_lexical_only_candidate_with_its_cosine_score():
    # Vector xếp C cuối, nhưng truy vấn không dấu khớp từ khóa của C => BM25 đưa C vào ứng viên.
    vs = StubVectorStore([(A, 0.9), (B, 0.6), (C, 0.1)])
    retriever = Retriever(settings(retrieval_mode="hybrid", fetch_k=1, top_k=5), vs)
    result = retriever.search("khung gio cot loi")
    by_source = {d.metadata["source"]: s for d, s in result}
    assert set(by_source) == {"a.pdf", "c.pdf"}
    assert by_source["c.pdf"] == 0.1, "điểm gate vẫn là cosine thật, không phải điểm RRF"
