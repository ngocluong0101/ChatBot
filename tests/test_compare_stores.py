"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>
"""

import math
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from qdrant_client import QdrantClient

from evals import compare_stores as cs
from evals.common import write_jsonl
from ragbot import indexing
from ragbot.config import Settings
from ragbot.vectorstore import QdrantStore
from tests.pdf_factory import build_pdf

TOPICS = [
    "nghi phep nam",
    "bao hiem xa hoi",
    "thai san",
    "an toan lao dong",
    "hop dong lao dong",
    "tien luong",
]


class NormalizedFake(Embeddings):
    """Giống `create_embeddings` ở thực tế: vector đã chuẩn hóa độ dài 1 (normalize_embeddings=True).
    Với vector chưa chuẩn hóa, điểm FAISS (tích vô hướng) và điểm Qdrant (cosine) không cùng thang."""

    def __init__(self):
        self._inner = DeterministicFakeEmbedding(size=16)

    @staticmethod
    def _unit(vector):
        norm = math.sqrt(sum(x * x for x in vector))
        return [x / norm for x in vector]

    def embed_documents(self, texts):
        return [self._unit(v) for v in self._inner.embed_documents(texts)]

    def embed_query(self, text):
        return self._unit(self._inner.embed_query(text))


@pytest.fixture
def stores(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    for name in ("a", "b"):
        lines = [
            f"Muc {i} quy dinh ve {TOPICS[(i + len(name)) % 6]} trong van ban {name} so {i}."
            for i in range(14)
        ]
        (docs / f"{name}.pdf").write_bytes(
            build_pdf([[("text", 50, 800 - 14 * i, ln) for i, ln in enumerate(lines)]])
        )
    base = {
        "docs_dir": docs,
        "pdf_loader": "pdfplumber",
        "chunking": "fixed",
        "chunk_size": 200,
        "chunk_overlap": 20,
        "sources_file": tmp_path / "S.md",
    }
    emb = NormalizedFake()
    faiss = indexing.load_or_build_index(
        Settings(**base, index_dir=tmp_path / "f", vector_store="faiss"), emb
    )
    settings = Settings(**base, index_dir=tmp_path / "q", vector_store="qdrant", qdrant_url=":memory:")
    qdrant = indexing.load_or_build_index(settings, emb, QdrantClient(":memory:"))
    return faiss, qdrant, settings, emb


QUERIES = [
    "quy dinh ve nghi phep nam",
    "bao hiem xa hoi",
    "tien luong trong van ban b",
    "thai san so 3",
    "hop dong lao dong",
]


def test_faiss_and_qdrant_over_the_same_chunks_return_identical_rankings(stores):
    faiss, qdrant, settings, _ = stores
    result = cs.compare(faiss, qdrant, settings, QUERIES)

    for mode in cs.MODES:
        assert result[mode]["queries"] == len(QUERIES)
        assert result[mode]["identical_ranking"] == 1.0 and result[mode]["top1_agreement"] == 1.0
        assert result[mode]["mean_overlap_at_k"] == 1.0 and result[mode]["mismatches_shown"] == []
        assert result[mode]["max_abs_score_difference"] < 1e-4


def test_comparison_detects_a_store_that_returns_different_chunks(stores):
    faiss, qdrant, settings, emb = stores
    store = QdrantStore.open(settings, emb, QdrantClient(":memory:"))  # kho rỗng
    result = cs.compare_mode(
        faiss, store, settings.model_copy(update={"retrieval_mode": "dense", "rerank_model": ""}), QUERIES
    )
    assert result["identical_ranking"] == 0.0 and result["mean_overlap_at_k"] == 0.0
    assert len(result["mismatches_shown"]) == len(QUERIES)


def test_load_queries_deduplicates_and_limits(tmp_path: Path):
    path = tmp_path / "q.jsonl"
    write_jsonl(path, [{"question": "a?"}, {"question": "b?"}, {"question": "a?"}, {"question": "c?"}])
    assert cs.load_queries(path) == ["a?", "b?", "c?"]
    assert cs.load_queries(path, limit=2) == ["a?", "b?"]


def test_equivalence_table_is_built_from_the_report(stores):
    faiss, qdrant, settings, _ = stores
    table = cs.equivalence_table(
        {**cs.compare(faiss, qdrant, settings, QUERIES), "meta": {"questions_file": "q.jsonl"}}
    )
    assert "| dense | 5 | 100.0% | 100.0% | 100.0% |" in table and "| hybrid | 5 | 100.0% |" in table
    assert "`q.jsonl`" in table
