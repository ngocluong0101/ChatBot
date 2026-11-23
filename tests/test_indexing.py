"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>
"""

from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from ragbot import indexing
from ragbot.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    papers = tmp_path / "docs"
    papers.mkdir()
    (papers / "a.pdf").write_bytes(b"alpha")
    (papers / "b.pdf").write_bytes(b"beta")
    # Các test dưới đây kiểm tra đường FAISS (pickle, một fingerprint cho cả thư mục); đường Qdrant ở
    # tests/test_indexing_qdrant.py.
    return Settings(docs_dir=papers, index_dir=tmp_path / "index", vector_store="faiss")


@pytest.fixture
def fake_ingest(monkeypatch: pytest.MonkeyPatch):
    """Thay bước đọc/chia PDF bằng dữ liệu giả và đếm số lần nó được gọi."""
    calls = {"load": 0}

    def fake_load(_settings):
        calls["load"] += 1
        return [
            Document(page_content="làm việc từ xa tối đa 2 ngày", metadata={"source": "a.pdf", "page": 1}),
            Document(page_content="nghỉ phép năm 12 ngày", metadata={"source": "b.pdf", "page": 1}),
        ]

    monkeypatch.setattr(indexing.ingest, "load_documents", fake_load)
    monkeypatch.setattr(indexing.ingest, "split_documents", lambda docs, _settings: docs)
    return calls


EMB = DeterministicFakeEmbedding(size=16)


def test_fingerprint_is_stable(settings: Settings):
    assert indexing.compute_fingerprint(settings) == indexing.compute_fingerprint(settings)


def test_fingerprint_changes_when_pdf_content_changes(settings: Settings):
    before = indexing.compute_fingerprint(settings)
    (settings.docs_dir / "a.pdf").write_bytes(b"alpha edited")
    assert indexing.compute_fingerprint(settings) != before


def test_fingerprint_changes_when_pdf_added(settings: Settings):
    before = indexing.compute_fingerprint(settings)
    (settings.docs_dir / "c.pdf").write_bytes(b"gamma")
    assert indexing.compute_fingerprint(settings) != before


def test_fingerprint_changes_when_chunking_config_changes(settings: Settings):
    before = indexing.compute_fingerprint(settings)
    other = settings.model_copy(update={"chunk_size": 800})
    assert indexing.compute_fingerprint(other) != before


def test_second_run_uses_cache(settings: Settings, fake_ingest):
    first = indexing.load_or_build_index(settings, EMB)
    second = indexing.load_or_build_index(settings, EMB)

    assert fake_ingest["load"] == 1, "lần 2 phải nạp từ cache, không đọc lại PDF"
    hit = second.similarity_search_with_score("làm việc từ xa tối đa 2 ngày", k=1)[0][0]
    assert hit.metadata == {"source": "a.pdf", "page": 1}
    assert first.index.ntotal == second.index.ntotal == 2


def test_cache_invalidated_when_pdf_changes(settings: Settings, fake_ingest):
    indexing.load_or_build_index(settings, EMB)
    (settings.docs_dir / "a.pdf").write_bytes(b"alpha edited")
    indexing.load_or_build_index(settings, EMB)
    assert fake_ingest["load"] == 2


def test_corrupt_cache_is_rebuilt(settings: Settings, fake_ingest):
    indexing.load_or_build_index(settings, EMB)
    (settings.index_dir / "index.faiss").write_bytes(b"garbage")

    vs = indexing.load_or_build_index(settings, EMB)

    assert fake_ingest["load"] == 2
    assert vs.index.ntotal == 2


def test_no_leftover_tmp_dir(settings: Settings, fake_ingest):
    indexing.load_or_build_index(settings, EMB)
    assert not settings.index_dir.with_name("index.tmp").exists()


def test_fingerprint_depends_on_pdf_loader_and_legal_sources(settings: Settings, tmp_path: Path):
    unstructured = settings.model_copy(update={"pdf_loader": "unstructured"})
    assert indexing.compute_fingerprint(settings) != indexing.compute_fingerprint(unstructured)

    sources = tmp_path / "SOURCES.md"
    sources.write_text("| File | Tên văn bản |\n|---|---|\n| a.pdf | Luật A |\n", encoding="utf-8")
    legal = settings.model_copy(update={"chunking": "legal", "sources_file": sources})
    first = indexing.compute_fingerprint(legal)
    sources.write_text("| File | Tên văn bản |\n|---|---|\n| a.pdf | Luật A sửa tên |\n", encoding="utf-8")
    assert indexing.compute_fingerprint(legal) != first

    # chunking khác "legal" không quan tâm SOURCES.md
    fixed = settings.model_copy(update={"chunking": "fixed", "sources_file": sources})
    fingerprint = indexing.compute_fingerprint(fixed)
    sources.write_text("| File | Tên văn bản |\n|---|---|\n| a.pdf | Đổi nữa |\n", encoding="utf-8")
    assert indexing.compute_fingerprint(fixed) == fingerprint


def test_create_embeddings_passes_device_and_dtype(monkeypatch):
    captured = {}

    class Recorder:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(indexing, "HuggingFaceEmbeddings", Recorder)
    indexing.create_embeddings(Settings())
    assert captured["model_kwargs"] == {}  # mặc định: để sentence-transformers tự chọn thiết bị, float32

    indexing.create_embeddings(Settings(embedding_device="cpu", embedding_dtype="float16"))
    assert captured["model_kwargs"]["device"] == "cpu"
    assert str(captured["model_kwargs"]["model_kwargs"]["torch_dtype"]) == "torch.float16"
