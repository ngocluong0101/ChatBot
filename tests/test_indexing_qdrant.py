from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from qdrant_client import QdrantClient

from ragbot import indexing, vectorstore
from ragbot.config import Settings
from ragbot.errors import NoDocumentsError
from ragbot.retrieval import Retriever
from ragbot.vectorstore import QdrantStore, all_chunks
from tests.pdf_factory import build_pdf


class CountingEmbeddings(Embeddings):
    """Giả embedding, ghi lại mọi văn bản đã được yêu cầu embed."""

    def __init__(self):
        self._inner = DeterministicFakeEmbedding(size=16)
        self.embedded: list[str] = []

    def embed_documents(self, texts):
        self.embedded.extend(texts)
        return self._inner.embed_documents(texts)

    def embed_query(self, text):
        return self._inner.embed_query(text)


def write_pdf(path: Path, *lines: str) -> None:
    path.write_bytes(build_pdf([[("text", 50, 800 - 14 * i, line) for i, line in enumerate(lines)]]))


LAW_A = [f"Quy dinh so {i} ve nghi phep nam cua nguoi lao dong trong doanh nghiep." for i in range(1, 9)]
LAW_B = [f"Muc {i} noi ve bao hieu xa hoi bat buoc va che do thai san cua nhan vien." for i in range(1, 9)]


@pytest.fixture
def env(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    write_pdf(docs / "a.pdf", *LAW_A)
    write_pdf(docs / "b.pdf", *LAW_B)
    settings = Settings(
        docs_dir=docs,
        index_dir=tmp_path / "index",
        sources_file=tmp_path / "SOURCES.md",
        chunking="fixed",
        chunk_size=200,
        chunk_overlap=20,
        pdf_loader="pdfplumber",
        vector_store="qdrant",
        qdrant_url=":memory:",
    )
    return settings, docs, QdrantClient(":memory:"), CountingEmbeddings()


def sync(settings, client, embeddings):
    store = QdrantStore.open(settings, embeddings, client)
    return store, indexing.sync_store(settings, store)


def sources_of(store) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in store.all_chunks():
        counts[chunk.metadata["source"]] = counts.get(chunk.metadata["source"], 0) + 1
    return counts


# ---------------------------------------------------------------- dựng lần đầu và nạp lại


def test_first_build_indexes_every_file_and_keeps_metadata(env):
    settings, _, client, emb = env
    store, report = sync(settings, client, emb)

    assert report.added == ["a.pdf", "b.pdf"] and not report.updated and not report.removed
    assert report.chunks_total == store.count() == len(emb.embedded) > 2
    chunks = store.all_chunks()
    assert {c.metadata["source"] for c in chunks} == {"a.pdf", "b.pdf"}
    assert all(c.metadata["page"] == 1 and "start_index" in c.metadata for c in chunks)
    assert [c.metadata["source"] for c in chunks] == sorted(
        c.metadata["source"] for c in chunks
    )  # ổn định theo file


def test_payload_has_hash_ordinal_and_metadata_for_filtering(env):
    settings, _, client, emb = env
    sync(settings, client, emb)
    point = client.scroll("ragbot", limit=1, with_payload=True)[0][0]
    assert {"source", "page", "page_content", "content_hash", "file_hash", "ordinal"} <= set(point.payload)
    assert len(point.payload["content_hash"]) == 16 and len(point.payload["file_hash"]) == 64


def test_second_run_embeds_nothing(env):
    settings, _, client, emb = env
    sync(settings, client, emb)
    emb.embedded.clear()

    store, report = sync(settings, client, emb)

    assert emb.embedded == [] and report.chunks_embedded == 0
    assert report.unchanged == ["a.pdf", "b.pdf"] and not (report.added or report.updated or report.removed)


# ---------------------------------------------------------------- cập nhật từng phần


def test_changing_one_file_reembeds_only_that_file(env):
    settings, docs, client, emb = env
    store, _ = sync(settings, client, emb)
    before_b = [c.page_content for c in store.all_chunks() if c.metadata["source"] == "b.pdf"]
    b_ids = {
        p.id
        for p in client.scroll("ragbot", limit=500, with_payload=["source"])[0]
        if p.payload["source"] == "b.pdf"
    }
    emb.embedded.clear()

    write_pdf(docs / "a.pdf", *LAW_A[:4], "Dong moi them vao sau khi sua file a.")
    store, report = sync(settings, client, emb)

    assert report.updated == ["a.pdf"] and report.unchanged == ["b.pdf"]
    assert emb.embedded and all(
        "nghi phep" in t or "sua file a" in t for t in emb.embedded
    )  # chỉ nội dung của a.pdf
    assert [c.page_content for c in store.all_chunks() if c.metadata["source"] == "b.pdf"] == before_b
    assert b_ids == {
        p.id
        for p in client.scroll("ragbot", limit=500, with_payload=["source"])[0]
        if p.payload["source"] == "b.pdf"
    }
    assert any("sua file a" in c.page_content for c in store.all_chunks())
    assert not any(
        "so 8 ve nghi phep" in c.page_content for c in store.all_chunks()
    )  # chunk cũ của a.pdf đã bị thay


def test_added_and_removed_files_are_picked_up(env):
    settings, docs, client, emb = env
    store, _ = sync(settings, client, emb)

    write_pdf(docs / "c.pdf", "Van ban moi them vao thu muc tai lieu de kiem tra viec cap nhat chi muc.")
    (docs / "b.pdf").unlink()
    store, report = sync(settings, client, emb)

    assert report.added == ["c.pdf"] and report.removed == ["b.pdf"] and report.unchanged == ["a.pdf"]
    assert set(sources_of(store)) == {"a.pdf", "c.pdf"}


def test_changing_chunking_parameters_rebuilds_every_file(env):
    settings, _, client, emb = env
    sync(settings, client, emb)
    emb.embedded.clear()

    _, report = sync(settings.model_copy(update={"chunk_size": 300}), client, emb)
    assert report.updated == ["a.pdf", "b.pdf"] and emb.embedded


def test_changing_a_sources_row_only_rebuilds_that_document_when_chunking_is_legal(env, tmp_path: Path):
    settings, _, client, emb = env
    sources = tmp_path / "SOURCES.md"
    sources.write_text(
        "| File | Tên văn bản |\n|---|---|\n| a.pdf | Luật A |\n| b.pdf | Luật B |\n", encoding="utf-8"
    )
    legal = settings.model_copy(update={"chunking": "legal", "sources_file": sources})
    sync(legal, client, emb)

    sources.write_text(
        "| File | Tên văn bản |\n|---|---|\n| a.pdf | Luật A đổi tên |\n| b.pdf | Luật B |\n",
        encoding="utf-8",
    )
    _, report = sync(legal, client, emb)
    assert report.updated == ["a.pdf"] and report.unchanged == ["b.pdf"]

    # chunking khác "legal" không phụ thuộc SOURCES.md
    sources.write_text("| File | Tên văn bản |\n|---|---|\n| a.pdf | Đổi nữa |\n", encoding="utf-8")
    _, report = sync(settings, client, emb)
    assert not report.updated or report.updated == [
        "a.pdf",
        "b.pdf",
    ]  # chỉ do tham số chunking khác lần trước


def test_scanned_pdf_is_skipped_and_all_scans_is_an_error(env):
    settings, docs, client, emb = env
    (docs / "scan.pdf").write_bytes(build_pdf([[]]))  # trang trống như PDF ảnh scan
    store, report = sync(settings, client, emb)
    assert report.skipped == ["scan.pdf"] and set(sources_of(store)) == {"a.pdf", "b.pdf"}

    only_scan = settings.model_copy(update={"docs_dir": docs / "scan_only"})
    (docs / "scan_only").mkdir()
    (docs / "scan_only" / "scan.pdf").write_bytes(build_pdf([[]]))
    with pytest.raises(NoDocumentsError, match="không trích xuất được"):
        sync(only_scan, QdrantClient(":memory:"), emb)


def test_two_pdfs_with_the_same_name_are_rejected(env):
    settings, docs, client, emb = env
    (docs / "nested").mkdir()
    write_pdf(docs / "nested" / "a.pdf", *LAW_A)
    with pytest.raises(NoDocumentsError, match="cùng tên"):
        sync(settings, client, emb)


def test_missing_or_empty_docs_dir_fails_clearly(env, tmp_path: Path):
    settings, _, client, emb = env
    with pytest.raises(NoDocumentsError, match="không tồn tại"):
        sync(settings.model_copy(update={"docs_dir": tmp_path / "khong-co"}), client, emb)
    (tmp_path / "rong").mkdir()
    with pytest.raises(NoDocumentsError, match="Không có file PDF"):
        sync(settings.model_copy(update={"docs_dir": tmp_path / "rong"}), client, emb)


# ---------------------------------------------------------------- an toàn khi lỗi / ghi dở


def test_failed_embedding_keeps_the_old_chunks(env):
    settings, docs, client, emb = env
    store, _ = sync(settings, client, emb)
    before = store.all_chunks()

    class Boom(CountingEmbeddings):
        def embed_documents(self, texts):
            raise RuntimeError("hết bộ nhớ")

    write_pdf(docs / "a.pdf", *LAW_A[:3], "noi dung moi")
    with pytest.raises(RuntimeError):
        indexing.sync_store(settings, QdrantStore.open(settings, Boom(), client))

    assert QdrantStore.open(settings, emb, client).all_chunks() == before  # chunk cũ còn nguyên


def test_file_whose_chunks_were_half_written_is_reindexed(env):
    settings, _, client, emb = env
    store, _ = sync(settings, client, emb)
    store.delete_file("a.pdf")  # giả lập tiến trình bị ngắt sau khi xoá chunk cũ, trước khi ghi chunk mới

    _, report = sync(settings, client, emb)
    assert report.added == ["a.pdf"] and "a.pdf" in sources_of(QdrantStore.open(settings, emb, client))


def test_inconsistent_file_hashes_within_one_file_force_a_reload(env):
    settings, _, client, emb = env
    store, _ = sync(settings, client, emb)
    point = next(
        p
        for p in client.scroll("ragbot", limit=500, with_payload=["source"])[0]
        if p.payload["source"] == "a.pdf"
    )
    client.set_payload("ragbot", payload={"file_hash": "cu"}, points=[point.id])

    assert store.indexed_files()["a.pdf"] is None
    _, report = sync(settings, client, emb)
    assert report.updated == ["a.pdf"]


# ---------------------------------------------------------------- tìm kiếm


def test_search_returns_cosine_scores_and_can_filter_by_document(env):
    settings, _, client, emb = env
    store, _ = sync(settings, client, emb)
    target = next(c for c in store.all_chunks() if c.metadata["source"] == "b.pdf")

    top = store.similarity_search_with_score(target.page_content, k=3)
    assert top[0][0].page_content == target.page_content and top[0][1] == pytest.approx(1.0, abs=1e-5)
    assert all(abs(score) <= 1.0001 for _, score in top) and [s for _, s in top] == sorted(
        (s for _, s in top), reverse=True
    )

    only_a = store.similarity_search_with_score(target.page_content, k=50, source="a.pdf")
    assert only_a and {d.metadata["source"] for d, _ in only_a} == {"a.pdf"}
    assert store.similarity_search_with_score(target.page_content, k=5, source="khong-co.pdf") == []


def test_hybrid_retriever_builds_bm25_from_the_store_and_filters_by_document(env):
    settings, _, client, emb = env
    store, _ = sync(settings, client, emb)
    retriever = Retriever(settings.model_copy(update={"retrieval_mode": "hybrid", "rerank_model": ""}), store)

    hits = retriever.search("bao hieu xa hoi bat buoc che do thai san")
    assert hits and hits[0][0].metadata["source"] == "b.pdf"  # BM25 bỏ dấu đưa đúng văn bản lên đầu

    filtered = retriever.search("bao hieu xa hoi bat buoc che do thai san", source="a.pdf")
    assert filtered and {d.metadata["source"] for d, _ in filtered} == {"a.pdf"}


def test_all_chunks_falls_back_to_the_faiss_docstore():
    class FakeFaiss:
        docstore = type("D", (), {"_dict": {0: Document(page_content="x", metadata={"source": "a.pdf"})}})()

    assert [c.page_content for c in all_chunks(FakeFaiss())] == ["x"]


# ---------------------------------------------------------------- mở chỉ mục theo cấu hình


def test_load_or_build_index_defaults_to_qdrant_without_pickle(env, tmp_path: Path):
    settings, _, client, emb = env
    assert Settings().vector_store == "qdrant"

    store = indexing.load_or_build_index(settings, emb, client)

    assert isinstance(store, QdrantStore) and store.count() > 0
    assert not list(tmp_path.rglob("*.faiss")) and not list(tmp_path.rglob("*.pkl"))


def test_embedded_on_disk_mode_persists_between_runs(env, tmp_path: Path):
    settings, _, _, emb = env
    local = settings.model_copy(update={"qdrant_url": ""})  # "" = nhúng, lưu ở <index_dir>/qdrant
    first = indexing.load_or_build_index(local, emb)
    total = first.count()
    first.close()
    assert (local.index_dir / "qdrant").is_dir()
    emb.embedded.clear()

    second = indexing.load_or_build_index(local, emb)
    assert second.count() == total and emb.embedded == []  # lần sau không embed lại


def test_faiss_mode_still_works_and_is_selected_explicitly(env, monkeypatch):
    settings, _, _, emb = env
    faiss_settings = settings.model_copy(update={"vector_store": "faiss"})
    store = indexing.load_or_build_index(faiss_settings, emb)
    assert hasattr(store, "docstore") and len(all_chunks(store)) > 2
    assert not isinstance(store, QdrantStore)


def test_content_hash_is_stable_and_sensitive():
    assert vectorstore.content_hash("abc") == vectorstore.content_hash("abc")
    assert vectorstore.content_hash("abc") != vectorstore.content_hash("abd")
