"""Dựng và cache chỉ mục vector (Qdrant mặc định, FAISS để chạy nhanh không cần Docker).

Embedding toàn bộ chunk bằng bge-m3 trên CPU rất chậm, nên chỉ mục được lưu xuống đĩa và dùng lại:

  * Qdrant (`sync_store`): mỗi file PDF có một hash riêng (nội dung file + dòng của nó trong SOURCES.md +
    tham số chunking/embedding). Mỗi lần khởi động chỉ file mới, đổi hoặc bị xoá mới phải xử lý; các file
    còn lại giữ nguyên chunk và embedding đã có. Không dùng pickle.
  * FAISS (`_load_or_build_faiss`): một fingerprint cho cả thư mục; đổi một file là dựng lại tất cả, và lưu
    bằng pickle (chỉ nạp thư mục do chính ứng dụng ghi ra).
"""

import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from ragbot import corpus, ingest
from ragbot.config import Settings
from ragbot.errors import NoDocumentsError
from ragbot.vectorstore import QdrantStore

logger = logging.getLogger(__name__)

# Tăng số này khi đổi logic đọc/chia tài liệu mà fingerprint không phản ánh được
# (ví dụ đổi separators, đổi loader) để buộc mọi cache cũ bị rebuild.
INDEX_SCHEMA_VERSION = 3  # v3: metadata văn bản pháp luật (Điều/Khoản), bộ đọc pdfplumber

META_FILENAME = "meta.json"
# normalize_embeddings=True + MAX_INNER_PRODUCT => điểm trả về chính là cosine similarity.
# (DistanceStrategy.COSINE của LangChain thực chất dùng IndexFlatL2 và công thức 1 - distance,
# nên điểm không phải cosine thật và có thể âm.)
DISTANCE_STRATEGY = DistanceStrategy.MAX_INNER_PRODUCT


def create_embeddings(settings: Settings) -> Embeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compute_fingerprint(settings: Settings) -> str:
    """Hash nội dung mọi PDF và các tham số ảnh hưởng tới nội dung index."""
    digest = hashlib.sha256()
    params = (
        f"schema={INDEX_SCHEMA_VERSION};embedding={settings.embedding_model};"
        f"chunking={settings.chunking};chunk_size={settings.chunk_size};"
        f"chunk_overlap={settings.chunk_overlap};pdf_loader={settings.pdf_loader}"
    )
    digest.update(params.encode())
    # Tên/số hiệu văn bản nằm trong nội dung chunk nên đổi SOURCES.md cũng phải dựng lại index.
    if settings.chunking == "legal" and settings.sources_file.is_file():
        digest.update(_file_sha256(settings.sources_file).encode())
    for pdf in sorted(settings.docs_dir.glob("**/*.pdf")):
        digest.update(pdf.relative_to(settings.docs_dir).as_posix().encode())
        digest.update(_file_sha256(pdf).encode())
    return digest.hexdigest()


def _try_load_cached(index_dir: Path, fingerprint: str, embeddings: Embeddings) -> FAISS | None:
    meta_path = index_dir / META_FILENAME
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") != fingerprint:
            logger.info("Tài liệu hoặc cấu hình đã thay đổi, sẽ dựng lại index")
            return None
        # allow_dangerous_deserialization: FAISS.load_local dùng pickle. Chấp nhận được ở đây vì
        # index_dir do chính ứng dụng ghi ra; KHÔNG trỏ nó tới thư mục không tin cậy.
        return FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
            distance_strategy=DISTANCE_STRATEGY,
        )
    except Exception:  # cache hỏng (json/pickle/faiss) không được làm chết app: dựng lại
        logger.warning("Không nạp được index đã cache, sẽ dựng lại", exc_info=True)
        return None


def _save_index(vectorstore: FAISS, index_dir: Path, fingerprint: str, chunks: int) -> None:
    """Ghi vào thư mục tạm rồi đổi tên, để lần ghi dở không để lại cache nửa vời."""
    tmp_dir = index_dir.with_name(index_dir.name + ".tmp")
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        vectorstore.save_local(str(tmp_dir))
        meta = {
            "fingerprint": fingerprint,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "chunks": chunks,
        }
        (tmp_dir / META_FILENAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        shutil.rmtree(index_dir, ignore_errors=True)
        tmp_dir.replace(index_dir)
        logger.info("Đã lưu index (%d chunk) vào %s", chunks, index_dir)
    except OSError:
        # Cache chỉ là tối ưu hóa: không ghi được thì vẫn dùng index trong RAM.
        logger.warning("Không lưu được index vào %s", index_dir, exc_info=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)


@dataclass
class SyncReport:
    """Kết quả một lần đồng bộ chỉ mục Qdrant với thư mục tài liệu."""

    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # file không trích được chữ (PDF scan)
    chunks_embedded: int = 0
    chunks_total: int = 0
    seconds: float = 0.0

    def summary(self) -> str:
        return (
            f"thêm {len(self.added)}, cập nhật {len(self.updated)}, xoá {len(self.removed)}, "
            f"giữ nguyên {len(self.unchanged)} file; embedding {self.chunks_embedded} chunk mới "
            f"(tổng {self.chunks_total}) trong {self.seconds:.1f} giây"
        )


def file_hashes(settings: Settings, pdfs: list[Path]) -> dict[str, str]:
    """Hash hiệu lực của từng file: đổi nội dung file, dòng của nó trong SOURCES.md (khi chunking="legal")
    hoặc bất kỳ tham số ảnh hưởng chunk/embedding đều làm đổi hash, chỉ file bị ảnh hưởng mới nạp lại."""
    params = (
        f"schema={INDEX_SCHEMA_VERSION};embedding={settings.embedding_model};chunking={settings.chunking};"
        f"chunk_size={settings.chunk_size};chunk_overlap={settings.chunk_overlap};"
        f"pdf_loader={settings.pdf_loader}"
    )
    sources = corpus.load_sources(settings.sources_file) if settings.chunking == "legal" else {}
    hashes: dict[str, str] = {}
    for pdf in pdfs:
        if pdf.name in hashes:
            raise NoDocumentsError(
                f"Hai file PDF cùng tên '{pdf.name}' trong {settings.docs_dir}: hãy đổi tên một file"
            )
        digest = hashlib.sha256(params.encode())
        digest.update(_file_sha256(pdf).encode())
        digest.update(repr(sources.get(pdf.name)).encode())
        hashes[pdf.name] = digest.hexdigest()
    return hashes


def sync_store(settings: Settings, store: QdrantStore) -> SyncReport:
    """Đưa kho Qdrant về đúng trạng thái của thư mục tài liệu, chỉ xử lý file thay đổi."""
    started = time.perf_counter()
    pdfs = ingest.list_pdfs(settings.docs_dir)
    desired = file_hashes(settings, pdfs)
    indexed = store.indexed_files()
    report = SyncReport()

    for name in sorted(set(indexed) - set(desired)):
        store.delete_file(name)
        report.removed.append(name)

    by_name = {p.name: p for p in pdfs}
    for name in sorted(desired):
        if indexed.get(name) == desired[name]:
            report.unchanged.append(name)
            continue
        try:
            pages = ingest.load_documents(settings, only=[by_name[name]])
        except NoDocumentsError:
            logger.warning("Không trích được chữ từ %s (PDF scan?): bỏ qua", name)
            store.delete_file(name)
            report.skipped.append(name)
            continue
        chunks = ingest.split_documents(pages, settings)
        logger.info("Đang tạo embedding cho %s (%d chunk)...", name, len(chunks))
        report.chunks_embedded += store.replace_file(name, desired[name], chunks)
        (report.updated if name in indexed else report.added).append(name)

    report.chunks_total = store.count()
    report.seconds = time.perf_counter() - started
    if report.chunks_total == 0:
        raise NoDocumentsError(
            f"Đọc được các file PDF trong {settings.docs_dir} nhưng không trích xuất được văn bản. "
            "Nếu là PDF scan, cần cài Tesseract (kèm gói ngôn ngữ 'vie') để OCR."
        )
    return report


def load_or_build_index(settings: Settings, embeddings: Embeddings | None = None, client=None):
    """Mở chỉ mục theo `settings.vector_store`, dựng hoặc cập nhật nếu tài liệu đã đổi.

    Trả về đối tượng có `similarity_search_with_score(query, k)`; liệt kê chunk bằng `vectorstore.all_chunks`.
    """
    embeddings = embeddings or create_embeddings(settings)
    if settings.vector_store == "faiss":
        return _load_or_build_faiss(settings, embeddings)
    store = QdrantStore.open(settings, embeddings, client)
    report = sync_store(settings, store)
    logger.info("Chỉ mục Qdrant: %s", report.summary())
    return store


def _load_or_build_faiss(settings: Settings, embeddings: Embeddings) -> FAISS:
    fingerprint = compute_fingerprint(settings)

    cached = _try_load_cached(settings.index_dir, fingerprint, embeddings)
    if cached is not None:
        logger.info("Đã nạp index từ cache %s", settings.index_dir)
        return cached

    docs = ingest.load_documents(settings)
    splits = ingest.split_documents(docs, settings)
    logger.info("Đang tạo embedding cho %d chunk (lần đầu có thể mất vài phút)...", len(splits))
    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embeddings,
        distance_strategy=DISTANCE_STRATEGY,
    )
    _save_index(vectorstore, settings.index_dir, fingerprint, len(splits))
    return vectorstore
