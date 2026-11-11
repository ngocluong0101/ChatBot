"""Dựng và cache FAISS index.

Embedding toàn bộ chunk bằng bge-m3 trên CPU rất chậm, nên index được lưu xuống đĩa cùng một
"fingerprint" (hash nội dung PDF + tham số ảnh hưởng tới chunk/embedding). Lần chạy sau nếu
fingerprint khớp thì nạp lại index, bỏ qua cả bước đọc PDF lẫn embedding.
"""

import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from ragbot import ingest
from ragbot.config import Settings

logger = logging.getLogger(__name__)

# Tăng số này khi đổi logic đọc/chia tài liệu mà fingerprint không phản ánh được
# (ví dụ đổi separators, đổi loader) để buộc mọi cache cũ bị rebuild.
INDEX_SCHEMA_VERSION = 1

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
        f"chunk_size={settings.chunk_size};chunk_overlap={settings.chunk_overlap}"
    )
    digest.update(params.encode())
    for pdf in sorted(settings.papers_dir.glob("**/*.pdf")):
        digest.update(pdf.relative_to(settings.papers_dir).as_posix().encode())
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


def load_or_build_index(settings: Settings, embeddings: Embeddings | None = None) -> FAISS:
    embeddings = embeddings or create_embeddings(settings)
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
