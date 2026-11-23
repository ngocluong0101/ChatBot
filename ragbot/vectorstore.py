"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>

Kho vector Qdrant, cùng giao diện tìm kiếm mà `Retriever` đã dùng với FAISS.

Mỗi chunk là một point có payload đầy đủ metadata, nên có thể:
  * thêm / xoá / thay chunk của MỘT file mà không đụng file khác (xem `indexing.sync_store`);
  * lọc theo văn bản ngay trong truy vấn (`similarity_search_with_score(..., source="file.pdf")`);
  * đọc lại chunk (để dựng BM25) mà không cần pickle.

Điểm trả về là cosine similarity (embedding đã chuẩn hóa), cùng thang với FAISS nên ngưỡng không đổi.
"""

import hashlib
import logging
import uuid
from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ragbot.config import Settings

logger = logging.getLogger(__name__)

POINT_NAMESPACE = uuid.UUID("6f1d3c8e-5a41-4c0e-9d57-2b7f0a8e4c11")
# Các khóa payload do kho tự thêm; phần còn lại của payload chính là `Document.metadata`.
RESERVED = ("page_content", "content_hash", "file_hash", "ordinal")
SCROLL_PAGE = 512
EMBED_BATCH = 32
UPSERT_BATCH = 64


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def all_chunks(store: Any) -> list[Document]:
    """Mọi chunk trong kho (dùng để dựng BM25 và để kiểm tra). Hỗ trợ cả FAISS (docstore) lẫn Qdrant."""
    getter = getattr(store, "all_chunks", None)
    if getter is not None:
        return list(getter())
    return list(store.docstore._dict.values())  # FAISS chưa có API công khai để liệt kê tài liệu


def open_client(settings: Settings) -> Any:
    from qdrant_client import QdrantClient

    url = settings.qdrant_url
    if url == ":memory:":
        return QdrantClient(":memory:")
    if url:
        return QdrantClient(url=url, timeout=60)
    path = settings.index_dir / "qdrant"
    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


class QdrantStore:
    def __init__(
        self, client: Any, collection: str, embeddings: Embeddings, payload_index: bool = False
    ) -> None:
        self._client = client
        self._collection = collection
        self._embeddings = embeddings
        self._payload_index = payload_index
        self._ensure_collection()

    @classmethod
    def open(cls, settings: Settings, embeddings: Embeddings, client: Any = None) -> "QdrantStore":
        server = bool(settings.qdrant_url) and settings.qdrant_url != ":memory:"
        return cls(
            client or open_client(settings), settings.qdrant_collection, embeddings, payload_index=server
        )

    # ------------------------------------------------------------------ collection

    def _ensure_collection(self) -> None:
        from qdrant_client import models

        if self._client.collection_exists(self._collection):
            return
        size = len(self._embeddings.embed_query("dimension probe"))
        self._client.create_collection(
            self._collection,
            vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE),
        )
        if self._payload_index:  # chỉ có tác dụng với server; chế độ nhúng bỏ qua và phát cảnh báo
            self._client.create_payload_index(
                self._collection, "source", field_schema=models.PayloadSchemaType.KEYWORD
            )

    def count(self) -> int:
        return int(self._client.count(self._collection, exact=True).count)

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close:
            close()

    # ------------------------------------------------------------------ ghi

    @staticmethod
    def _source_filter(source: str) -> Any:
        from qdrant_client import models

        return models.Filter(
            must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))]
        )

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH):
            vectors.extend(self._embeddings.embed_documents(list(texts[start : start + EMBED_BATCH])))
        return vectors

    def delete_file(self, source: str) -> None:
        from qdrant_client import models

        self._client.delete(
            self._collection, points_selector=models.FilterSelector(filter=self._source_filter(source))
        )

    def replace_file(self, source: str, file_hash: str, chunks: Sequence[Document]) -> int:
        """Thay toàn bộ chunk của `source`. Embedding tính TRƯỚC khi xoá chunk cũ; nếu bị ngắt sau khi
        xoá, lần đồng bộ sau thấy file chưa có trong kho và làm lại."""
        from qdrant_client import models

        vectors = self._embed([c.page_content for c in chunks])
        self.delete_file(source)
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(POINT_NAMESPACE, f"{source}|{ordinal}")),
                vector=vector,
                payload={
                    **chunk.metadata,
                    "source": source,
                    "page_content": chunk.page_content,
                    "content_hash": content_hash(chunk.page_content),
                    "file_hash": file_hash,
                    "ordinal": ordinal,
                },
            )
            for ordinal, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]
        for start in range(0, len(points), UPSERT_BATCH):
            self._client.upsert(self._collection, points=points[start : start + UPSERT_BATCH])
        return len(points)

    # ------------------------------------------------------------------ đọc

    @staticmethod
    def _to_document(payload: dict[str, Any]) -> Document:
        metadata = {k: v for k, v in payload.items() if k not in RESERVED}
        return Document(page_content=payload["page_content"], metadata=metadata)

    def _scroll(self, payload: Any, scroll_filter: Any = None) -> list[Any]:
        points: list[Any] = []
        offset = None
        while True:
            batch, offset = self._client.scroll(
                self._collection,
                scroll_filter=scroll_filter,
                limit=SCROLL_PAGE,
                offset=offset,
                with_payload=payload,
                with_vectors=False,
            )
            points.extend(batch)
            if offset is None:
                return points

    def indexed_files(self) -> dict[str, str | None]:
        """{tên file: hash đã dùng khi nạp}; None = hash các chunk của file không thống nhất (ghi dở),
        buộc nạp lại."""
        files: dict[str, str | None] = {}
        for point in self._scroll(["source", "file_hash"]):
            source, file_hash = point.payload["source"], point.payload.get("file_hash")
            files[source] = file_hash if files.get(source, file_hash) == file_hash else None
        return files

    def all_chunks(self) -> list[Document]:
        points = self._scroll(True)
        points.sort(key=lambda p: (p.payload["source"], p.payload["ordinal"]))
        return [self._to_document(p.payload) for p in points]

    def similarity_search_with_score(
        self, query: str, k: int, source: str | None = None
    ) -> list[tuple[Document, float]]:
        """Top-k chunk gần nhất (điểm = cosine similarity); `source` giới hạn trong một văn bản."""
        vector = self._embeddings.embed_query(query)
        response = self._client.query_points(
            self._collection,
            query=vector,
            limit=k,
            query_filter=self._source_filter(source) if source else None,
            with_payload=True,
        )
        return [(self._to_document(p.payload), float(p.score)) for p in response.points]
