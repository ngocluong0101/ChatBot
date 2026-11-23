"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>

Đọc PDF và chia thành các chunk.
"""

import logging
import re
from collections.abc import Sequence
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragbot import corpus, legal, pdf_loader
from ragbot.config import Settings
from ragbot.errors import NoDocumentsError

logger = logging.getLogger(__name__)

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]


def list_pdfs(docs_dir: Path) -> list[Path]:
    """Mọi PDF trong thư mục tài liệu, theo thứ tự tên; báo lỗi sớm nếu thư mục sai hoặc rỗng."""
    if not docs_dir.is_dir():
        raise NoDocumentsError(
            f"Thư mục tài liệu không tồn tại: {docs_dir}. Đặt các file PDF vào đó "
            "(xem data/corpus/README.md) hoặc đổi RAG_DOCS_DIR."
        )
    pdfs = sorted(docs_dir.glob("**/*.pdf"))
    if not pdfs:
        raise NoDocumentsError(
            f"Không có file PDF nào trong: {docs_dir}. PDF không nằm trong repo: xem data/corpus/README.md "
            "để biết cách lấy văn bản và ghi nguồn vào data/SOURCES.md."
        )
    return pdfs


def load_documents(settings: Settings, only: Sequence[Path] | None = None) -> list[Document]:
    """Đọc PDF theo từng trang và chuẩn hóa metadata về {source, page}.

    `only`: chỉ đọc các file này (dùng khi cập nhật chỉ mục từng phần); mặc định đọc cả thư mục.
    """
    docs_dir = settings.docs_dir
    paths = list(only) if only is not None else list_pdfs(docs_dir)
    if settings.pdf_loader == "pdfplumber":
        docs = _load_pdfplumber(paths)
    elif only is None:
        docs = _load_unstructured(docs_dir)
    else:
        docs = _load_unstructured_files(paths)
    if not docs:
        raise NoDocumentsError(
            f"Đọc được các file PDF trong {docs_dir} nhưng không trích xuất được văn bản. "
            "Nếu là PDF scan, cần cài Tesseract (kèm gói ngôn ngữ 'vie') để OCR."
        )
    logger.info("Đã đọc %d trang từ %s", len(docs), docs_dir)
    return docs


def _load_pdfplumber(paths: Sequence[Path]) -> list[Document]:
    docs: list[Document] = []
    for pdf in paths:
        docs.extend(pdf_loader.load_pdf(pdf))
    return docs


def _clean_unstructured(raw_docs: list[Document]) -> list[Document]:
    docs: list[Document] = []
    for doc in raw_docs:
        meta = doc.metadata
        source = meta.get("filename") or meta.get("source", "?")
        if not doc.page_content.strip():
            # PDF scan (ảnh) không có text layer sẽ trả về rỗng nếu không có OCR.
            logger.warning("Bỏ qua trang không có nội dung văn bản: %s", source)
            continue
        # Loader trả về nhiều metadata nhiễu (tọa độ pixel, parent_id...). Chỉ giữ thứ cần cho citation.
        doc.metadata = {"source": source, "page": meta.get("page_number")}
        docs.append(doc)
    return docs


def _load_unstructured_files(paths: Sequence[Path]) -> list[Document]:
    raw: list[Document] = []
    for path in paths:
        raw.extend(UnstructuredFileLoader(str(path), mode="paged", languages=["vie"]).load())
    return _clean_unstructured(raw)


def _load_unstructured(docs_dir: Path) -> list[Document]:
    loader = DirectoryLoader(
        path=str(docs_dir),
        glob="**/*.pdf",
        loader_cls=UnstructuredFileLoader,
        # mode="paged" để mỗi trang là một Document có page_number (mode "single" không có).
        loader_kwargs={"mode": "paged", "languages": ["vie"]},
        show_progress=True,
        use_multithreading=True,
    )
    return _clean_unstructured(loader.load())


# Tiêu đề mục "3. Quà tặng"; dấu chấm phải theo sau bởi khoảng trắng để không nhận nhầm "1.000.000".
HEADING_RE = re.compile(r"^(\d{1,2})\.\s+(\S.{1,100})$")
SENTENCE_END = (".", ":", ";", "!", "?")


def _reflow(lines: list[str]) -> str:
    """PDF ngắt dòng cứng giữa câu: nối các dòng chưa kết thúc bằng dấu câu thành một đoạn văn."""
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        current.append(line)
        if line.endswith(SENTENCE_END):
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return "\n".join(paragraphs)


def split_sections(doc: Document, settings: Settings) -> list[Document]:
    """Tách một trang theo mục đánh số; [] nếu không có mục nào (để fallback).

    Mỗi chunk lặp lại tên công ty + tiêu đề tài liệu để embedding/BM25 biết chunk thuộc file nào.
    """
    lines = [ln.strip() for ln in doc.page_content.splitlines() if ln.strip()]
    heading_idx: list[int] = []
    expected: int | None = None
    for i, line in enumerate(lines):
        match = HEADING_RE.match(line)
        if match and (expected is None or int(match.group(1)) == expected):
            heading_idx.append(i)
            expected = int(match.group(1)) + 1
    if not heading_idx:
        return []

    preamble = lines[: heading_idx[0]]
    prefix = " — ".join(preamble[:2])  # dòng 1: công ty, dòng 2: tên tài liệu
    chunks: list[Document] = []
    if preamble:
        chunks.append(
            Document(
                page_content="\n".join(preamble), metadata={**doc.metadata, "section": "Thông tin tài liệu"}
            )
        )

    bounds = [*heading_idx, len(lines)]
    for start, end in zip(bounds, bounds[1:], strict=False):
        heading = lines[start]
        header = f"{prefix}\n{heading}" if prefix else heading
        body = _reflow(lines[start + 1 : end])
        pieces = [body]
        if len(header) + len(body) + 1 > settings.chunk_size:
            body_splitter = RecursiveCharacterTextSplitter(
                chunk_size=max(200, settings.chunk_size - len(header) - 1),
                chunk_overlap=min(settings.chunk_overlap, 100),
            )
            pieces = body_splitter.split_text(body)
        for piece in pieces:
            chunks.append(
                Document(page_content=f"{header}\n{piece}", metadata={**doc.metadata, "section": heading})
            )
    return chunks


def _plain(doc: Document) -> Document:
    """Bỏ dòng đánh dấu bảng (chỉ bộ đọc pdfplumber tạo ra) cho các cách chia không theo cấu trúc."""
    cleaned = pdf_loader.strip_table_markers(doc.page_content)
    return doc if cleaned == doc.page_content else Document(page_content=cleaned, metadata=doc.metadata)


def split_documents(docs: list[Document], settings: Settings) -> list[Document]:
    fixed_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )
    if settings.chunking == "fixed":
        return fixed_splitter.split_documents([_plain(d) for d in docs])

    def by_section(pages: list[Document]) -> list[Document]:
        chunks: list[Document] = []
        for doc in map(_plain, pages):
            # Trang không có mục đánh số (hoặc PDF dạng văn xuôi) rơi về cách cắt theo độ dài.
            chunks.extend(split_sections(doc, settings) or fixed_splitter.split_documents([doc]))
        return chunks

    if settings.chunking == "section":
        return by_section(docs)

    # "legal": gộp các trang của từng file lại vì một Điều có thể kéo dài qua nhiều trang.
    sources = corpus.load_sources(settings.sources_file)
    by_file: dict[str, list[Document]] = {}
    for doc in docs:
        by_file.setdefault(doc.metadata.get("source", "?"), []).append(doc)
    chunks: list[Document] = []
    for filename, pages in by_file.items():
        info = corpus.resolve_info(filename, sources, pages[0].page_content)
        legal_chunks = legal.split_legal(pages, info, settings)
        if legal_chunks is None:
            logger.info("%s không có cấu trúc Điều/Khoản: chia theo mục/độ dài", filename)
            legal_chunks = by_section(pages)
        chunks.extend(legal_chunks)
    return chunks
