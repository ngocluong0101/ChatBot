"""Đọc PDF và chia thành các chunk."""

import logging
import re

from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

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


def load_documents(settings: Settings) -> list[Document]:
    """Đọc PDF theo từng trang và chuẩn hóa metadata về {source, page}."""
    papers_dir = settings.papers_dir
    if not papers_dir.is_dir():
        raise NoDocumentsError(f"Thư mục tài liệu không tồn tại: {papers_dir}")
    if not any(papers_dir.glob("**/*.pdf")):
        raise NoDocumentsError(f"Không có file PDF nào trong: {papers_dir}")

    loader = DirectoryLoader(
        path=str(papers_dir),
        glob="**/*.pdf",
        loader_cls=UnstructuredFileLoader,
        # mode="paged" để mỗi trang là một Document có page_number (mode "single" không có).
        loader_kwargs={"mode": "paged", "languages": ["vie"]},
        show_progress=True,
        use_multithreading=True,
    )
    raw_docs = loader.load()

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

    if not docs:
        raise NoDocumentsError(
            f"Đọc được các file PDF trong {papers_dir} nhưng không trích xuất được văn bản. "
            "Nếu là PDF scan, cần cài Tesseract (kèm gói ngôn ngữ 'vie') để OCR."
        )
    logger.info("Đã đọc %d trang từ %s", len(docs), papers_dir)
    return docs


# Tiêu đề mục dạng "3. Quà tặng và tiếp khách". Dấu chấm phải theo sau bởi khoảng trắng nên
# "1.000.000 VNĐ" không bị nhận nhầm.
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
    """Tách một trang theo mục đánh số. Trả về [] nếu trang không có mục nào (để fallback).

    Mỗi chunk gồm: tên công ty + tiêu đề tài liệu, tiêu đề mục, nội dung mục. Phần tiêu đề được lặp
    lại ở mọi chunk để embedding/BM25 biết chunk thuộc tài liệu nào (các tài liệu có mục trùng tên,
    ví dụ "Nghỉ phép", nằm ở nhiều file).
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


def split_documents(docs: list[Document], settings: Settings) -> list[Document]:
    fixed_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )
    if settings.chunking == "fixed":
        return fixed_splitter.split_documents(docs)

    chunks: list[Document] = []
    for doc in docs:
        # Trang không có mục đánh số (hoặc PDF dạng văn xuôi) rơi về cách cắt theo độ dài.
        chunks.extend(split_sections(doc, settings) or fixed_splitter.split_documents([doc]))
    return chunks
