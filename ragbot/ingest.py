"""Đọc PDF và chia thành các chunk."""

import logging

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


def split_documents(docs: list[Document], settings: Settings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )
    return splitter.split_documents(docs)
