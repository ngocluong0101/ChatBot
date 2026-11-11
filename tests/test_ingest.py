from langchain_core.documents import Document

from ragbot import ingest
from ragbot.config import Settings


def page(*lines: str) -> Document:
    """Giống đầu ra của unstructured: mỗi dòng hiển thị cách nhau một dòng trống."""
    return Document(page_content="\n\n".join(lines), metadata={"source": "x.pdf", "page": 1})


PREAMBLE = ("CÔNG TY ABC", "QUY ĐỊNH NGHỈ PHÉP", "Mã tài liệu", "HR-001")
HEADER = "CÔNG TY ABC — QUY ĐỊNH NGHỈ PHÉP"


def settings(**kwargs) -> Settings:
    return Settings(chunking="section", **kwargs)


def test_splits_into_preamble_and_one_chunk_per_section():
    doc = page(*PREAMBLE, "1. Nghỉ phép năm", "Có 12 ngày phép.", "2. Nghỉ ốm", "Báo trước 09:00.")
    chunks = ingest.split_sections(doc, settings())

    sections = [c.metadata["section"] for c in chunks]
    assert sections == ["Thông tin tài liệu", "1. Nghỉ phép năm", "2. Nghỉ ốm"]
    assert chunks[0].page_content.startswith("CÔNG TY ABC\nQUY ĐỊNH NGHỈ PHÉP")
    # tiêu đề tài liệu được lặp lại ở mọi chunk mục để biết chunk thuộc file nào
    assert chunks[1].page_content == f"{HEADER}\n1. Nghỉ phép năm\nCó 12 ngày phép."
    assert all(c.metadata["source"] == "x.pdf" and c.metadata["page"] == 1 for c in chunks)


def test_reflows_hard_wrapped_lines_into_sentences():
    doc = page(*PREAMBLE, "1. Mục", "Nhân viên được đăng ký tối đa", "2 ngày mỗi tuần.", "Câu thứ hai.")
    body = ingest.split_sections(doc, settings())[1].page_content.splitlines()[2:]
    assert body == ["Nhân viên được đăng ký tối đa 2 ngày mỗi tuần.", "Câu thứ hai."]


def test_numbers_inside_text_are_not_headings():
    doc = page(
        *PREAMBLE,
        "1. Quà tặng",
        "Trên 1.000.000 VNĐ phải khai báo.",
        "7. Dòng lạc số thứ tự.",
        "2. Mục hai",
        "Nội dung.",
    )
    sections = [c.metadata["section"] for c in ingest.split_sections(doc, settings())]
    assert sections == ["Thông tin tài liệu", "1. Quà tặng", "2. Mục hai"]


def test_returns_empty_when_no_numbered_headings():
    assert ingest.split_sections(page("Đoạn văn xuôi", "không có mục đánh số."), settings()) == []


def test_long_section_is_split_and_every_piece_keeps_header():
    long_body = " ".join(f"Câu số {i} nói về quy định." for i in range(40))
    doc = page(*PREAMBLE, "1. Mục dài", long_body)
    chunks = ingest.split_sections(doc, settings(chunk_size=300, chunk_overlap=50))
    section_chunks = [c for c in chunks if c.metadata["section"] == "1. Mục dài"]

    assert len(section_chunks) > 1
    assert all(c.page_content.startswith(f"{HEADER}\n1. Mục dài\n") for c in section_chunks)
    assert all(len(c.page_content) <= 300 + 60 for c in section_chunks)


def test_split_documents_falls_back_to_fixed_for_pages_without_sections():
    plain = page("Đoạn văn xuôi không có mục.")
    structured = page(*PREAMBLE, "1. Mục", "Nội dung.")
    chunks = ingest.split_documents([plain, structured], settings())

    assert "section" not in chunks[0].metadata  # chunk fixed: không có metadata section
    assert any(c.metadata.get("section") == "1. Mục" for c in chunks)


def test_fixed_strategy_never_adds_sections():
    doc = page(*PREAMBLE, "1. Mục", "Nội dung.")
    chunks = ingest.split_documents([doc], Settings(chunking="fixed"))
    assert chunks and all("section" not in c.metadata for c in chunks)
