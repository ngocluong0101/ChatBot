"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>
"""

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


# --- chunking="legal" -------------------------------------------------------------------------------

LEGAL_PAGE = (
    "Điều 1. Phạm vi\nBộ luật này quy định tiêu chuẩn.\nĐiều 2. Đối tượng\nNgười lao động.\n"
    "Điều 3. Giải thích\nCác từ ngữ sau đây.\n"
)


def legal_settings(tmp_path, rows: str = "", **kwargs) -> Settings:
    sources = tmp_path / "SOURCES.md"
    sources.write_text("| File | Tên văn bản | Số hiệu |\n|---|---|---|\n" + rows, encoding="utf-8")
    return Settings(chunking="legal", sources_file=sources, **kwargs)


def test_legal_chunking_uses_sources_table_and_falls_back_for_unstructured_files(tmp_path):
    s = legal_settings(tmp_path, "| luat.pdf | Luật Mẫu 2020 | 01/2020/QH14 |\n")
    docs = [
        Document(page_content=LEGAL_PAGE, metadata={"source": "luat.pdf", "page": 1}),
        Document(page_content="1. Mục\nNội dung.", metadata={"source": "noi-quy.pdf", "page": 1}),
        Document(page_content="Văn xuôi không có mục.", metadata={"source": "van-xuoi.pdf", "page": 1}),
    ]
    chunks = ingest.split_documents(docs, s)

    legal_chunks = [c for c in chunks if c.metadata["source"] == "luat.pdf"]
    assert [c.metadata["dieu"] for c in legal_chunks] == ["1", "2", "3"]
    assert legal_chunks[0].metadata["doc_title"] == "Luật Mẫu 2020"
    assert legal_chunks[0].page_content.startswith("Luật Mẫu 2020 (01/2020/QH14)\nĐiều 1. Phạm vi")
    # file không có cấu trúc Điều: rơi về chia theo mục rồi theo độ dài như trước
    assert any(c.metadata.get("section") == "1. Mục" for c in chunks if c.metadata["source"] == "noi-quy.pdf")
    assert any(c.metadata["source"] == "van-xuoi.pdf" for c in chunks)


def test_legal_chunking_merges_pages_of_one_file(tmp_path):
    s = legal_settings(tmp_path)
    docs = [
        Document(
            page_content="Điều 1. A\nx.\nĐiều 2. B\ny.\nĐiều 3. Dài\nCâu bắt đầu ở",
            metadata={"source": "a.pdf", "page": 1},
        ),
        Document(page_content="trang hai.\nĐiều 4. D\nz.", metadata={"source": "a.pdf", "page": 2}),
    ]
    d3 = next(c for c in ingest.split_documents(docs, s) if c.metadata.get("dieu") == "3")
    assert d3.page_content.endswith("Câu bắt đầu ở trang hai.") and d3.metadata["page"] == 1
    assert d3.metadata["doc_title"] == "a"  # không có trong SOURCES.md: lấy từ tên file


def test_fixed_and_section_chunking_never_see_table_markers():
    from ragbot.pdf_loader import TABLE_END, TABLE_START

    text = f"1. Mục\nTrước bảng.\n{TABLE_START}\n| a | b |\n| --- | --- |\n| 1 | 2 |\n{TABLE_END}\nSau bảng."
    doc = Document(page_content=text, metadata={"source": "x.pdf", "page": 1})
    for chunking in ("fixed", "section"):
        chunks = ingest.split_documents([doc], Settings(chunking=chunking))
        joined = "\n".join(c.page_content for c in chunks)
        assert TABLE_START not in joined and TABLE_END not in joined
        assert "| a | b |" in joined
