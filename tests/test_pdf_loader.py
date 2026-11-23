"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>
"""

from pathlib import Path

from langchain_core.documents import Document

from ragbot import pdf_loader as pl
from tests.pdf_factory import build_pdf, grid


def test_table_to_markdown_uses_first_row_as_header_and_cleans_cells():
    rows = [["Vùng", "Mức\nlương"], ["Vùng I", "4.960.000"], [None, ""], ["A|B", "x"]]
    assert pl.table_to_markdown(rows) == (
        "| Vùng | Mức lương |\n| --- | --- |\n| Vùng I | 4.960.000 |\n| A/B | x |"
    )


def test_table_to_markdown_drops_empty_columns_and_rejects_tiny_tables():
    assert pl.table_to_markdown([["a", "", "b"], ["1", "", "2"]]) == "| a | b |\n| --- | --- |\n| 1 | 2 |"
    assert pl.table_to_markdown([["chỉ một hàng", "x"]]) is None
    assert pl.table_to_markdown([["một cột"], ["x"]]) is None
    assert pl.table_to_markdown([]) is None


def test_strip_running_lines_removes_page_numbers_and_repeated_headers():
    # Chữ số được coi như nhau khi so khớp ("CÔNG BÁO số 1" == "CÔNG BÁO số 2"), còn các dòng nội dung
    # khác nhau về chữ thì phải được giữ.
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    pages = [
        [f"CÔNG BÁO số {n}", f"Điều {n}. Tiêu đề {w}", f"nội dung {w}", f"Trang {n}/9"]
        for n, w in enumerate(words, 1)
    ]
    cleaned = pl.strip_running_lines(pages)
    assert cleaned == [[f"Điều {n}. Tiêu đề {w}", f"nội dung {w}"] for n, w in enumerate(words, 1)]


def test_strip_running_lines_keeps_tables_and_ignores_rare_lines():
    table = f"{pl.TABLE_START}\n| a | b |\n{pl.TABLE_END}"
    pages = [["Tiêu đề riêng trang 1", table, "12"], ["nội dung", "3"], ["nội dung", "4"]]
    cleaned = pl.strip_running_lines(pages)
    assert cleaned[0] == ["Tiêu đề riêng trang 1", table]  # chỉ số trang đứng một mình bị bỏ
    assert cleaned[1] == ["nội dung"]


def test_strip_table_markers_keeps_table_text():
    text = f"trước\n{pl.TABLE_START}\n| a | b |\n{pl.TABLE_END}\nsau"
    assert pl.strip_table_markers(text) == "trước\n| a | b |\nsau"


def test_table_block_regex_roundtrip():
    text = f"x\n{pl.TABLE_START}\n| a | b |\n| --- | --- |\n{pl.TABLE_END}\ny"
    assert pl.TABLE_BLOCK_RE.split(text) == ["x\n", "| a | b |\n| --- | --- |", "\ny"]


def test_load_pdf_extracts_table_in_place_and_not_twice(tmp_path: Path):
    cells = [["Vung", "Luong thang"], ["Vung I", "4.960.000"], ["Vung II", "4.410.000"]]
    page1 = [
        ("text", 50, 800, "Dieu 1. Pham vi"),
        ("text", 50, 785, "Noi dung truoc bang."),
        *grid(50, 760, 120, 24, 3, 2, cells),
        ("text", 50, 660, "Noi dung sau bang."),
    ]
    pdf = tmp_path / "van-ban.pdf"
    pdf.write_bytes(build_pdf([page1, [("text", 50, 800, "Trang hai.")]]))

    docs = pl.load_pdf(pdf)

    assert [d.metadata for d in docs] == [
        {"source": "van-ban.pdf", "page": 1},
        {"source": "van-ban.pdf", "page": 2},
    ]
    assert docs[0].page_content == (
        "Dieu 1. Pham vi\nNoi dung truoc bang.\n"
        f"{pl.TABLE_START}\n| Vung | Luong thang |\n| --- | --- |\n| Vung I | 4.960.000 |\n"
        f"| Vung II | 4.410.000 |\n{pl.TABLE_END}\nNoi dung sau bang."
    )
    assert docs[0].page_content.count("4.960.000") == 1  # chữ trong bảng không bị lặp ở phần văn bản


def test_load_pdf_skips_blank_pages(tmp_path: Path):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(build_pdf([[("text", 50, 800, "Trang mot.")], []]))
    docs = pl.load_pdf(pdf)
    assert [d.metadata["page"] for d in docs] == [1]
    assert isinstance(docs[0], Document)


def test_highlighted_paragraphs_are_not_mistaken_for_a_table(tmp_path: Path):
    # Công báo tô nền các đoạn sửa đổi: hình tô màu không viền không được coi là đường kẻ bảng.
    ops = [
        ("fillrect", 50, 740, 480, 17),
        ("fillrect", 50, 722, 480, 17),
        ("text", 54, 744, "Nguoi su dung lao dong phai xay dung"),
        ("text", 54, 726, "thang luong, bang luong va dinh muc."),
    ]
    pdf = tmp_path / "nen.pdf"
    pdf.write_bytes(build_pdf([ops]))

    (doc,) = pl.load_pdf(pdf)

    assert pl.TABLE_START not in doc.page_content
    assert doc.page_content == "Nguoi su dung lao dong phai xay dung\nthang luong, bang luong va dinh muc."


def test_prose_staggered_across_cells_is_not_a_table():
    rows = [["", "Người sử dụng lao động phải xây dựng"], ["mức lao động làm cơ sở", ""], ["lao động.", ""]]
    assert pl.table_to_markdown(rows) is None
    assert pl.table_to_markdown([["Vùng", "Mức"], ["I", "4.960.000"]]) is not None


def test_table_without_header_row_keeps_its_first_row_as_data():
    rows = [
        ["36", "Lệ phí cấp giấy chứng nhận", "Bộ Tài chính"],
        ["37", "Lệ phí cấp chứng chỉ", "Bộ Tài chính"],
    ]
    assert pl.table_to_markdown(rows) == (
        "|  |  |  |\n| --- | --- | --- |\n"
        "| 36 | Lệ phí cấp giấy chứng nhận | Bộ Tài chính |\n| 37 | Lệ phí cấp chứng chỉ | Bộ Tài chính |"
    )
