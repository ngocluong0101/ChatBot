"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>
"""

from ragbot import corpus

TABLE = """
# Nguồn tài liệu

Mô tả bất kỳ không phải bảng.

| File | Tên văn bản | Số hiệu | Loại | Ngày ban hành | Ngày hiệu lực | Link nguồn |
|---|---|---|---|---|---|---|
| 45-2019-QH14.pdf | Bộ luật Lao động 2019 | 45/2019/QH14 | gốc | 20/11/2019 | 01/01/2021 | https://x.org/45 |
| 145-2020-ND-CP.pdf | Nghị định 145/2020/NĐ-CP | 145/2020/NĐ-CP | hướng dẫn | 14/12/2020 | 01/02/2021 | |
|  | Hàng thiếu file | | | | | |
| thieu-ten.pdf | | | | | | |

<!-- | vi-du.pdf | Dòng nằm trong comment vẫn bắt đầu bằng gạch đứng | | | | | | -->
"""


def test_parse_sources_reads_rows_and_skips_incomplete_ones():
    result = corpus.parse_sources(TABLE)

    assert set(result) == {"45-2019-QH14.pdf", "145-2020-ND-CP.pdf"}
    labor = result["45-2019-QH14.pdf"]
    assert labor.title == "Bộ luật Lao động 2019"
    assert labor.so_hieu == "45/2019/QH14"
    assert labor.effective == "01/01/2021"
    assert labor.url == "https://x.org/45"
    assert result["145-2020-ND-CP.pdf"].url is None  # ô trống -> None


def test_parse_sources_matches_columns_by_header_not_position():
    text = "| Tên văn bản | Link | File |\n|---|---|---|\n| Luật A | http://a | a.pdf |\n"
    info = corpus.parse_sources(text)["a.pdf"]
    assert (info.title, info.url) == ("Luật A", "http://a")


def test_parse_sources_without_required_columns_returns_empty():
    assert corpus.parse_sources("| A | B |\n|---|---|\n| 1 | 2 |\n") == {}


def test_load_sources_missing_file_is_empty(tmp_path):
    assert corpus.load_sources(tmp_path / "khong-co.md") == {}


def test_resolve_info_prefers_table_then_falls_back_to_filename():
    known = corpus.parse_sources(TABLE)
    assert corpus.resolve_info("45-2019-QH14.pdf", known).title == "Bộ luật Lao động 2019"

    guessed = corpus.resolve_info("luat_an_toan-thong-tin.pdf", known, "Luật số: 86/2015/QH13\nAn toàn")
    assert guessed.title == "luat an toan thong tin"
    assert guessed.so_hieu == "86/2015/QH13"


def test_guess_so_hieu_returns_none_when_absent():
    assert corpus.guess_so_hieu("Không có số hiệu ở đây") is None
