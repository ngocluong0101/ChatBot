from langchain_core.documents import Document

from ragbot import legal
from ragbot.config import Settings
from ragbot.corpus import DocInfo
from ragbot.pdf_loader import TABLE_END, TABLE_START

INFO = DocInfo(file="blld.pdf", title="Bộ luật Lao động 2019", so_hieu="45/2019/QH14")
TITLE_LINE = "Bộ luật Lao động 2019 (45/2019/QH14)"


def pages(*texts: str) -> list[Document]:
    return [
        Document(page_content=t, metadata={"source": "blld.pdf", "page": n}) for n, t in enumerate(texts, 1)
    ]


def split(*texts: str, **settings) -> list[Document]:
    chunks = legal.split_legal(pages(*texts), INFO, Settings(chunking="legal", **settings))
    assert chunks is not None
    return chunks


def by_dieu(chunks: list[Document], number: str) -> list[Document]:
    return [c for c in chunks if c.metadata.get("dieu") == number]


def table(*rows: str) -> str:
    return f"{TABLE_START}\n" + "\n".join(rows) + f"\n{TABLE_END}"


BASIC = """QUỐC HỘI
Luật số: 45/2019/QH14
CHƯƠNG I
NHỮNG QUY ĐỊNH CHUNG
Điều 1. Phạm vi điều chỉnh
Bộ luật này quy định tiêu chuẩn lao động.
Điều 2. Đối tượng áp dụng
1. Người lao động, người học nghề.
2. Người sử dụng lao động.
Điều 3. Giải thích từ ngữ
Nội dung điều ba.
CHƯƠNG II
THỜI GIỜ LÀM VIỆC
Mục 1. Thời giờ làm việc
Điều 105. Thời giờ làm việc bình thường
Không quá 08 giờ trong 01 ngày."""


def test_one_chunk_per_dieu_with_metadata_and_context_header():
    chunks = split(BASIC)

    assert [c.metadata.get("dieu") for c in chunks] == [None, "1", "2", "3", "105"]
    d1 = by_dieu(chunks, "1")[0]
    assert d1.page_content == (
        f"{TITLE_LINE}\nChương I. NHỮNG QUY ĐỊNH CHUNG\n"
        "Điều 1. Phạm vi điều chỉnh\nBộ luật này quy định tiêu chuẩn lao động."
    )
    assert d1.metadata == {
        "source": "blld.pdf",
        "doc_title": "Bộ luật Lao động 2019",
        "so_hieu": "45/2019/QH14",
        "chuong": "Chương I. NHỮNG QUY ĐỊNH CHUNG",
        "dieu": "1",
        "dieu_title": "Phạm vi điều chỉnh",
        "page": 1,
        "type": "text",
        "section": "Điều 1. Phạm vi điều chỉnh",
    }


def test_chuong_and_muc_context_is_carried_into_later_chunks():
    d105 = by_dieu(split(BASIC), "105")[0]
    assert d105.metadata["chuong"] == "Chương II. THỜI GIỜ LÀM VIỆC"
    assert d105.metadata["muc"] == "Mục 1. Thời giờ làm việc"
    assert "Chương II. THỜI GIỜ LÀM VIỆC; Mục 1. Thời giờ làm việc\nĐiều 105." in d105.page_content


def test_preamble_becomes_its_own_chunk_without_dieu():
    first = split(BASIC)[0]
    assert first.metadata["type"] == "preamble"
    assert "dieu" not in first.metadata
    assert "Luật số: 45/2019/QH14" in first.page_content


def test_short_dieu_keeps_whole_text_and_khoan_range_label():
    d2 = by_dieu(split(BASIC), "2")[0]
    assert d2.metadata["khoan"] == "1–2"
    assert d2.page_content.endswith("1. Người lao động, người học nghề.\n2. Người sử dụng lao động.")


def test_hard_wrapped_lines_are_joined_into_sentences():
    text = (
        "Điều 1. A\nNội dung một.\nĐiều 2. B\nNội dung hai.\nĐiều 3. Nghỉ phép\n"
        "1. Người lao động làm việc đủ 12 tháng cho\nngười sử dụng lao động thì được nghỉ\nhằng năm.\n"
        "2. Số ngày nghỉ tăng thêm."
    )
    d3 = by_dieu(split(text), "3")[0]
    assert d3.page_content.splitlines()[-2:] == [
        "1. Người lao động làm việc đủ 12 tháng cho người sử dụng lao động thì được nghỉ hằng năm.",
        "2. Số ngày nghỉ tăng thêm.",
    ]


def test_dieu_spanning_pages_is_one_chunk_citing_its_first_page():
    p1 = "Điều 1. A\nNội dung một.\nĐiều 2. B\nNội dung hai.\nĐiều 3. Dài\nCâu đầu tiên của điều ba nằm ở"
    p2 = "cuối trang một và kết thúc ở trang hai.\nĐiều 4. D\nNội dung bốn."
    d3 = by_dieu(split(p1, p2), "3")
    assert len(d3) == 1
    assert d3[0].metadata["page"] == 1
    assert d3[0].page_content.endswith("nằm ở cuối trang một và kết thúc ở trang hai.")
    assert by_dieu(split(p1, p2), "4")[0].metadata["page"] == 2


def test_long_dieu_is_split_by_khoan_and_every_chunk_keeps_context():
    khoans = [f"{i}. " + " ".join(f"Câu số {i}-{j} nói về quy định." for j in range(8)) for i in range(1, 5)]
    text = "Điều 1. A\nx.\nĐiều 2. B\ny.\nĐiều 3. Dài\nMở đầu điều:\n" + "\n".join(khoans)
    d3 = by_dieu(split(text, chunk_size=400, chunk_overlap=0), "3")

    assert len(d3) >= 4
    assert all(c.page_content.startswith(f"{TITLE_LINE}\nĐiều 3. Dài\n") for c in d3)
    assert all(len(c.page_content) <= 400 for c in d3)
    labels = [c.metadata.get("khoan") for c in d3]
    assert "1" in labels and "4" in labels
    assert next(c for c in d3 if c.metadata.get("khoan") == "2").page_content.count("Câu số 2-") == 8


def test_khoan_longer_than_a_chunk_is_split_by_diem_then_length():
    diems = [f"{chr(96 + i)}) " + "từ " * 40 + f"điểm {i}." for i in range(1, 6)]
    text = "Điều 1. A\nx.\nĐiều 2. B\ny.\nĐiều 3. Dài\n1. Các trường hợp sau đây:\n" + "\n".join(diems)
    d3 = by_dieu(split(text, chunk_size=500, chunk_overlap=0), "3")

    assert len(d3) > 1
    assert all(c.metadata["khoan"] == "1" for c in d3)
    assert all(len(c.page_content) <= 500 for c in d3)
    joined = "\n".join(c.page_content for c in d3)
    assert all(f"điểm {i}." in joined for i in range(1, 6))


def test_reference_to_earlier_dieu_and_quoted_dieu_are_not_headings():
    text = (
        "Điều 3. A\nx.\nĐiều 4. B\ny.\nĐiều 5. C\nNội dung điều năm.\n"
        "Điều 2. Quy định này được viện dẫn ở dòng đầu.\n"
        "“Điều 9. Trích nguyên văn trong ngoặc kép\n"
        "Điều 6. D\nz."
    )
    chunks = split(text)
    assert [c.metadata.get("dieu") for c in chunks if "dieu" in c.metadata] == ["3", "4", "5", "6"]
    d5 = by_dieu(chunks, "5")[0].page_content
    assert "Điều 2. Quy định này được viện dẫn" in d5 and "“Điều 9." in d5


def test_numbers_inside_text_and_stray_numbering_are_not_khoan():
    text = (
        "Điều 1. A\nx.\nĐiều 2. B\ny.\nĐiều 3. Quà\n"
        "1. Trên 1.000.000 VNĐ phải khai báo.\n7. Dòng lạc số thứ tự.\n2. Khoản hai."
    )
    d3 = by_dieu(split(text), "3")[0]
    assert d3.metadata["khoan"] == "1–2"
    assert "7. Dòng lạc số thứ tự." in d3.page_content


def test_dieu_with_letter_suffix_follows_its_base_number():
    text = "Điều 9. A\nx.\nĐiều 10. B\ny.\nĐiều 10a. Bổ sung\nz.\nĐiều 11. C\nw."
    assert [c.metadata.get("dieu") for c in split(text) if "dieu" in c.metadata] == ["9", "10", "10a", "11"]


def test_wrapped_dieu_title_is_joined():
    text = (
        "Điều 1. A\nx.\nĐiều 2. B\ny.\nĐiều 3. Các hành vi bị nghiêm cấm trong\nlĩnh vực lao động\nNội dung."
    )
    d3 = by_dieu(split(text), "3")[0]
    assert d3.metadata["dieu_title"] == "Các hành vi bị nghiêm cấm trong lĩnh vực lao động"
    assert d3.page_content.endswith("Nội dung.")


def test_table_of_contents_is_discarded_and_numbering_restarts():
    toc = "MỤC LỤC\nĐiều 1. Phạm vi\nĐiều 2. Đối tượng\nĐiều 3. Giải thích\n"
    body = (
        "Điều 1. Phạm vi\nNội dung thật một.\nĐiều 2. Đối tượng\nNội dung thật hai.\n"
        "Điều 3. Giải thích\nNội dung ba."
    )
    chunks = split(toc + body)
    assert [c.metadata.get("dieu") for c in chunks if "dieu" in c.metadata] == ["1", "2", "3"]
    assert "Nội dung thật một." in by_dieu(chunks, "1")[0].page_content


def test_body_line_starting_with_this_code_is_not_mistaken_for_closing():
    text = "Điều 1. A\nx.\nĐiều 2. B\ny.\nĐiều 3. C\nBộ luật này được áp dụng đối với mọi người.\nCâu sau."
    assert by_dieu(split(text), "3")[0].page_content.endswith("áp dụng đối với mọi người.\nCâu sau.")


def test_closing_signature_is_dropped_from_the_last_dieu():
    text = (
        "Điều 1. A\nx.\nĐiều 2. B\ny.\nĐiều 3. Hiệu lực thi hành\n"
        "Bộ luật này có hiệu lực từ ngày 01/01/2021.\n"
        "Bộ luật này được Quốc hội nước Cộng hòa xã hội chủ nghĩa Việt Nam thông qua ngày 20/11/2019./.\n"
        "CHỦ TỊCH QUỐC HỘI\nNguyễn Thị Kim Ngân"
    )
    d3 = by_dieu(split(text), "3")[0].page_content
    assert d3.endswith("có hiệu lực từ ngày 01/01/2021.")
    assert "Kim Ngân" not in d3 and "thông qua" not in d3


def test_appendix_after_signature_is_kept_outside_any_chuong():
    text = (
        "CHƯƠNG I\nQUY ĐỊNH CHUNG\nĐiều 1. A\nx.\nĐiều 2. B\ny.\nĐiều 3. C\nz.\nNơi nhận:\nVăn phòng\n"
        "PHỤ LỤC I\nDANH MỤC CÔNG VIỆC\nCông việc số một.\n"
    )
    appendix = [c for c in split(text) if "phu_luc" in c.metadata]
    assert len(appendix) == 1
    meta = appendix[0].metadata
    assert meta["phu_luc"] == "I" and meta["section"] == "Phụ lục I. DANH MỤC CÔNG VIỆC"
    assert "chuong" not in meta and "dieu" not in meta
    assert "Văn phòng" not in appendix[0].page_content
    assert by_dieu(split(text), "3")[0].page_content.endswith("z.")


def test_table_becomes_its_own_chunk_with_context_and_surrounding_text_kept():
    md = ["| Vùng | Mức lương |", "| --- | --- |", "| Vùng I | 4.960.000 |"]
    text = (
        "Điều 1. A\nx.\nĐiều 2. B\ny.\nĐiều 3. Lương tối thiểu\n"
        f"1. Mức lương như sau:\n{table(*md)}\n2. Áp dụng từ ngày 01/7/2024."
    )
    d3 = by_dieu(split(text), "3")
    kinds = [(c.metadata["type"], c.metadata.get("khoan")) for c in d3]

    assert ("table", "1") in kinds
    tbl = next(c for c in d3 if c.metadata["type"] == "table")
    assert tbl.page_content == f"{TITLE_LINE}\nĐiều 3. Lương tối thiểu\n" + "\n".join(md)
    assert TABLE_START not in "".join(c.page_content for c in d3)
    texts = [c.page_content for c in d3 if c.metadata["type"] == "text"]
    assert any("Mức lương như sau:" in t for t in texts) and any("Áp dụng từ ngày" in t for t in texts)


def test_large_table_is_split_by_rows_and_repeats_its_header():
    rows = [f"| Tỉnh {i} | {i * 1000} |" for i in range(40)]
    md = ["| Tỉnh | Mức |", "| --- | --- |", *rows]
    text = f"Điều 1. A\nx.\nĐiều 2. B\ny.\nĐiều 3. Bảng\n{table(*md)}"
    tables = [c for c in split(text, chunk_size=400, chunk_overlap=0) if c.metadata["type"] == "table"]

    assert len(tables) > 1
    assert all("| Tỉnh | Mức |\n| --- | --- |\n" in c.page_content for c in tables)
    assert all(len(c.page_content) <= 400 for c in tables)
    body = "\n".join(c.page_content for c in tables)
    assert all(row in body for row in rows)


def test_prose_starting_with_chuong_or_muc_is_not_a_heading():
    text = (
        "Điều 1. A\nChương trình đào tạo do người sử dụng\nMục 3 của Điều này không áp dụng.\n"
        "Điều 2. B\ny.\nĐiều 3. C\nz."
    )
    chunks = split(text)
    assert not any("chuong" in c.metadata or "muc" in c.metadata for c in chunks)
    assert "Chương trình đào tạo" in by_dieu(chunks, "1")[0].page_content


def test_returns_none_without_enough_dieu():
    plain = pages("Đoạn văn xuôi không có Điều.\nThêm một dòng.")
    assert legal.split_legal(plain, INFO, Settings(chunking="legal")) is None
    two = pages("Điều 1. A\nx.\nĐiều 2. B\ny.")
    assert legal.split_legal(two, INFO, Settings(chunking="legal")) is None
    assert legal.split_legal([], INFO, Settings(chunking="legal")) is None


def test_chunks_without_so_hieu_or_when_title_already_has_it():
    info = DocInfo(file="a.pdf", title="Nghị định 145/2020/NĐ-CP", so_hieu="145/2020/NĐ-CP")
    docs = [
        Document(
            page_content="Điều 1. A\nx.\nĐiều 2. B\ny.\nĐiều 3. C\nz.",
            metadata={"source": "a.pdf", "page": 1},
        )
    ]
    chunks = legal.split_legal(docs, info, Settings(chunking="legal"))
    assert chunks is not None
    assert chunks[0].page_content.splitlines()[0] == "Nghị định 145/2020/NĐ-CP"  # không lặp số hiệu

    bare = DocInfo(file="a.pdf", title="Văn bản A")
    chunks = legal.split_legal(docs, bare, Settings(chunking="legal"))
    assert chunks is not None and "so_hieu" not in chunks[0].metadata


def test_multiline_uppercase_chuong_and_muc_titles_are_joined():
    text = (
        "CHƯƠNG V\nĐỐI THOẠI TẠI NƠI LÀM VIỆC, THƯƠNG LƯỢNG TẬP THỂ,\nTHỎA ƯỚC LAO ĐỘNG TẬP THỂ\n"
        "Điều 1. A\nx.\n"
        "Mục 3. THỜI GIỜ LÀM VIỆC, NGHỈ NGƠI\nĐỐI VỚI NGƯỜI LÀM CÔNG VIỆC ĐẶC BIỆT\n"
        "Điều 2. B\ny.\nĐiều 3. C\nz."
    )
    chunks = split(text)
    d1, d2 = by_dieu(chunks, "1")[0], by_dieu(chunks, "2")[0]
    assert d1.metadata["chuong"] == (
        "Chương V. ĐỐI THOẠI TẠI NƠI LÀM VIỆC, THƯƠNG LƯỢNG TẬP THỂ, THỎA ƯỚC LAO ĐỘNG TẬP THỂ"
    )
    assert d2.metadata["muc"] == "Mục 3. THỜI GIỜ LÀM VIỆC, NGHỈ NGƠI ĐỐI VỚI NGƯỜI LÀM CÔNG VIỆC ĐẶC BIỆT"
    # tiêu đề không còn sót lại thành chunk "mở đầu" riêng
    assert not [c for c in chunks if c.metadata["type"] == "preamble"]


def test_long_dieu_title_wrapping_onto_lines_that_start_with_digits_is_joined():
    text = (
        "Điều 1. A\nx.\nĐiều 2. B\ny.\n"
        "Điều 3. Sửa đổi, bổ sung Luật Phí số 97/2015/QH13 đã được sửa đổi theo Luật số\n"
        "09/2017/QH14, Luật số 23/2018/QH14 và Luật\nsố 35/2024/QH15\n"
        "Bổ sung số thứ tự 36 vào sau số thứ tự 35 như sau:"
    )
    d3 = by_dieu(split(text), "3")[0]
    assert d3.metadata["dieu_title"].endswith(
        "Luật số 09/2017/QH14, Luật số 23/2018/QH14 và Luật số 35/2024/QH15"
    )
    assert d3.page_content.endswith("Bổ sung số thứ tự 36 vào sau số thứ tự 35 như sau:")


def test_continuation_marker_of_a_multi_issue_gazette_is_not_body_text():
    text = (
        "Điều 1. A\nx.\nĐiều 2. B\ny.\nĐiều 3. C\nNội dung cuối của phần này.\n"
        "(Xem tiếp theo Công báo số 1205 + 1206)"
    )
    assert by_dieu(split(text), "3")[0].page_content.endswith("Nội dung cuối của phần này.")
