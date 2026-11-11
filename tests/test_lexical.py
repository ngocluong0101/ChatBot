from langchain_core.documents import Document

from ragbot.lexical import BM25Index, strip_accents, tokenize


def docs(*texts: str) -> list[Document]:
    return [Document(page_content=t) for t in texts]


def test_strip_accents_handles_d_with_stroke():
    assert strip_accents("Đặt mật khẩu đường") == "Dat mat khau duong"


def test_tokenize_has_unigrams_and_bigrams():
    tokens = tokenize("Mật khẩu 12 ký tự")
    assert {"mat", "khau", "12", "ky", "tu"} <= set(tokens)
    assert "mat khau" in tokens and "ky tu" in tokens


def test_bm25_matches_unaccented_query_to_accented_text():
    index = BM25Index(
        docs(
            "Mật khẩu phải có ít nhất 12 ký tự",
            "Nhân viên chính thức có 12 ngày phép năm",
            "Khung giờ cốt lõi 09:30-11:30",
        )
    )
    top = index.top("mat khau toi thieu bao nhieu ky tu", k=3)
    assert top[0][0] == 0


def test_bm25_drops_zero_score_documents_and_unknown_terms():
    index = BM25Index(docs("Mật khẩu phải có 12 ký tự", "Nghỉ phép năm"))
    assert index.top("thủ đô nước Pháp", k=5) == []
    assert [i for i, _ in index.top("nghỉ phép", k=5)] == [1]


def test_bm25_handles_empty_corpus():
    assert BM25Index([]).top("bất kỳ", k=3) == []
