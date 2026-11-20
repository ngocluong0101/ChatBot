from langchain_core.documents import Document

from ragbot import stats


def chunk(source: str, text: str, **meta) -> Document:
    return Document(page_content=text, metadata={"source": source, **meta})


def test_compute_stats_counts_documents_pages_chunks_and_flags_problems():
    pages = [Document(page_content="p", metadata={"source": s}) for s in ("a.pdf", "a.pdf", "b.pdf")]
    chunks = [
        chunk("a.pdf", "x" * 100, dieu="1", doc_title="Luật A", type="text"),
        chunk("a.pdf", "x" * 200, dieu="2", type="text"),
        chunk("a.pdf", "x" * 300, dieu="5", type="table"),
        chunk("a.pdf", "x" * 400, dieu="5a", type="text"),
        chunk("b.pdf", "x" * 500),
    ]
    result = stats.compute_stats(pages, chunks, known={"a.pdf": object()})

    assert (result["documents"], result["pages"], result["chunks"]) == (2, 3, 5)
    assert result["chunk_chars"]["mean"] == 300.0 and result["chunk_chars"]["max"] == 500
    assert result["chunk_types"] == {"text": 3, "table": 1, "n/a": 1}
    a, b = result["per_document"]
    assert (a["title"], a["pages"], a["dieu"], a["dieu_range"], a["dieu_missing"], a["tables"]) == (
        "Luật A",
        2,
        3,
        [1, 5],
        [3, 4],
        1,
    )
    assert a["structured"] and a["in_sources"]
    assert not b["structured"] and not b["in_sources"]


def test_format_stats_reports_totals_and_notes():
    pages = [Document(page_content="p", metadata={"source": "b.pdf"})]
    text = stats.format_stats(stats.compute_stats(pages, [chunk("b.pdf", "xyz")], known={}))
    assert "Số văn bản : 1" in text and "Số chunk   : 1" in text
    assert "KHÔNG nhận ra cấu trúc Điều" in text and "chưa có trong SOURCES.md" in text


def test_compute_stats_handles_empty_input():
    result = stats.compute_stats([], [], known={})
    assert result["chunks"] == 0 and result["chunk_chars"]["mean"] is None


def test_files_without_extractable_text_still_show_up_with_a_warning():
    pages = [Document(page_content="p", metadata={"source": "ok.pdf"})]
    chunks = [chunk("ok.pdf", "x", dieu="1")]
    result = stats.compute_stats(pages, chunks, known={}, all_files=["ok.pdf", "scan.pdf"])

    scan = next(d for d in result["per_document"] if d["file"] == "scan.pdf")
    assert scan["no_text"] and scan["pages"] == 0 and scan["chunks"] == 0
    assert "KHÔNG trích được chữ" in stats.format_stats(result)


def test_document_not_ending_with_a_closing_article_is_flagged_as_possibly_truncated():
    def result(last_title: str):
        pages = [Document(page_content="p", metadata={"source": "a.pdf"})]
        chunks = [
            chunk("a.pdf", "x", dieu="1", dieu_title="Phạm vi"),
            chunk("a.pdf", "x", dieu="2", dieu_title=last_title),
        ]
        return stats.format_stats(stats.compute_stats(pages, chunks, known={"a.pdf": 1}))

    assert "có thể bị cắt dở" in result("Thành lập Hội đồng trọng tài lao động")
    assert "có thể bị cắt dở" not in result("Hiệu lực thi hành")
    assert "có thể bị cắt dở" not in result("Quy định chuyển tiếp")
