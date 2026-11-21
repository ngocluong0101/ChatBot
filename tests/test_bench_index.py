from pathlib import Path

from langchain_core.embeddings import DeterministicFakeEmbedding

from evals import bench_index as bench
from ragbot.config import Settings
from tests.pdf_factory import build_pdf


def write_pdf(path: Path, n: int) -> None:
    lines = [f"Dieu {i} quy dinh ve van ban {path.stem} so {i} va cac noi dung lien quan." for i in range(n)]
    path.write_bytes(build_pdf([[("text", 50, 800 - 14 * i, line) for i, line in enumerate(lines)]]))


def make_settings(tmp_path: Path) -> Settings:
    docs = tmp_path / "docs"
    docs.mkdir()
    write_pdf(docs / "nho.pdf", 3)
    write_pdf(docs / "vua.pdf", 8)
    write_pdf(docs / "lon.pdf", 20)
    return Settings(
        docs_dir=docs,
        index_dir=tmp_path / "index-main",
        sources_file=tmp_path / "SOURCES.md",
        chunking="fixed",
        chunk_size=200,
        chunk_overlap=20,
        pdf_loader="pdfplumber",
    )


def test_pick_files_returns_smallest_and_largest_pdf(tmp_path: Path):
    settings = make_settings(tmp_path)
    small, large = bench.pick_files(settings.docs_dir)
    assert (small.name, large.name) == ("nho.pdf", "lon.pdf")


def test_touch_pdf_changes_the_bytes_but_the_file_still_loads(tmp_path: Path):
    from ragbot import pdf_loader

    path = tmp_path / "a.pdf"
    write_pdf(path, 2)
    before = path.read_bytes()
    bench.touch_pdf(path)
    assert path.read_bytes() == before + b"\n" and pdf_loader.load_pdf(path)


def test_benchmark_updates_only_touched_files_and_never_modifies_the_originals(tmp_path: Path):
    settings = make_settings(tmp_path)
    originals = {p.name: p.read_bytes() for p in settings.docs_dir.glob("*.pdf")}
    result = bench.run_benchmark(settings, DeterministicFakeEmbedding(size=16), tmp_path / "work")

    q = result["qdrant"]
    assert sorted(q["first_build"]["added"]) == ["lon.pdf", "nho.pdf", "vua.pdf"]
    assert q["restart_unchanged"]["chunks_embedded"] == 0 and q["restart_unchanged"]["unchanged"] == 3
    assert q["update_small_file"]["updated"] == ["nho.pdf"] and q["update_large_file"]["updated"] == [
        "lon.pdf"
    ]
    assert (
        0
        < q["update_small_file"]["chunks_embedded"]
        < q["update_large_file"]["chunks_embedded"]
        < q["first_build"]["chunks_embedded"]
    )
    assert q["remove_file"]["removed"] == ["nho.pdf"]
    assert result["faiss"]["chunks"] == q["first_build"]["chunks_total"]
    assert set(result["speedup_vs_faiss_rebuild"]) == {"update_small_file", "update_large_file"}
    assert {
        p.name: p.read_bytes() for p in settings.docs_dir.glob("*.pdf")
    } == originals  # file gốc nguyên vẹn


def test_benchmark_can_skip_the_slow_faiss_step(tmp_path: Path):
    result = bench.run_benchmark(
        make_settings(tmp_path), DeterministicFakeEmbedding(size=16), tmp_path / "w", include_faiss=False
    )
    assert "faiss" not in result and "qdrant" in result


def test_timing_table_comes_from_the_report_and_marks_missing_measurements():
    result = {
        "qdrant": {
            "first_build": {"seconds": 100.0, "chunks_total": 50, "chunks_embedded": 50},
            "restart_unchanged": {"seconds": 0.5, "chunks_total": 50, "chunks_embedded": 0},
            "update_small_file": {
                "seconds": 2.0,
                "chunks_total": 50,
                "chunks_embedded": 5,
                "file": "nho.pdf",
            },
            "update_large_file": {
                "seconds": 20.0,
                "chunks_total": 50,
                "chunks_embedded": 30,
                "file": "lon.pdf",
            },
            "remove_file": {"seconds": 0.1, "chunks_total": 45, "chunks_embedded": 0, "file": "nho.pdf"},
        },
        "faiss": {"full_rebuild_seconds": 99.5, "reload_unchanged_seconds": 1.5},
        "meta": {"corpus_files": 3, "embedding_model": "m", "device": "cpu", "cpu_count": 16},
    }
    table = bench.timing_table(result)
    assert "| Dựng lần đầu | 99.5 giây | 100.0 giây |" in table
    assert "| Khởi động lại khi không có gì đổi | 1.5 giây | 0.50 giây |" in table
    assert "(`nho.pdf`, xử lý lại 5/50 đoạn) | 99.5 giây (dựng lại toàn bộ) | 2.0 giây |" in table
    assert "3 văn bản, 50 đoạn" in table and "CPU (16 luồng)" in table
    del result["faiss"]
    assert "không đo" in bench.timing_table(result)


def test_small_timings_keep_two_decimals_and_from_report_does_not_remeasure(tmp_path: Path):
    import json

    result = {
        "qdrant": {
            "first_build": {"seconds": 100.0, "chunks_total": 5, "chunks_embedded": 5},
            "restart_unchanged": {"seconds": 0.02, "chunks_total": 5, "chunks_embedded": 0},
            "update_small_file": {"seconds": 2.0, "chunks_total": 5, "chunks_embedded": 1, "file": "a.pdf"},
            "update_large_file": {"seconds": 3.0, "chunks_total": 5, "chunks_embedded": 2, "file": "b.pdf"},
            "remove_file": {"seconds": 0.34, "chunks_total": 3, "chunks_embedded": 0, "file": "a.pdf"},
        },
        "faiss": {"full_rebuild_seconds": 90.0, "reload_unchanged_seconds": 0.01},
    }
    report, readme = tmp_path / "r.json", tmp_path / "README.md"
    report.write_text(json.dumps(result), encoding="utf-8")
    readme.write_text("a\n<!-- indextiming:start -->\ncũ\n<!-- indextiming:end -->\nb\n", encoding="utf-8")

    assert bench.main(["--from-report", str(report), "--readme", str(readme)]) == 0
    text = readme.read_text(encoding="utf-8")
    assert "| 0.01 giây | 0.02 giây |" in text and "| 0.34 giây |" in text and "cũ" not in text
