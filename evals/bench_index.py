"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>

Đo thời gian dựng và cập nhật chỉ mục: Qdrant (cập nhật từng file) so với FAISS (dựng lại toàn bộ).

    python -m evals.bench_index --out evals/reports/final-index-timing.json

Các bước: Qdrant dựng lần đầu, khởi động lại khi không đổi, sửa một file nhỏ rồi một file lớn và xoá một file
(trên bản sao), rồi FAISS dựng lại toàn bộ. Thời gian tải model embedding được tính riêng.
"""

import argparse
import json
import os
import platform
import shutil
import tempfile
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings

from evals.common import EVALS_DIR
from ragbot import indexing, ingest
from ragbot.config import PROJECT_ROOT, Settings, parse_overrides
from ragbot.vectorstore import QdrantStore, all_chunks

REPORTS_DIR = EVALS_DIR / "reports"


def touch_pdf(path: Path) -> None:
    """Đổi hash của file mà không đổi nội dung chữ (thêm một ký tự xuống dòng sau %%EOF)."""
    with path.open("ab") as handle:
        handle.write(b"\n")


def pick_files(docs_dir: Path) -> tuple[Path, Path]:
    """(file nhỏ nhất, file lớn nhất) theo dung lượng PDF."""
    pdfs = sorted(ingest.list_pdfs(docs_dir), key=lambda p: p.stat().st_size)
    return pdfs[0], pdfs[-1]


def _sync(settings: Settings, embeddings: Embeddings) -> dict[str, Any]:
    store = QdrantStore.open(settings, embeddings)
    report = indexing.sync_store(settings, store)
    store.close()
    return {
        "seconds": round(report.seconds, 2),
        "added": report.added,
        "updated": report.updated,
        "removed": report.removed,
        "unchanged": len(report.unchanged),
        "chunks_embedded": report.chunks_embedded,
        "chunks_total": report.chunks_total,
    }


def run_benchmark(
    settings: Settings, embeddings: Embeddings, workdir: Path, include_faiss: bool = True
) -> dict[str, Any]:
    """`settings.index_dir` giữ chỉ mục Qdrant chính; thao tác sửa/xoá chạy trên bản sao trong `workdir`."""
    qdrant = settings.model_copy(update={"vector_store": "qdrant", "qdrant_url": ""})
    shutil.rmtree(qdrant.index_dir, ignore_errors=True)
    result: dict[str, Any] = {"qdrant": {}}

    result["qdrant"]["first_build"] = _sync(qdrant, embeddings)
    result["qdrant"]["restart_unchanged"] = _sync(qdrant, embeddings)

    copy_docs, copy_index = workdir / "corpus", workdir / "index"
    shutil.copytree(settings.docs_dir, copy_docs)
    shutil.copytree(qdrant.index_dir, copy_index)
    edited = qdrant.model_copy(update={"docs_dir": copy_docs, "index_dir": copy_index})
    small, large = pick_files(copy_docs)
    for label, victim in (("update_small_file", small), ("update_large_file", large)):
        touch_pdf(victim)
        step = _sync(edited, embeddings)
        step["file"] = victim.name
        result["qdrant"][label] = step
    (copy_docs / small.name).unlink()
    result["qdrant"]["remove_file"] = {**_sync(edited, embeddings), "file": small.name}

    if include_faiss:
        faiss = settings.model_copy(update={"vector_store": "faiss", "index_dir": workdir / "faiss"})
        started = time.perf_counter()
        store = indexing.load_or_build_index(faiss, embeddings)
        result["faiss"] = {
            "full_rebuild_seconds": round(time.perf_counter() - started, 2),
            "chunks": len(all_chunks(store)),
        }
        started = time.perf_counter()
        indexing.load_or_build_index(faiss, embeddings)  # lần thứ hai: nạp từ cache (pickle)
        result["faiss"]["reload_unchanged_seconds"] = round(time.perf_counter() - started, 2)
        full = result["faiss"]["full_rebuild_seconds"]
        for label in ("update_small_file", "update_large_file"):
            seconds = result["qdrant"][label]["seconds"]
            result.setdefault("speedup_vs_faiss_rebuild", {})[label] = (
                round(full / seconds, 1) if seconds else None
            )
    return result


def timing_table(result: dict[str, Any]) -> str:
    """Bảng markdown cho README, mọi số lấy từ báo cáo đo."""
    q, f = result["qdrant"], result.get("faiss", {})
    chunks = q["first_build"]["chunks_total"]
    rebuild = f.get("full_rebuild_seconds")
    full = "không đo" if rebuild is None else f"{rebuild:.1f} giây (dựng lại toàn bộ)"

    def sec(value: float | None) -> str:
        if value is None:
            return "không đo"
        return f"{value:.2f} giây" if value < 1 else f"{value:.1f} giây"

    def edit_row(label: str, step: dict[str, Any]) -> tuple[str, str, str]:
        redone = f"xử lý lại {step['chunks_embedded']}/{step['chunks_total']} đoạn"
        return (f"{label} (`{step['file']}`, {redone})", full, sec(step["seconds"]))

    rows = [
        ("Dựng lần đầu", sec(rebuild), sec(q["first_build"]["seconds"])),
        (
            "Khởi động lại khi không có gì đổi",
            sec(f.get("reload_unchanged_seconds")),
            sec(q["restart_unchanged"]["seconds"]),
        ),
        edit_row("Sửa 1 file nhỏ", q["update_small_file"]),
        edit_row("Sửa 1 file lớn", q["update_large_file"]),
        (f"Xóa 1 file (`{q['remove_file']['file']}`)", full, sec(q["remove_file"]["seconds"])),
    ]
    meta = result.get("meta", {})
    lines = ["| Thao tác | FAISS | Qdrant |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in rows]
    note = (
        f"Đo trên {meta.get('corpus_files', '?')} văn bản, {chunks} đoạn, "
        f"embedding `{meta.get('embedding_model')}` "
        f"chạy trên {str(meta.get('device', 'cpu')).upper()} ({meta.get('cpu_count')} luồng); "
        "báo cáo: `final-index-timing.json`."
    )
    return "\n".join(lines) + "\n\n" + note


def main(argv: Sequence[str] | None = None) -> int:
    from ragbot.cli import configure_logging
    from ragbot.indexing import create_embeddings

    parser = argparse.ArgumentParser(description="Đo thời gian dựng/cập nhật chỉ mục")
    parser.add_argument("--out", type=Path, default=REPORTS_DIR / "final-index-timing.json")
    parser.add_argument("--index-dir", type=Path, default=PROJECT_ROOT / ".index-qdrant")
    parser.add_argument("--no-faiss", action="store_true", help="bỏ bước dựng FAISS (chậm)")
    parser.add_argument(
        "--readme", type=Path, help="điền bảng vào README (giữa các dấu <!-- indextiming:start -->)"
    )
    parser.add_argument(
        "--from-report", type=Path, help="không đo lại: chỉ dựng bảng README từ báo cáo có sẵn"
    )
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    if args.from_report:
        from evals.summarize import inject

        table = timing_table(json.loads(args.from_report.read_text(encoding="utf-8")))
        if args.readme:
            text = inject(args.readme.read_text(encoding="utf-8"), "indextiming", table)
            args.readme.write_text(text, encoding="utf-8")
        print(table)
        return 0

    overrides = {"docs_dir": "data/corpus", "pdf_loader": "pdfplumber", "chunking": "legal"}
    overrides |= {"index_dir": str(args.index_dir)} | parse_overrides(args.overrides)
    settings = Settings(**overrides)
    configure_logging(settings.log_level)

    started = time.perf_counter()
    embeddings = create_embeddings(settings)
    embeddings.embed_query("khởi động")  # nạp model xong trước khi bắt đầu bấm giờ
    model_load = round(time.perf_counter() - started, 2)

    with tempfile.TemporaryDirectory(prefix="ragbot-bench-") as tmp:
        result = run_benchmark(settings, embeddings, Path(tmp), include_faiss=not args.no_faiss)
    result["meta"] = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding_model": settings.embedding_model,
        "embedding_model_load_seconds": model_load,
        "device": "cpu",
        "cpu_count": os.cpu_count(),
        "machine": platform.processor() or platform.machine(),
        "corpus_files": len(ingest.list_pdfs(settings.docs_dir)),
        "chunking": settings.chunking,
        "pdf_loader": settings.pdf_loader,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nĐã ghi {args.out}")
    if args.readme:
        from evals.summarize import inject

        args.readme.write_text(
            inject(args.readme.read_text(encoding="utf-8"), "indextiming", timing_table(result)),
            encoding="utf-8",
        )
        print(f"Đã cập nhật {args.readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
