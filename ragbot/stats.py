"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>

Thống kê corpus sau khi đọc và chia chunk (không cần embedding hay Ollama).

    python -m ragbot.stats --set docs_dir=data/corpus --set chunking=legal --set pdf_loader=pdfplumber
    python -m ragbot.stats ... --out evals/reports/corpus-stats.json

Ngoài số văn bản / trang / chunk / độ dài trung bình, liệt kê dấu hiệu trích xuất sai để kiểm tra tay: văn bản
không nhận ra cấu trúc Điều, Điều bị thiếu trong dãy số, văn bản chưa có trong SOURCES.md.
"""

import argparse
import json
import statistics
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from ragbot import corpus, ingest
from ragbot.cli import configure_logging
from ragbot.config import Settings, parse_overrides


def _dieu_numbers(chunks: Sequence[Document]) -> list[int]:
    numbers = {
        int("".join(ch for ch in str(c.metadata["dieu"]) if ch.isdigit()))
        for c in chunks
        if c.metadata.get("dieu")
    }
    return sorted(numbers)


# Điều cuối của văn bản đầy đủ thường là điều khoản thi hành; nếu không, PDF có thể bị cắt dở.
_ENDING_WORDS = (
    "thi hành",
    "hiệu lực",
    "chuyển tiếp",
    "quy định chi tiết",
    "tổ chức thực hiện",
    "điều khoản",
)


def _ends_like_a_law(title: str | None) -> bool:
    return bool(title) and any(word in title.casefold() for word in _ENDING_WORDS)


def _missing(numbers: Sequence[int]) -> list[int]:
    return [n for n in range(numbers[0], numbers[-1] + 1) if n not in set(numbers)] if numbers else []


def compute_stats(
    pages: Sequence[Document],
    chunks: Sequence[Document],
    known: dict[str, Any],
    all_files: Sequence[str] = (),
) -> dict[str, Any]:
    """`known`: tên file có trong SOURCES.md; `all_files`: mọi PDF trong thư mục (để phát hiện
    file không trích được chữ)."""
    lengths = [len(c.page_content) for c in chunks]
    files = sorted({d.metadata.get("source", "?") for d in pages} | set(all_files))
    documents = []
    for name in files:
        mine = [c for c in chunks if c.metadata.get("source") == name]
        numbers = _dieu_numbers(mine)
        documents.append(
            {
                "file": name,
                "title": next((c.metadata["doc_title"] for c in mine if c.metadata.get("doc_title")), None),
                "pages": sum(1 for d in pages if d.metadata.get("source") == name),
                "chunks": len(mine),
                "dieu": len(numbers),
                "dieu_range": [numbers[0], numbers[-1]] if numbers else None,
                "last_dieu_title": next(
                    (c.metadata.get("dieu_title") for c in reversed(mine) if c.metadata.get("dieu")), None
                ),
                "dieu_missing": _missing(numbers),
                "tables": sum(c.metadata.get("type") == "table" for c in mine),
                "structured": bool(numbers),
                "no_text": not any(d.metadata.get("source") == name for d in pages),
                "in_sources": name in known,
            }
        )
    return {
        "documents": len(files),
        "pages": len(pages),
        "chunks": len(chunks),
        "chunk_chars": {
            "mean": round(statistics.fmean(lengths), 1) if lengths else None,
            "median": statistics.median(lengths) if lengths else None,
            "max": max(lengths, default=None),
            "min": min(lengths, default=None),
        },
        "chunk_types": dict(Counter(c.metadata.get("type", "n/a") for c in chunks)),
        "per_document": documents,
    }


def format_stats(stats: dict[str, Any]) -> str:
    chars = stats["chunk_chars"]
    lines = [
        f"Số văn bản : {stats['documents']}",
        f"Số trang   : {stats['pages']} (trang có văn bản)",
        f"Số chunk   : {stats['chunks']}  {stats['chunk_types']}",
        f"Độ dài chunk (ký tự): TB {chars['mean']}, trung vị {chars['median']}, "
        f"nhỏ nhất {chars['min']}, lớn nhất {chars['max']}",
        "",
        f"{'file':<34}{'trang':>6}{'chunk':>7}{'Điều':>6}{'bảng':>6}  ghi chú",
    ]
    warnings = []
    for d in stats["per_document"]:
        notes = []
        if d["no_text"]:
            notes.append("KHÔNG trích được chữ (PDF scan/ảnh: cần bản có text layer hoặc OCR)")
        elif not d["structured"]:
            notes.append("KHÔNG nhận ra cấu trúc Điều (chia theo mục/độ dài)")
        if d["structured"] and not _ends_like_a_law(d["last_dieu_title"]):
            notes.append(f"Điều cuối là '{d['last_dieu_title']}': văn bản có thể bị cắt dở")
        if d["dieu_missing"]:
            shown = ", ".join(map(str, d["dieu_missing"][:12])) + (
                " ..." if len(d["dieu_missing"]) > 12 else ""
            )
            notes.append(f"thiếu Điều {shown}")
        if not d["in_sources"]:
            notes.append("chưa có trong SOURCES.md")
        lines.append(
            f"{d['file'][:33]:<34}{d['pages']:>6}{d['chunks']:>7}{d['dieu']:>6}{d['tables']:>6}  "
            + "; ".join(notes)
        )
        warnings.extend(notes)
    if warnings:
        lines += [
            "",
            "Cần kiểm tra tay: các ghi chú ở cột cuối. Điều 'thiếu' có thể là Điều đã bị bãi bỏ,"
            " hoặc dấu hiệu trích xuất PDF lỗi (đối chiếu với file gốc).",
        ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Thống kê corpus sau khi chia chunk")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--out", type=Path, help="ghi thống kê JSON ra file này")
    args = parser.parse_args(argv)

    settings = Settings(**parse_overrides(args.overrides))
    configure_logging(settings.log_level)
    pages = ingest.load_documents(settings)
    chunks = ingest.split_documents(pages, settings)
    on_disk = sorted(p.name for p in settings.docs_dir.glob("**/*.pdf"))
    stats = compute_stats(pages, chunks, corpus.load_sources(settings.sources_file), on_disk)
    stats["settings"] = {
        k: str(getattr(settings, k))
        for k in ("docs_dir", "pdf_loader", "chunking", "chunk_size", "chunk_overlap")
    }
    print(format_stats(stats))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nĐã ghi: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
