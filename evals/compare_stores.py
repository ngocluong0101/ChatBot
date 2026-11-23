"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>

So sánh kết quả truy hồi của FAISS và Qdrant trên cùng một bộ câu hỏi.

    python -m evals.compare_stores --questions evals/drafts/draft-001.jsonl

Hai kho tương đương nếu cùng câu hỏi trả về cùng các đoạn, cùng thứ tự, điểm gần bằng nhau.
Chỉ so phần truy hồi (dense và hybrid, không reranker, không viết lại câu hỏi).
"""

import argparse
import json
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from evals.common import DRAFTS_DIR, EVALS_DIR, read_jsonl
from ragbot.config import PROJECT_ROOT, Settings, parse_overrides
from ragbot.retrieval import Retriever
from ragbot.vectorstore import content_hash

MODES = ("dense", "hybrid")
EXAMPLES_SHOWN = 10


def chunk_id(doc: Document) -> tuple[str, int | None, str]:
    return (doc.metadata.get("source", "?"), doc.metadata.get("page"), content_hash(doc.page_content))


def compare_mode(a: Any, b: Any, settings: Settings, queries: Sequence[str]) -> dict[str, Any]:
    """So `a` với `b` ở một chế độ truy hồi (settings.retrieval_mode, không reranker)."""
    ra, rb = Retriever(settings, a), Retriever(settings, b)
    identical = same_set = top1 = 0
    overlaps: list[float] = []
    max_diff = 0.0
    examples: list[dict[str, Any]] = []
    for query in queries:
        hits_a, hits_b = ra.search(query), rb.search(query)
        ids_a, ids_b = [chunk_id(d) for d, _ in hits_a], [chunk_id(d) for d, _ in hits_b]
        identical += ids_a == ids_b
        same_set += set(ids_a) == set(ids_b)
        top1 += bool(ids_a) and bool(ids_b) and ids_a[0] == ids_b[0]
        overlaps.append(len(set(ids_a) & set(ids_b)) / max(len(ids_a), 1))
        scores_b = {chunk_id(d): s for d, s in hits_b}
        for doc, score in hits_a:
            if chunk_id(doc) in scores_b:
                max_diff = max(max_diff, abs(score - scores_b[chunk_id(doc)]))
        if ids_a != ids_b and len(examples) < EXAMPLES_SHOWN:
            examples.append({"query": query, "a": [i[:2] for i in ids_a], "b": [i[:2] for i in ids_b]})
    n = len(queries)
    return {
        "queries": n,
        "identical_ranking": identical / n if n else None,
        "same_top_k_set": same_set / n if n else None,
        "top1_agreement": top1 / n if n else None,
        "mean_overlap_at_k": sum(overlaps) / n if n else None,
        "max_abs_score_difference": max_diff,
        "mismatches_shown": examples,
    }


def compare(a: Any, b: Any, settings: Settings, queries: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"top_k": settings.top_k}
    for mode in MODES:
        started = time.perf_counter()
        mode_settings = settings.model_copy(update={"retrieval_mode": mode, "rerank_model": ""})
        result[mode] = compare_mode(a, b, mode_settings, queries)
        result[mode]["seconds"] = round(time.perf_counter() - started, 1)
    return result


def load_queries(path: Path, limit: int | None = None) -> list[str]:
    rows = read_jsonl(path)
    questions = list(dict.fromkeys(r["question"] for r in rows))  # bỏ trùng, giữ thứ tự
    return questions[:limit] if limit else questions


def equivalence_table(result: dict[str, Any]) -> str:
    """Bảng markdown cho README, mọi số lấy từ báo cáo so sánh."""

    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.1%}"

    lines = [
        "| Chế độ truy hồi | Số câu | Cùng thứ tự top-"
        f"{result['top_k']} | Cùng tập top-{result['top_k']} | Cùng đoạn đứng đầu | Chênh điểm lớn nhất |",
        "|---|---|---|---|---|---|",
    ]
    for mode in MODES:
        r = result[mode]
        lines.append(
            f"| {mode} | {r['queries']} | {pct(r['identical_ranking'])} | {pct(r['same_top_k_set'])} | "
            f"{pct(r['top1_agreement'])} | {r['max_abs_score_difference']:.2g} |"
        )
    meta = result.get("meta", {})
    return "\n".join(lines) + (
        f"\n\nSo trên cùng bộ câu hỏi (`{meta.get('questions_file')}`) và cùng các đoạn tài liệu; "
        "báo cáo: `final-store-equivalence.json`."
    )


def main(argv: Sequence[str] | None = None) -> int:
    from ragbot.cli import configure_logging
    from ragbot.indexing import create_embeddings, load_or_build_index

    parser = argparse.ArgumentParser(description="So sánh truy hồi FAISS và Qdrant")
    parser.add_argument("--questions", type=Path, default=DRAFTS_DIR / "draft-001.jsonl")
    parser.add_argument("--faiss-index", type=Path, default=PROJECT_ROOT / ".index-legal")
    parser.add_argument("--qdrant-index", type=Path, default=PROJECT_ROOT / ".index-qdrant")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--readme", type=Path, help="điền bảng vào README (giữa các dấu <!-- equivalence:start -->)"
    )
    parser.add_argument("--out", type=Path, default=EVALS_DIR / "reports" / "final-store-equivalence.json")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    base = {"docs_dir": "data/corpus", "pdf_loader": "pdfplumber", "chunking": "legal"} | parse_overrides(
        args.overrides
    )
    settings = Settings(**base)
    configure_logging(settings.log_level)
    embeddings = create_embeddings(settings)
    faiss = load_or_build_index(
        Settings(**base, vector_store="faiss", index_dir=str(args.faiss_index)), embeddings
    )
    qdrant = load_or_build_index(
        Settings(**base, vector_store="qdrant", qdrant_url="", index_dir=str(args.qdrant_index)), embeddings
    )
    queries = load_queries(args.questions, args.limit)
    result = compare(faiss, qdrant, settings, queries)
    result["meta"] = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "questions_file": args.questions.name,
        "embedding_model": settings.embedding_model,
        "chunking": settings.chunking,
        "note": "Câu hỏi nháp chưa duyệt chỉ làm đầu vào truy vấn; phép so sánh không cần nhãn đúng/sai.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {m: {k: v for k, v in result[m].items() if k != "mismatches_shown"} for m in MODES}, indent=2
        )
    )
    print(f"\nĐã ghi {args.out}")
    if args.readme:
        from evals.summarize import inject

        args.readme.write_text(
            inject(args.readme.read_text(encoding="utf-8"), "equivalence", equivalence_table(result)),
            encoding="utf-8",
        )
        print(f"Đã cập nhật {args.readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
