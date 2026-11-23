"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>

Duyệt tay câu hỏi nháp, rồi gộp câu đã duyệt vào bộ đánh giá chính thức.

    python -m evals.review --draft evals/drafts/draft-001.jsonl      # duyệt từng câu
    python -m evals.review --draft evals/drafts/draft-001.jsonl --stats
    python -m evals.review --finalize     # gộp câu đã duyệt vào evals/dataset_legal.jsonl

Mỗi quyết định được ghi ngay, thoát giữa chừng rồi chạy lại là tiếp tục.
Phím: y = dùng | n = loại | e = sửa | g = tìm trong tài liệu | s = để sau | q = thoát.
Cần kiểm tra: câu hỏi tự đủ nghĩa; must_include và evidence đúng với đoạn gốc; với câu KHÔNG có đáp án
(near_domain, off_topic) thì tài liệu thật sự không trả lời được (dùng phím g để tìm).
"""

import argparse
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from evals.common import DATASET_LEGAL, DRAFTS_DIR, assign_splits, read_jsonl, write_jsonl
from ragbot.evaluate import contains_term, normalize

PRIVATE_FIELDS = ("_review", "review")
DATASET_FIELDS = (
    "id",
    "question",
    "category",
    "answerable",
    "gold_sources",
    "evidence",
    "must_include",
    "split",
    "difficulty",
    "verified",
)
EXCERPT_CHARS = 700
SearchFn = Callable[[str], list[str]]


def is_pending(row: dict[str, Any]) -> bool:
    return row.get("review") not in ("accepted", "rejected")


def progress(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = {"accepted": 0, "rejected": 0, "pending": 0}
    for row in rows:
        counts[row.get("review") or "pending"] += 1
    return counts


def format_groups(groups: Sequence[Sequence[str]]) -> str:
    """[["12 ngày", "mười hai ngày"], ["VPN"]] -> '12 ngày | mười hai ngày ; VPN'."""
    return " ; ".join(" | ".join(group) for group in groups)


def parse_groups(text: str) -> list[list[str]]:
    groups = [[alt.strip() for alt in part.split("|") if alt.strip()] for part in text.split(";")]
    return [g for g in groups if g]


def highlight(text: str, phrases: Sequence[str]) -> str:
    for phrase in phrases:
        text = re.sub(f"({re.escape(phrase)})", r"[[\1]]", text, count=1, flags=re.IGNORECASE)
    return text


def render(row: dict[str, Any], position: str = "") -> str:
    review = row.get("_review", {})
    lines = [
        "=" * 78,
        f"{position}  {row['id']}  [{row['category']} / {row.get('difficulty')} / {row.get('split')}]",
        f"CÂU HỎI: {row['question']}",
    ]
    if row["answerable"]:
        lines.append(f"ĐÁP ÁN BẮT BUỘC (mọi nhóm phải có): {format_groups(row['must_include'])}")
        lines.append(f"NGUỒN: {', '.join(row['gold_sources'])}")
        phrases = [alt for group in row["must_include"] for alt in group]
        for chunk in review.get("chunks", []):
            where = f"{chunk.get('doc_title')}, {chunk.get('section')}"
            if chunk.get("khoan"):
                where += f", khoản {chunk['khoan']}"
            text = highlight(chunk.get("text", ""), phrases)
            lines.append(f"\n--- ĐOẠN GỐC ({where}, trang {chunk.get('page')}) ---")
            lines.append(text[:EXCERPT_CHARS] + (" …" if len(text) > EXCERPT_CHARS else ""))
    else:
        lines.append("KHÔNG CÓ ĐÁP ÁN trong tài liệu (y = đúng là tài liệu không trả lời được).")
    for warning in review.get("warnings", []):
        lines.append(f"⚠ {warning}")
    return "\n".join(lines)


def answer_keys_problem(row: dict[str, Any]) -> str | None:
    """Sau khi sửa tay: mỗi nhóm đáp án phải còn xuất hiện trong một đoạn gốc đã lưu cùng câu hỏi."""
    texts = [normalize(c.get("text", "")) for c in row.get("_review", {}).get("chunks", [])]
    if not texts or not row.get("must_include"):
        return None
    for group in row["must_include"]:
        if not any(contains_term(text, alt) for text in texts for alt in group):
            return f"nhóm {group} không có trong đoạn gốc nào"
    return None


def edit(row: dict[str, Any], input_fn: Callable[[str], str], print_fn: Callable[[str], None]) -> None:
    question = input_fn("Câu hỏi mới (Enter = giữ): ").strip()
    if question:
        row["question"] = question
    if row["answerable"]:
        current = format_groups(row["must_include"])
        groups = input_fn(
            f"Đáp án [{current}] (nhóm cách nhau ';', cách viết tương đương cách nhau '|'): "
        ).strip()
        if groups:
            row["must_include"] = parse_groups(groups)
        if problem := answer_keys_problem(row):
            print_fn(f"⚠ {problem}: câu này sẽ khó được chấm đúng, hãy kiểm tra lại")
    difficulty = input_fn("Độ khó easy/medium/hard (Enter = giữ): ").strip()
    if difficulty in ("easy", "medium", "hard"):
        row["difficulty"] = difficulty


def review_loop(
    rows: list[dict[str, Any]],
    path: Path,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
    search: SearchFn | None = None,
) -> bool:
    """Duyệt các câu chưa quyết định. True nếu duyệt xong hết; False nếu còn câu để sau hoặc thoát."""
    todo = [row for row in rows if is_pending(row)]
    skipped: list[dict[str, Any]] = []
    for index, row in enumerate(todo, start=1):
        while True:
            print_fn(render(row, f"({index}/{len(todo)})"))
            choice = (
                input_fn("[y] dùng  [n] loại  [e] sửa  [g] tìm  [s] để sau  [q] thoát > ").strip().lower()
            )
            if choice == "g":
                if search is None:
                    print_fn("(chưa nạp được tài liệu để tìm)")
                    continue
                for hit in search(input_fn("Từ khóa (Enter = dùng câu hỏi): ").strip() or row["question"]):
                    print_fn(hit)
                continue
            if choice == "e":
                edit(row, input_fn, print_fn)
                write_jsonl(path, rows)
                continue
            if choice in ("y", "n", "s", "q"):
                break
            print_fn("Phím không hợp lệ.")
        if choice == "q":
            write_jsonl(path, rows)
            return False
        if choice == "y":
            row["review"], row["verified"] = "accepted", True
        elif choice == "n":
            row["review"], row["verified"] = "rejected", False
        else:
            skipped.append(row)
        write_jsonl(path, rows)
    return not skipped


def to_dataset_row(row: dict[str, Any]) -> dict[str, Any]:
    out = {k: row[k] for k in DATASET_FIELDS if k in row}
    out.pop("split", None)  # split trong bản nháp chỉ là tạm; split chính thức được gán khi gộp
    out["verified"] = True
    if not out.get("answerable"):
        for key in ("gold_sources", "evidence", "must_include"):
            out.pop(key, None)
    return out


def finalize(draft_paths: Sequence[Path], dataset_path: Path = DATASET_LEGAL) -> dict[str, Any]:
    """Gộp câu đã duyệt vào bộ chính thức. Chạy lại được nhiều lần: câu đã có giữ nguyên, kể cả split."""
    existing = {row["id"]: row for row in read_jsonl(dataset_path)}
    added = []
    for path in draft_paths:
        for row in read_jsonl(path):
            if row.get("review") == "accepted" and row["id"] not in existing:
                existing[row["id"]] = to_dataset_row(row)
                added.append(row["id"])
    rows = list(existing.values())
    splits = assign_splits(rows)
    for row in rows:
        row["split"] = splits[row["id"]]
    rows.sort(key=lambda r: r["id"])
    write_jsonl(dataset_path, rows)
    table: dict[str, dict[str, int]] = {}
    for row in rows:
        cell = table.setdefault(row["category"], {"dev": 0, "test": 0})
        cell[row["split"]] += 1
    return {"total": len(rows), "added": len(added), "by_category": table}


def make_search(settings_overrides: Sequence[str] = ()) -> SearchFn:
    """Tìm nhanh trong corpus bằng BM25 (nạp tài liệu ở lần gọi đầu, mất vài chục giây)."""
    cache: dict[str, Any] = {}

    def search(query: str) -> list[str]:
        if "index" not in cache:
            from ragbot import ingest
            from ragbot.config import Settings, parse_overrides
            from ragbot.lexical import BM25Index

            print("(đang nạp tài liệu để tìm, vài chục giây…)", file=sys.stderr)
            overrides = {"docs_dir": "data/corpus", "pdf_loader": "pdfplumber", "chunking": "legal"}
            settings = Settings(**(overrides | parse_overrides(settings_overrides)))
            chunks = ingest.split_documents(ingest.load_documents(settings), settings)
            cache["chunks"], cache["index"] = chunks, BM25Index(chunks)
        hits = []
        for i, score in cache["index"].top(query, 4):
            chunk = cache["chunks"][i]
            body = chunk.page_content.replace("\n", " ")
            hits.append(
                f"  [{score:.1f}] {chunk.metadata['source']} | {chunk.metadata.get('section')}\n"
                f"      {body[:300]} …"
            )
        return hits or ["  (không có đoạn nào khớp từ khóa)"]

    return search


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Duyệt câu hỏi nháp và gộp vào bộ đánh giá")
    parser.add_argument("--draft", type=Path, default=DRAFTS_DIR / "draft-001.jsonl")
    parser.add_argument("--stats", action="store_true", help="chỉ in tiến độ duyệt")
    parser.add_argument("--finalize", action="store_true", help="gộp câu đã duyệt vào dataset_legal.jsonl")
    parser.add_argument("--dataset", type=Path, default=DATASET_LEGAL)
    args = parser.parse_args(argv)

    if args.finalize:
        summary = finalize([args.draft], args.dataset)
        print(f"Đã ghi {args.dataset}: {summary['total']} câu (thêm {summary['added']}).")
        for category, cell in sorted(summary["by_category"].items()):
            print(f"  {category:<14} dev={cell['dev']:<3} test={cell['test']}")
        return 0

    rows = read_jsonl(args.draft)
    if not rows:
        print(f"Không có câu nháp ở {args.draft}", file=sys.stderr)
        return 2
    counts = progress(rows)
    print(f"Tiến độ: đã dùng {counts['accepted']}, đã loại {counts['rejected']}, còn {counts['pending']}")
    if args.stats:
        return 0
    done = review_loop(rows, args.draft, search=make_search())
    counts = progress(rows)
    print(f"\nTiến độ: đã dùng {counts['accepted']}, đã loại {counts['rejected']}, còn {counts['pending']}")
    if done:
        print("Duyệt xong. Chạy: python -m evals.review --finalize")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
