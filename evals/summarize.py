"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>

Tạo bảng markdown so sánh các báo cáo đánh giá (dùng cho README).

    python -m evals.summarize evals/reports/final-*-test.json
    python -m evals.summarize evals/reports/final-*-test.json --out evals/reports/final-ablation.md
    python -m evals.summarize evals/reports/final-*-test.json --readme README.md   # tự điền bảng vào README

`--readme` thay nội dung giữa các cặp `<!-- ablation:start -->`…`<!-- ablation:end -->` và
`<!-- dataset:start -->`…`<!-- dataset:end -->`. Mọi số lấy trực tiếp từ file báo cáo/dataset.
"""

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

DIACRITICS_CATEGORY = "no_diacritics"


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def fmt_ci(value: Any, ci: Sequence[float] | None) -> str:
    return fmt(value) if not ci else f"{fmt(value)} [{ci[0]:.2f}–{ci[1]:.2f}]"


def summary_row(report: dict[str, Any], path: Path) -> dict[str, Any]:
    meta, ret, gen = report["meta"], report["retrieval"], report.get("generation") or {}
    return {
        "label": meta.get("label") or path.stem,
        "split": meta.get("split"),
        "draft": bool(meta.get("include_unverified")),
        "limited": meta.get("limit") is not None,
        "n_answerable": ret["n_answerable"],
        "n_total": meta["dataset"]["total"],
        "hit1": fmt_ci(ret.get("hit@1"), ret.get("hit@1_ci95")),
        "mrr": fmt(ret.get("mrr")),
        "context_precision": fmt(ret.get("context_precision")),
        "answerable_accuracy": fmt_ci(gen.get("answerable_accuracy"), gen.get("answerable_accuracy_ci95")),
        "refusal_accuracy": fmt_ci(gen.get("refusal_accuracy"), gen.get("refusal_accuracy_ci95")),
        "threshold": meta.get("active_threshold"),
        "unsupported": len(meta.get("unsupported_examples") or []),
        "file": path.name,
    }


def ablation_table(rows: Sequence[dict[str, Any]]) -> str:
    header = (
        "| Cấu hình | hit@1 (95% CI) | MRR | Context precision | Answerable acc. (95% CI) | Refusal acc. "
        "(95% CI) | Ngưỡng | Báo cáo |\n|---|---|---|---|---|---|---|---|"
    )
    lines = [header]
    for r in rows:
        name = r["label"] + (" ⚠ NHÁP" if r["draft"] else "")
        lines.append(
            f"| {name} | {r['hit1']} | {r['mrr']} | {r['context_precision']} | {r['answerable_accuracy']} | "
            f"{r['refusal_accuracy']} | {fmt(r['threshold'], 2)} | `{r['file']}` |"
        )
    return "\n".join(lines)


def diacritics_table(reports: Sequence[tuple[dict[str, Any], Path]]) -> str:
    """Riêng nhóm câu gõ không dấu: nơi tác dụng của bước viết lại câu hỏi lộ rõ nhất."""
    lines = ["| Cấu hình | Số câu | hit@1 | Accuracy |", "|---|---|---|---|"]
    for report, path in reports:
        cat = report["retrieval"]["by_category"].get(DIACRITICS_CATEGORY)
        gen = (report.get("generation") or {}).get("by_category", {}).get(DIACRITICS_CATEGORY)
        if cat:
            label = report["meta"].get("label") or path.stem
            lines.append(
                f"| {label} | {cat['n']} | {fmt(cat.get('hit@1'))} | "
                f"{fmt(gen['accuracy']) if gen else 'n/a'} |"
            )
    return "\n".join(lines)


def warnings(rows: Sequence[dict[str, Any]]) -> list[str]:
    notes = []
    if any(r["split"] != "test" for r in rows):
        notes.append("Có báo cáo không phải split=test: bảng ablation chính thức chỉ nên dùng báo cáo test.")
    if len({r["n_total"] for r in rows}) > 1:
        notes.append("Các báo cáo có số câu khác nhau: không so sánh trực tiếp được.")
    if any(r["limited"] for r in rows):
        notes.append("Có báo cáo chỉ chạy một phần câu hỏi (--limit): không dùng làm số liệu công bố.")
    if any(r["draft"] for r in rows):
        notes.append("Có báo cáo gồm câu CHƯA duyệt tay (NHÁP): không dùng làm số liệu công bố.")
    for r in rows:
        if r["unsupported"]:
            notes.append(
                f"{r['label']}: {r['unsupported']} câu không có chunk nào chứa bằng chứng "
                "(tính là trượt retrieval)."
            )
    return notes


def render(paths: Sequence[Path]) -> str:
    reports = [(load_report(p), p) for p in paths]
    rows = [summary_row(report, path) for report, path in reports]
    parts = [ablation_table(rows)]
    if any(DIACRITICS_CATEGORY in report["retrieval"]["by_category"] for report, _ in reports):
        parts += ["", f"Riêng nhóm `{DIACRITICS_CATEGORY}`:", "", diacritics_table(reports)]
    notes = warnings(rows)
    if notes:
        parts += ["", *[f"> ⚠ {note}" for note in notes]]
    return "\n".join(parts)


def dataset_table(path: Path) -> str:
    """Số câu theo nhóm và split của bộ câu hỏi đã duyệt (mọi dòng trong dataset_legal.jsonl đều verified)."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cells: dict[str, dict[str, int]] = {}
    for row in rows:
        cell = cells.setdefault(row["category"], {"dev": 0, "test": 0})
        cell[row.get("split") or "dev"] += 1
    lines = ["| Nhóm câu hỏi | dev | test | Tổng |", "|---|---|---|---|"]
    for category, cell in sorted(cells.items()):
        lines.append(f"| {category} | {cell['dev']} | {cell['test']} | {cell['dev'] + cell['test']} |")
    dev, test = sum(c["dev"] for c in cells.values()), sum(c["test"] for c in cells.values())
    lines.append(f"| **Tổng** | **{dev}** | **{test}** | **{dev + test}** |")
    no_answer = sum(1 for r in rows if not r["answerable"])
    return (
        "\n".join(lines)
        + f"\n\n{no_answer}/{len(rows)} câu ({no_answer / len(rows):.0%}) không có đáp án trong tài liệu."
    )


def inject(text: str, name: str, content: str) -> str:
    """Thay nội dung giữa `<!-- name:start -->` và `<!-- name:end -->`; báo lỗi nếu README thiếu cặp dấu."""
    pattern = re.compile(
        rf"(<!-- {re.escape(name)}:start -->\n).*?(\n<!-- {re.escape(name)}:end -->)", re.DOTALL
    )
    if not pattern.search(text):
        raise ValueError(f"README thiếu cặp dấu <!-- {name}:start --> / <!-- {name}:end -->")
    return pattern.sub(lambda m: m.group(1) + content + m.group(2), text, count=1)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bảng markdown từ các báo cáo đánh giá")
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, help="ghi ra file markdown")
    parser.add_argument(
        "--readme", type=Path, help="điền bảng vào README (giữa các dấu <!-- ablation:start -->)"
    )
    parser.add_argument("--dataset", type=Path, default=Path(__file__).parent / "dataset_legal.jsonl")
    args = parser.parse_args(argv)
    missing = [p for p in args.reports if not p.is_file()]
    if missing:
        print(f"Không thấy: {', '.join(map(str, missing))}", file=sys.stderr)
        return 2
    text = render(args.reports)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    if args.readme:
        readme = inject(args.readme.read_text(encoding="utf-8"), "ablation", text)
        if args.dataset.is_file():
            readme = inject(readme, "dataset", dataset_table(args.dataset))
        args.readme.write_text(readme, encoding="utf-8")
        print(f"\nĐã cập nhật {args.readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
