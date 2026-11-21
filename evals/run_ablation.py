"""Chạy ablation 5 cấu hình, mỗi lần đổi MỘT thứ, cùng LLM và cùng corpus.

    python -m evals.run_ablation --dry-run                      # chỉ in các lệnh sẽ chạy
    python -m evals.run_ablation                                # chạy thật (chỉ dùng câu đã duyệt tay)
    python -m evals.run_ablation --include-unverified           # chạy THỬ quy trình trên câu nháp
    python -m evals.run_ablation --only 03,04 --skip-existing   # chạy tiếp những cấu hình còn thiếu

Mỗi cấu hình gồm hai lần chạy `ragbot.evaluate`:
  1) dev  : chỉ đo retrieval và quét ngưỡng từ chối trên dev  -> evals/reports/dev-<tên>.json
  2) test : chạy đầy đủ trên test, dùng NGUYÊN ngưỡng chọn ở bước 1 -> evals/reports/final-<tên>-test.json
Chạy thử với câu chưa duyệt ghi ra `draft-*` thay vì `final-*` để không bao giờ lọt vào git/README.
"""

import argparse
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from evals.common import DATASET_LEGAL, EVALS_DIR

REPORTS_DIR = EVALS_DIR / "reports"
RERANKER = "BAAI/bge-reranker-v2-m3"

BASE = {
    "docs_dir": "data/corpus",
    "pdf_loader": "pdfplumber",
    "vector_store": "qdrant",  # ghim rõ: kết quả ablation không phụ thuộc vào mặc định của cấu hình
    "llm_model": "qwen2.5:7b-instruct",
    "top_k": "5",
    "fetch_k": "10",
}

# Mỗi cấu hình khác cấu hình trước đúng một thứ (ghi ở cột "đổi").
CONFIGS: list[tuple[str, str, dict[str, str], str]] = [
    (
        "01-dense-fixed",
        "1. Dense, chunk theo độ dài",
        {"chunking": "fixed", "retrieval_mode": "dense", "rerank_model": "", "rewrite_queries": "false"},
        "gốc",
    ),
    (
        "02-dense-legal",
        "2. + chunk theo Điều/Khoản",
        {"chunking": "legal", "retrieval_mode": "dense", "rerank_model": "", "rewrite_queries": "false"},
        "chunking",
    ),
    (
        "03-hybrid",
        "3. + BM25 hybrid",
        {"chunking": "legal", "retrieval_mode": "hybrid", "rerank_model": "", "rewrite_queries": "false"},
        "retrieval_mode",
    ),
    (
        "04-rerank",
        "4. + reranker",
        {
            "chunking": "legal",
            "retrieval_mode": "hybrid",
            "rerank_model": RERANKER,
            "rewrite_queries": "false",
        },
        "rerank_model",
    ),
    (
        "05-rewrite",
        "5. + viết lại câu hỏi",
        {
            "chunking": "legal",
            "retrieval_mode": "hybrid",
            "rerank_model": RERANKER,
            "rewrite_queries": "true",
        },
        "rewrite_queries",
    ),
]


@dataclass(frozen=True)
class Command:
    step: str  # "dev" | "test"
    name: str
    argv: list[str]
    out: Path


def settings_for(overrides: dict[str, str]) -> dict[str, str]:
    """Mỗi kiểu chunking dùng index riêng, nếu không hai cấu hình sẽ ghi đè index của nhau."""
    merged = {**BASE, **overrides}
    merged["index_dir"] = f".index-{merged['chunking']}"
    return merged


def build_commands(
    name: str,
    label: str,
    overrides: dict[str, str],
    dataset: Path,
    reports_dir: Path = REPORTS_DIR,
    include_unverified: bool = False,
    judge_model: str | None = None,
    limit: int | None = None,
) -> list[Command]:
    prefix = "draft" if include_unverified else "final"
    dev_out = (
        reports_dir / f"dev-{name}.json" if not include_unverified else reports_dir / f"draft-dev-{name}.json"
    )
    test_out = reports_dir / f"{prefix}-{name}-test.json"
    common = ["--dataset", str(dataset), "--label", label, "--allow-unsupported"]
    for key, value in settings_for(overrides).items():
        common += ["--set", f"{key}={value}"]
    if include_unverified:
        common.append("--include-unverified")
    if limit is not None:
        common += ["--limit", str(limit)]
    dev = [*common, "--split", "dev", "--mode", "retrieval", "--out", str(dev_out)]
    test = [
        *common,
        "--split",
        "test",
        "--mode",
        "full",
        "--threshold-from",
        str(dev_out),
        "--out",
        str(test_out),
    ]
    if judge_model:
        test += ["--judge-model", judge_model]
    return [Command("dev", name, dev, dev_out), Command("test", name, test, test_out)]


def select_configs(only: str | None) -> list[tuple[str, str, dict[str, str], str]]:
    if not only:
        return CONFIGS
    wanted = {part.strip() for part in only.split(",") if part.strip()}
    chosen = [c for c in CONFIGS if c[0].split("-")[0] in wanted or c[0] in wanted]
    unknown = wanted - {c[0].split("-")[0] for c in chosen} - {c[0] for c in chosen}
    if unknown:
        raise SystemExit(f"--only không hợp lệ: {', '.join(sorted(unknown))}. Hợp lệ: 01..05")
    return chosen


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chạy ablation 5 cấu hình (dev -> test)")
    parser.add_argument("--dataset", type=Path, default=DATASET_LEGAL)
    parser.add_argument("--only", help="ví dụ 03,04")
    parser.add_argument("--skip-existing", action="store_true", help="bỏ qua bước đã có file báo cáo")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-unverified", action="store_true", help="CHỈ để chạy thử; ghi ra draft-*")
    parser.add_argument("--judge-model", help="bật chấm faithfulness ở lần chạy test")
    parser.add_argument("--limit", type=int, help="CHỈ để chạy thử (bắt buộc kèm --include-unverified)")
    args = parser.parse_args(argv)
    if args.limit is not None and not args.include_unverified:
        raise SystemExit(
            "--limit chỉ dùng để chạy thử: thêm --include-unverified (báo cáo sẽ ghi ra draft-*)"
        )

    for name, label, overrides, _ in select_configs(args.only):
        for command in build_commands(
            name,
            label,
            overrides,
            args.dataset,
            REPORTS_DIR,
            args.include_unverified,
            args.judge_model,
            args.limit,
        ):
            full = [sys.executable, "-m", "ragbot.evaluate", *command.argv]
            print(f"\n[{command.name} / {command.step}] {' '.join(full)}")
            if args.dry_run:
                continue
            if args.skip_existing and command.out.is_file():
                print("  đã có báo cáo, bỏ qua")
                continue
            started = time.perf_counter()
            result = subprocess.run(full, check=False)
            print(f"  xong sau {time.perf_counter() - started:.0f} giây (mã {result.returncode})")
            if result.returncode != 0:
                print("Dừng vì lỗi. Sửa rồi chạy lại với --skip-existing.", file=sys.stderr)
                return result.returncode
    prefix = "draft" if args.include_unverified else "final"
    print(f"\nTổng hợp: python -m evals.summarize evals/reports/{prefix}-*-test.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
