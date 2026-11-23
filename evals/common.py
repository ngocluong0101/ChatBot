"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>

Hàm dùng chung cho các công cụ trong evals/: đọc/ghi JSONL, chia dev/test cố định bằng seed.
"""

import json
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

EVALS_DIR = Path(__file__).resolve().parent
DATASET_LEGAL = EVALS_DIR / "dataset_legal.jsonl"
DRAFTS_DIR = EVALS_DIR / "drafts"
SPLIT_SEED = 20261120


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)  # ghi nguyên tử: ngắt giữa chừng không để lại file cụt


def assign_splits(
    rows: Sequence[dict[str, Any]], seed: int = SPLIT_SEED, ratio_dev: float = 0.5
) -> dict[str, str]:
    """Gán dev/test theo category, cố định bằng seed.

    Câu đã có split giữ nguyên (đổi split sẽ làm lộ tập test vào bước chọn ngưỡng).
    """
    result = {row["id"]: row["split"] for row in rows if row.get("split")}
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row.get("split"):
            by_category[row["category"]].append(row)
    for category in sorted(by_category):
        pending = sorted(by_category[category], key=lambda r: r["id"])
        random.Random(f"{seed}:{category}").shuffle(pending)
        dev = sum(1 for r in rows if r["category"] == category and result.get(r["id"]) == "dev")
        test = sum(1 for r in rows if r["category"] == category and result.get(r["id"]) == "test")
        for row in pending:
            want_dev = dev / (dev + test) < ratio_dev if dev + test else True
            result[row["id"]] = "dev" if want_dev else "test"
            dev, test = dev + want_dev, test + (not want_dev)
    return result
