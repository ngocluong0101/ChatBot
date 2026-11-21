"""Bảng hiệu năng markdown từ các báo cáo load test (mọi số lấy từ file báo cáo).

python -m evals.perf.summarize evals/reports/final-perf-*.json --readme README.md
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from evals.summarize import inject

ORDER = ("baseline", "serial-1", "parallel-2", "parallel-4", "probe-fp16", "probe-cpu", "probe-embcpu")
PROBE_LABELS = {
    "probe-fp16": "embedding GPU fp16, reranker GPU",
    "probe-cpu": "embedding CPU, reranker CPU",
    "probe-embcpu": "embedding CPU, reranker GPU",
}


def _fmt(value: float | None, digits: int = 1) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def _errors(level: dict[str, Any]) -> str:
    errors = level.get("errors") or {}
    return "0" if not errors else ", ".join(f"{n} {k}" for k, n in sorted(errors.items()))


def sort_key(report: dict[str, Any]) -> tuple[int, str]:
    label = report["meta"]["label"]
    return (ORDER.index(label) if label in ORDER else len(ORDER), label)


def perf_table(reports: Sequence[dict[str, Any]], labels: dict[str, str] | None = None) -> str:
    """Mỗi dòng một (cấu hình, số người dùng). Thông lượng = yêu cầu thành công / giây tường."""
    labels = labels or {}
    lines = [
        "| Cấu hình | Người dùng | Yêu cầu | Thông lượng (yêu cầu/s) | Độ trễ p50 (s) | p95 (s) "
        "| TTFT p50 (s) | Lỗi | VRAM đỉnh (MiB) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for report in sorted(reports, key=sort_key):
        label = report["meta"]["label"]
        for level in report["levels"]:
            lat, ttft, gpu = level["latency_seconds"], level["ttft_seconds"], level.get("gpu") or {}
            lines.append(
                f"| {labels.get(label, label)} | {level['users']} | {level['requests']} "
                f"| {_fmt(level['throughput_rps'], 2)} | {_fmt(lat['p50'])} | {_fmt(lat['p95'])} "
                f"| {_fmt(ttft['p50'])} | {_errors(level)} | {gpu.get('vram_peak_mib', '–')} |"
            )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bảng hiệu năng từ báo cáo load test")
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--readme", type=Path, help="điền bảng vào README (giữa các dấu <!-- perf:start -->)")
    parser.add_argument(
        "--probe",
        nargs="*",
        type=Path,
        default=[],
        help="báo cáo thăm dò thiết bị (một người dùng) -> perfprobe",
    )
    args = parser.parse_args(argv)
    missing = [p for p in args.reports if not p.is_file()]
    if missing:
        print(f"Không thấy: {', '.join(map(str, missing))}", file=sys.stderr)
        return 2
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in args.reports]
    table = perf_table(reports)
    print(table)
    if args.readme:
        text = inject(args.readme.read_text(encoding="utf-8"), "perf", table)
        if args.probe:
            probes = [json.loads(p.read_text(encoding="utf-8")) for p in args.probe]
            text = inject(text, "perfprobe", perf_table(probes, PROBE_LABELS))
        args.readme.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
