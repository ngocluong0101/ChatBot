"""Load test cho API chatbot: N người dùng đồng thời, mỗi người hỏi liên tiếp (closed loop, không nghỉ).

    # chạy API ở terminal khác (uvicorn ragbot.api:app --port 8000), rồi:
    python -m evals.perf.load_test --label baseline --users 1,5,10,20 \
        --out evals/reports/final-perf-baseline.json

Đo qua HTTP đúng như người dùng thật (endpoint /chat/stream để có thời gian-tới-token-đầu, TTFT):
  * độ trễ trọn yêu cầu và TTFT: p50 / p95 / p99, trung bình, lớn nhất
  * thông lượng (yêu cầu hoàn tất / giây), tỉ lệ lỗi và lý do
  * VRAM và mức dùng GPU lấy mẫu mỗi giây bằng `nvidia-smi` trong lúc chạy

Mọi mức người dùng dùng CÙNG một dãy câu hỏi (cố định theo seed) từ evals/perf/questions.jsonl,
nên các cấu hình so sánh được với nhau. Số yêu cầu mỗi người:
max(`--min-per-user`, ceil(`--min-total` / người dùng)).
Bộ câu hỏi lấy mẫu từ bản nháp (chưa duyệt) vì ở đây chỉ đo thời gian, không đo độ đúng.
"""

import argparse
import json
import math
import platform
import random
import statistics
import subprocess
import threading
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

PERF_DIR = Path(__file__).parent
DEFAULT_QUESTIONS = PERF_DIR / "questions.jsonl"


def percentile(values: Sequence[float], q: float) -> float | None:
    """Phân vị q (0-100) nội suy tuyến tính; None nếu không có mẫu."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize_times(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    r = lambda x: None if x is None else round(x, 3)  # noqa: E731
    return {
        "n": len(values),
        "mean": r(statistics.fmean(values)),
        "p50": r(percentile(values, 50)),
        "p95": r(percentile(values, 95)),
        "p99": r(percentile(values, 99)),
        "max": r(max(values)),
    }


def load_questions(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def plan_requests(
    questions: Sequence[dict[str, Any]], users: int, per_user: int, seed: int
) -> list[list[dict[str, Any]]]:
    """Dãy câu hỏi của từng người dùng; xác định theo (seed, số người dùng) nên chạy lại ra đúng dãy đó."""
    rng = random.Random(f"{seed}:{users}")
    return [[rng.choice(questions) for _ in range(per_user)] for _ in range(users)]


def parse_sse_line(line: str) -> tuple[str | None, str | None]:
    if line.startswith("event: "):
        return "event", line[7:]
    if line.startswith("data: "):
        return "data", line[6:]
    return None, None


def one_request(client: httpx.Client, url: str, question: str, headers: dict[str, str]) -> dict[str, Any]:
    """Một lượt hỏi qua /chat/stream; trả về kết quả đo, không bao giờ ném lỗi."""
    started = time.perf_counter()
    result: dict[str, Any] = {"ok": False, "ttft": None, "tokens": 0, "refused": None, "error": None}
    try:
        with client.stream(
            "POST", url.rstrip("/") + "/chat/stream", json={"question": question}, headers=headers
        ) as response:
            if response.status_code != 200:
                result["error"] = f"http_{response.status_code}"
                result["retry_after"] = response.headers.get("retry-after")
            else:
                event = None
                for line in response.iter_lines():
                    kind, value = parse_sse_line(line)
                    if kind == "event":
                        event = value
                    elif kind == "data":
                        if event == "token":
                            result["tokens"] += 1
                            if result["ttft"] is None:
                                result["ttft"] = time.perf_counter() - started
                        elif event == "done":
                            result["ok"] = True
                            result["refused"] = bool(json.loads(value).get("refused"))
                        elif event == "error":
                            result["error"] = "sse_error"
    except httpx.TimeoutException:
        result["error"] = "timeout"
    except httpx.HTTPError as exc:
        result["error"] = type(exc).__name__
    result["seconds"] = time.perf_counter() - started
    if result["ok"] and result["ttft"] is None:
        result["ttft"] = result["seconds"]  # lời từ chối không có token: người dùng thấy câu trả lời khi xong
    return result


class GpuSampler:
    """Lấy mẫu `nvidia-smi` mỗi `interval` giây trong lúc chạy; không có GPU thì không ghi gì."""

    QUERY = "memory.used,memory.total,utilization.gpu"

    def __init__(self, interval: float = 1.0) -> None:
        self.interval = interval
        self.samples: list[tuple[float, float, float]] = []
        self.total_mib: float | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @classmethod
    def read(cls) -> tuple[float, float, float] | None:
        try:
            out = subprocess.run(
                ["nvidia-smi", f"--query-gpu={cls.QUERY}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.splitlines()[0]
            used, total, util = (float(x) for x in out.split(","))
            return used, total, util
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return None

    def _run(self) -> None:
        while not self._stop.is_set():
            reading = self.read()
            if reading:
                self.total_mib = reading[1]
                self.samples.append((reading[0], reading[2], time.time()))
            self._stop.wait(self.interval)

    def __enter__(self) -> "GpuSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def summary(self) -> dict[str, Any] | None:
        if not self.samples:
            return None
        used = [s[0] for s in self.samples]
        util = [s[1] for s in self.samples]
        return {
            "vram_peak_mib": round(max(used)),
            "vram_mean_mib": round(statistics.fmean(used)),
            "vram_total_mib": round(self.total_mib) if self.total_mib else None,
            "gpu_util_mean_pct": round(statistics.fmean(util), 1),
            "gpu_util_peak_pct": round(max(util)),
            "samples": len(used),
        }


def run_level(
    url: str,
    users: int,
    plan: list[list[dict[str, Any]]],
    timeout: float,
    headers: dict[str, str],
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    lock = threading.Lock()
    barrier = threading.Barrier(users)

    def user(index: int) -> None:
        with httpx.Client(timeout=httpx.Timeout(timeout, connect=10)) as client:
            barrier.wait()
            for item in plan[index]:
                res = one_request(client, url, item["question"], headers)
                res["category"] = item["category"]
                with lock:
                    results.append(res)

    with GpuSampler() as gpu:
        started = time.perf_counter()
        threads = [threading.Thread(target=user, args=(i,)) for i in range(users)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        wall = time.perf_counter() - started

    good = [r for r in results if r["ok"]]
    errors: dict[str, int] = {}
    for r in results:
        if r["error"]:
            errors[r["error"]] = errors.get(r["error"], 0) + 1
    answered = [r for r in good if not r["refused"]]
    token_rates = [
        r["tokens"] / (r["seconds"] - r["ttft"]) for r in answered if r["seconds"] - r["ttft"] > 0.2
    ]
    return {
        "users": users,
        "requests": len(results),
        "ok": len(good),
        "errors": errors,
        "error_rate": round(1 - len(good) / len(results), 4) if results else None,
        "wall_seconds": round(wall, 2),
        "throughput_rps": round(len(good) / wall, 3) if wall else None,
        "latency_seconds": summarize_times([r["seconds"] for r in good]),
        "ttft_seconds": summarize_times([r["ttft"] for r in good]),
        "latency_seconds_answered": summarize_times([r["seconds"] for r in answered]),
        "refused": len(good) - len(answered),
        "stream_events_per_second_per_user": round(statistics.fmean(token_rates), 1) if token_rates else None,
        "gpu": gpu.summary(),
    }


def run_load_test(
    url: str,
    levels: Sequence[int],
    questions: Sequence[dict[str, Any]],
    *,
    min_per_user: int = 3,
    min_total: int = 20,
    seed: int = 7,
    timeout: float = 600.0,
    api_key: str = "",
    warmup: int = 2,
) -> dict[str, Any]:
    headers = {"X-API-Key": api_key} if api_key else {}
    idle = GpuSampler.read()
    if warmup:  # nạp model vào VRAM, tránh tính thời gian khởi động nguội vào mức người dùng đầu tiên
        with httpx.Client(timeout=timeout) as client:
            for item in plan_requests(questions, 1, warmup, seed - 1)[0]:
                one_request(client, url, item["question"], headers)
    out_levels = []
    for users in levels:
        per_user = max(min_per_user, math.ceil(min_total / users))
        plan = plan_requests(questions, users, per_user, seed)
        out_levels.append({**run_level(url, users, plan, timeout, headers), "requests_per_user": per_user})
        print(
            f"  {users:>2} người dùng: p50 {out_levels[-1]['latency_seconds']['p50']}s, "
            f"p95 {out_levels[-1]['latency_seconds']['p95']}s, lỗi {out_levels[-1]['errors'] or 0}, "
            f"{out_levels[-1]['throughput_rps']} yêu cầu/s"
        )
    return {
        "levels": out_levels,
        "gpu_idle_before_mib": idle[0] if idle else None,
        "questions": {
            "n": len(questions),
            "seed": seed,
            "min_per_user": min_per_user,
            "min_total": min_total,
        },
    }


def server_info(url: str) -> dict[str, Any]:
    try:
        return httpx.get(url.rstrip("/") + "/health", timeout=10).json()
    except (httpx.HTTPError, ValueError):
        return {}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load test API chatbot")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--users", default="1,5,10,20", help="các mức số người dùng, cách nhau bằng dấu phẩy")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--min-per-user", type=int, default=3)
    parser.add_argument("--min-total", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=600.0, help="giây chờ tối đa cho MỘT yêu cầu")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--warmup", type=int, default=2, help="số yêu cầu khởi động trước khi đo")
    parser.add_argument("--label", required=True, help="tên cấu hình, ví dụ baseline hoặc parallel-4")
    parser.add_argument("--note", default="", help="ghi chú cấu hình server (biến môi trường, phiên bản...)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    levels = [int(x) for x in args.users.split(",")]
    print(f"Load test '{args.label}' trên {args.url}, mức người dùng {levels}")
    result = run_load_test(
        args.url,
        levels,
        load_questions(args.questions),
        min_per_user=args.min_per_user,
        min_total=args.min_total,
        seed=args.seed,
        timeout=args.timeout,
        api_key=args.api_key,
        warmup=args.warmup,
    )
    result["meta"] = {
        "label": args.label,
        "note": args.note,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "url": args.url,
        "server": server_info(args.url),
        "machine": platform.processor() or platform.machine(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Đã ghi {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
