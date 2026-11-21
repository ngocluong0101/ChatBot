"""Kiểm tra LLM-judge (chấm "trung thực") có đồng ý với người không.

    python -m evals.judge_check build --n 20 --perturbed 10   # chạy pipeline + judge, ghi judge_check.jsonl
    python -m evals.judge_check label                          # bạn chấm tay (f = trung thực, u = không)
    python -m evals.judge_check score                          # độ đồng thuận judge–người (agreement, kappa)

Câu trả lời của mô hình hầu như luôn trung thực với ngữ cảnh, nên nếu chỉ chấm câu thật thì judge trả lời
"trung thực" cho tất cả và độ đồng thuận cao một cách vô nghĩa. Vì vậy một phần các câu được cố ý làm sai:
đổi một con số trong câu trả lời (`origin = "perturbed"`). Người chấm không thấy câu nào bị đổi (origin nằm ở
file riêng `judge_check.origin.json`), judge chấm cả hai loại. Cần ~30 câu có nhãn của người mới đáng tin.
"""

import argparse
import json
import random
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from evals.common import EVALS_DIR, read_jsonl, write_jsonl

CHECK_FILE = EVALS_DIR / "judge_check.jsonl"
ORIGIN_FILE = EVALS_DIR / "judge_check.origin.json"
CONTEXT_SEPARATOR = "\n\n---\n\n"
SHOW_CONTEXT_CHARS = 4000
JudgeFn = Callable[[str, str, str], bool | None]


def perturb_answer(answer: str, rng: random.Random) -> str | None:
    """Đổi một con số trong câu trả lời thành số khác (giữ độ rộng "05" -> "07"). None nếu không có số."""
    numbers = list(re.finditer(r"\d+", answer))
    if not numbers:
        return None
    match = rng.choice(numbers)
    value = int(match.group())
    changed = value + rng.choice([1, 2, 3, 5, 10]) if value < 100 else value * 2
    return answer[: match.start()] + str(changed).zfill(len(match.group())) + answer[match.end() :]


def cohen_kappa(pairs: Sequence[tuple[bool, bool]]) -> float | None:
    """Kappa giữa hai người chấm nhị phân. None khi không xác định (thiếu dữ liệu/không biến thiên)."""
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    p_a = sum(a for a, _ in pairs) / n
    p_b = sum(b for _, b in pairs) / n
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def _agreement(pairs: Sequence[tuple[bool, bool]]) -> dict[str, Any]:
    return {
        "n": len(pairs),
        "agreement": sum(a == b for a, b in pairs) / len(pairs) if pairs else None,
        "kappa": cohen_kappa(pairs),
    }


def score_rows(rows: Sequence[dict[str, Any]], origins: dict[str, str]) -> dict[str, Any]:
    """So nhãn judge với nhãn người trên các câu đã được chấm tay (human không null, judge không null)."""
    labeled = [r for r in rows if r.get("human") is not None]
    usable = [r for r in labeled if r.get("judge") is not None]
    pairs = [(bool(r["judge"]), bool(r["human"])) for r in usable]
    human_bad = [r for r in usable if not r["human"]]
    human_good = [r for r in usable if r["human"]]
    result = {
        "rows": len(rows),
        "human_labeled": len(labeled),
        "judge_unparsable": len(labeled) - len(usable),
        **_agreement(pairs),
        # Judge có bắt được câu sai không (quan trọng hơn tỉ lệ đồng thuận chung)?
        "catches_unfaithful": sum(not r["judge"] for r in human_bad) / len(human_bad) if human_bad else None,
        "accepts_faithful": sum(bool(r["judge"]) for r in human_good) / len(human_good)
        if human_good
        else None,
        "human_unfaithful": len(human_bad),
        "by_origin": {},
    }
    for origin in sorted({origins.get(r["id"], "?") for r in usable}):
        part = [(bool(r["judge"]), bool(r["human"])) for r in usable if origins.get(r["id"], "?") == origin]
        result["by_origin"][origin] = _agreement(part)
    return result


def format_score(result: dict[str, Any]) -> str:
    def f(value: Any) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    lines = [
        f"Số câu đã chấm tay      : {result['human_labeled']}/{result['rows']}",
        f"Judge không đọc được    : {result['judge_unparsable']}",
        f"Đồng thuận (agreement)  : {f(result['agreement'])}",
        f"Cohen's kappa           : {f(result['kappa'])}",
        f"Judge bắt được câu sai  : {f(result['catches_unfaithful'])}"
        f"  (trên {result['human_unfaithful']} câu người cho là sai)",
        f"Judge nhận câu đúng     : {f(result['accepts_faithful'])}",
    ]
    for origin, part in result["by_origin"].items():
        lines.append(
            f"  {origin:<10} n={part['n']:<3} agreement={f(part['agreement'])} kappa={f(part['kappa'])}"
        )
    if result["human_labeled"] < 30:
        lines.append("\n⚠ Dưới 30 câu có nhãn: kết quả chưa đáng tin để kết luận.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- build


def build_rows(
    pipeline: Any,
    judge: JudgeFn,
    examples: Sequence[Any],
    n_real: int,
    n_perturbed: int,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Chạy pipeline trên câu có đáp án, lấy câu trả lời không bị từ chối; một phần được đổi số cho sai."""
    rng = random.Random(seed)
    pool = [e for e in examples if e.answerable]
    rng.shuffle(pool)
    collected: list[tuple[Any, Any]] = []
    for example in pool:
        if len(collected) >= n_real + n_perturbed * 2:  # dư để bỏ qua các câu không có số để đổi
            break
        answer = pipeline.ask(example.question)
        if not answer.refused and answer.contexts:
            collected.append((example, answer))

    candidates: list[tuple[Any, Any, str, str]] = [(e, a, a.text, "model") for e, a in collected[:n_real]]
    for example, answer in collected[n_real:]:
        changed = perturb_answer(answer.text, rng)
        if changed is not None and sum(c[3] == "perturbed" for c in candidates) < n_perturbed:
            candidates.append((example, answer, changed, "perturbed"))

    rows = []
    for example, answer, text, origin in candidates:
        context = CONTEXT_SEPARATOR.join(answer.contexts)
        judged = judge(example.question, context, text)
        rows.append(
            {"question": example.question, "context": context, "answer": text, "judge": judged, "human": None}
            | {"_origin": origin}
        )
    rng.shuffle(rows)  # người chấm không đoán được câu nào bị đổi số qua thứ tự
    origins: dict[str, str] = {}
    for number, row in enumerate(rows, start=1):
        row["id"] = f"jc-{number:02d}"
        origins[row["id"]] = row.pop("_origin")
    ordered = [{k: row[k] for k in ("id", "question", "context", "answer", "judge", "human")} for row in rows]
    return ordered, origins


def label_loop(
    rows: list[dict[str, Any]],
    path: Path,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> bool:
    """Chấm tay các câu chưa có nhãn. Ghi sau mỗi câu. True nếu chấm xong hết."""
    todo = [r for r in rows if r.get("human") is None]
    for index, row in enumerate(todo, start=1):
        print_fn("=" * 78)
        print_fn(f"({index}/{len(todo)})  {row['id']}")
        print_fn(f"CÂU HỎI: {row['question']}\n")
        print_fn("NGỮ CẢNH:\n" + row["context"][:SHOW_CONTEXT_CHARS])
        print_fn(f"\nCÂU TRẢ LỜI CẦN CHẤM: {row['answer']}")
        print_fn("Trung thực = MỌI thông tin trong câu trả lời đều được ngữ cảnh ủng hộ.")
        while True:
            choice = (
                input_fn("[f] trung thực  [u] không trung thực  [s] bỏ qua  [q] thoát > ").strip().lower()
            )
            if choice in ("f", "u", "s", "q"):
                break
        if choice == "q":
            return False
        if choice in ("f", "u"):
            row["human"] = choice == "f"
            write_jsonl(path, rows)
    return all(r.get("human") is not None for r in rows)


# --------------------------------------------------------------------------- CLI


def _build(args: argparse.Namespace) -> int:
    from ragbot.cli import configure_logging
    from ragbot.config import Settings, parse_overrides
    from ragbot.evaluate import default_split, load_dataset, make_judge, select_examples
    from ragbot.pipeline import build_pipeline

    overrides = {"docs_dir": "data/corpus", "pdf_loader": "pdfplumber", "chunking": "legal"}
    overrides |= {"index_dir": ".index-legal"} | parse_overrides(args.overrides)
    settings = Settings(**overrides)
    configure_logging(settings.log_level)
    examples = load_dataset(args.dataset)
    examples = select_examples(examples, args.split or default_split(examples), args.include_unverified)
    pipeline = build_pipeline(settings)
    judge = make_judge(settings, args.judge_model or settings.llm_model)
    rows, origins = build_rows(pipeline, judge, examples, args.n, args.perturbed, args.seed)
    write_jsonl(args.out, rows)
    args.origin_out.write_text(json.dumps(origins, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Đã ghi {len(rows)} câu vào {args.out}. Chấm tay: python -m evals.judge_check label")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kiểm tra LLM-judge so với người chấm")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "label", "score"):
        p = sub.add_parser(name)
        p.add_argument("--out" if name == "build" else "--file", dest="out", type=Path, default=CHECK_FILE)
        p.add_argument("--origin-out", dest="origin_out", type=Path, default=ORIGIN_FILE)
        if name == "build":
            p.add_argument("--dataset", type=Path, default=EVALS_DIR / "dataset_legal.jsonl")
            p.add_argument("--split", choices=("dev", "test", "all"), default="dev")
            p.add_argument("--n", type=int, default=20, help="số câu trả lời thật của mô hình")
            p.add_argument("--perturbed", type=int, default=10, help="số câu bị đổi số cho sai")
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--judge-model")
            p.add_argument("--include-unverified", action="store_true")
            p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    if args.command == "build":
        return _build(args)
    rows = read_jsonl(args.out)
    if not rows:
        print(f"Chưa có {args.out}: chạy `build` trước.", file=sys.stderr)
        return 2
    if args.command == "label":
        done = label_loop(rows, args.out)
        print("Chấm xong." if done else "Còn câu chưa chấm; chạy lại để tiếp tục.")
        return 0
    origins = json.loads(args.origin_out.read_text(encoding="utf-8")) if args.origin_out.is_file() else {}
    print(format_score(score_rows(rows, origins)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
