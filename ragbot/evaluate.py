"""Đánh giá RAG trên bộ câu hỏi `evals/dataset.jsonl`.

Ba nhóm chỉ số:
  * Retrieval  : hit@k, MRR, recall nguồn  (chỉ cần embedding, không cần Ollama)
  * Ngưỡng     : quét score_threshold để thấy ngưỡng nào tách được câu có/không có đáp án
  * Generation : đúng/sai theo từ khóa bắt buộc, từ chối đúng, citation đúng, latency,
                 và (tùy chọn) faithfulness do LLM-judge chấm

Cách chạy:
    python -m ragbot.evaluate --mode retrieval
    python -m ragbot.evaluate --mode full --out evals/reports/run.json
    python -m ragbot.evaluate --mode full --set llm_model=qwen2.5:7b-instruct --set top_k=3
    python -m ragbot.evaluate --mode full --judge-model qwen2.5:7b-instruct
"""

import argparse
import json
import math
import re
import statistics
import sys
import time
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from ragbot.cli import configure_logging
from ragbot.config import PROJECT_ROOT, Settings
from ragbot.errors import RagError
from ragbot.indexing import load_or_build_index
from ragbot.pipeline import RagPipeline, build_pipeline

DEFAULT_DATASET = PROJECT_ROOT / "evals" / "dataset.jsonl"
REPORT_SETTINGS_KEYS = (
    "llm_model",
    "embedding_model",
    "chunk_size",
    "chunk_overlap",
    "top_k",
    "score_threshold",
    "source_score_margin",
    "num_ctx",
)

JUDGE_PROMPT = (
    "Bạn là giám khảo kiểm tra tính trung thực của câu trả lời.\n"
    "Bạn được cho CÂU HỎI, NGỮ CẢNH và CÂU TRẢ LỜI cho câu hỏi đó. Chỉ trả về JSON dạng "
    '{{"faithful": true}} hoặc {{"faithful": false}}.\n'
    "faithful=true nếu MỌI thông tin trong câu trả lời đều được NGỮ CẢNH hỗ trợ "
    "(câu trả lời ngắn gọn vẫn hợp lệ nếu đúng với ngữ cảnh).\n"
    "faithful=false nếu câu trả lời có thông tin không có trong ngữ cảnh hoặc mâu thuẫn với ngữ cảnh.\n\n"
    "CÂU HỎI:\n{question}\n\n"
    "NGỮ CẢNH:\n{context}\n\n"
    "CÂU TRẢ LỜI:\n{answer}\n"
)


# --------------------------------------------------------------------------- dataset


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    category: str
    answerable: bool
    gold_sources: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    must_include: tuple[tuple[str, ...], ...] = ()


def load_dataset(path: Path = DEFAULT_DATASET) -> list[Example]:
    examples: list[Example] = []
    seen: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            example = Example(
                id=raw["id"],
                question=raw["question"],
                category=raw["category"],
                answerable=raw["answerable"],
                gold_sources=tuple(raw.get("gold_sources", ())),
                evidence=tuple(raw.get("evidence", ())),
                must_include=tuple(tuple(group) for group in raw.get("must_include", ())),
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{path}:{line_no}: dòng dataset không hợp lệ ({exc!r})") from exc
        if example.id in seen:
            raise ValueError(f"{path}:{line_no}: id trùng lặp: {example.id}")
        seen.add(example.id)
        if example.answerable and not (example.gold_sources and example.evidence and example.must_include):
            raise ValueError(f"{example.id}: câu có đáp án phải có gold_sources, evidence, must_include")
        examples.append(example)
    return examples


# --------------------------------------------------------------------------- matching


def normalize(text: str) -> str:
    """NFC + casefold + gộp khoảng trắng: PDF hay ngắt dòng/ngắt đoạn giữa câu."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip().casefold()


def contains_term(normalized_text: str, term: str) -> bool:
    """Khớp theo ranh giới từ, để "7 ngày" không khớp nhầm trong "17 ngày"."""
    pattern = r"(?<!\w)" + re.escape(normalize(term)) + r"(?!\w)"
    return re.search(pattern, normalized_text) is not None


def answer_is_correct(answer: str, must_include: Sequence[Sequence[str]]) -> bool:
    """Đúng khi MỖI nhóm có ít nhất một cách diễn đạt xuất hiện trong câu trả lời."""
    text = normalize(answer)
    return all(any(contains_term(text, alt) for alt in group) for group in must_include)


def is_relevant_chunk(doc: Document, example: Example) -> bool:
    """Chunk liên quan = đúng file gold và chứa ít nhất một đoạn bằng chứng."""
    if doc.metadata.get("source") not in example.gold_sources:
        return False
    content = normalize(doc.page_content)
    return any(normalize(snippet) in content for snippet in example.evidence)


def find_unsupported_examples(examples: Sequence[Example], chunks: Sequence[Document]) -> list[str]:
    """Id của câu có đáp án nhưng không chunk nào chứa bằng chứng (dataset sai hoặc chunking làm vỡ)."""
    return [e.id for e in examples if e.answerable and not any(is_relevant_chunk(c, e) for c in chunks)]


# --------------------------------------------------------------------------- stats helpers


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def percentile(values: Sequence[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(pct / 100 * len(ordered)) - 1)]


def _round(value: Any, digits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {k: _round(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [_round(v, digits) for v in value]
    return value


# --------------------------------------------------------------------------- retrieval


def evaluate_retrieval(
    pipeline: RagPipeline, examples: Sequence[Example], settings: Settings
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for ex in examples:
        results = pipeline.search(ex.question)
        record: dict[str, Any] = {
            "id": ex.id,
            "category": ex.category,
            "answerable": ex.answerable,
            "top1_score": results[0][1] if results else None,
        }
        if ex.answerable:
            flags = [is_relevant_chunk(doc, ex) for doc, _ in results]
            kept = [
                flag
                for (_, score), flag in zip(results, flags, strict=True)
                if score >= settings.score_threshold
            ]
            found = {doc.metadata.get("source") for doc, _ in results}
            record.update(
                first_rank=flags.index(True) + 1 if True in flags else None,
                source_recall=len(found & set(ex.gold_sources)) / len(ex.gold_sources),
                context_precision=ratio(sum(kept), len(kept)),
                retrieved_sources=[doc.metadata.get("source") for doc, _ in results],
            )
        records.append(record)
    return records, summarize_retrieval(records, settings.top_k)


def summarize_retrieval(records: Sequence[dict[str, Any]], top_k: int) -> dict[str, Any]:
    answerable = [r for r in records if r["answerable"]]

    def hit_at(rs: Sequence[dict[str, Any]], k: int) -> float | None:
        return ratio(sum(1 for r in rs if r["first_rank"] is not None and r["first_rank"] <= k), len(rs))

    def mrr(rs: Sequence[dict[str, Any]]) -> float | None:
        return mean([1 / r["first_rank"] if r["first_rank"] else 0.0 for r in rs])

    by_category: dict[str, Any] = {}
    for category in sorted({r["category"] for r in answerable}):
        rs = [r for r in answerable if r["category"] == category]
        by_category[category] = {"n": len(rs), f"hit@{top_k}": hit_at(rs, top_k), "mrr": mrr(rs)}

    precisions = [r["context_precision"] for r in answerable if r["context_precision"] is not None]
    return {
        "n_answerable": len(answerable),
        "hit@1": hit_at(answerable, 1),
        "hit@3": hit_at(answerable, 3),
        f"hit@{top_k}": hit_at(answerable, top_k),
        "mrr": mrr(answerable),
        f"source_recall@{top_k}": mean([r["source_recall"] for r in answerable]),
        "context_precision": mean(precisions),
        "by_category": by_category,
    }


def sweep_thresholds(
    records: Sequence[dict[str, Any]], start: float = 0.30, stop: float = 0.80, step: float = 0.02
) -> dict[str, Any]:
    """Với mỗi ngưỡng: bao nhiêu câu có đáp án còn giữ được chunk, bao nhiêu câu không có đáp án bị chặn.

    Quyết định "có chunk nào đạt ngưỡng" chỉ phụ thuộc điểm top-1. Đây là cổng chặn TRƯỚC khi gọi LLM.
    """
    answerable = [r["top1_score"] for r in records if r["answerable"] and r["top1_score"] is not None]
    unanswerable = [r["top1_score"] for r in records if not r["answerable"] and r["top1_score"] is not None]
    rows = []
    steps = round((stop - start) / step) + 1
    for i in range(steps):
        threshold = round(start + i * step, 4)
        kept = ratio(sum(s >= threshold for s in answerable), len(answerable))
        rejected = ratio(sum(s < threshold for s in unanswerable), len(unanswerable))
        balanced = None if kept is None or rejected is None else (kept + rejected) / 2
        rows.append(
            {
                "threshold": threshold,
                "answerable_kept": kept,
                "unanswerable_rejected": rejected,
                "balanced_accuracy": balanced,
            }
        )
    scored = [r for r in rows if r["balanced_accuracy"] is not None]
    best = None
    if scored:
        top = max(r["balanced_accuracy"] for r in scored)
        plateau = [r for r in scored if r["balanced_accuracy"] == top]
        best = plateau[len(plateau) // 2]  # giữa vùng bằng điểm, ổn định hơn mép
    return {
        "min_answerable_top1": min(answerable) if answerable else None,
        "max_unanswerable_top1": max(unanswerable) if unanswerable else None,
        "best": best,
        "rows": rows,
    }


# --------------------------------------------------------------------------- generation

JudgeFn = Callable[[str, str, str], bool | None]  # (câu hỏi, ngữ cảnh, câu trả lời) -> trung thực?


def make_judge(settings: Settings, model: str) -> JudgeFn:
    llm = ChatOllama(
        model=model,
        temperature=0,
        num_ctx=settings.num_ctx,
        base_url=settings.ollama_base_url,
        format="json",
    )
    chain = ChatPromptTemplate.from_template(JUDGE_PROMPT) | llm | StrOutputParser()

    def judge(question: str, context: str, answer: str) -> bool | None:
        raw = chain.invoke({"question": question, "context": context, "answer": answer})
        try:
            return bool(json.loads(raw)["faithful"])
        except (ValueError, KeyError, TypeError):
            return None

    return judge


def evaluate_generation(
    pipeline: RagPipeline, examples: Sequence[Example], judge: JudgeFn | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for ex in examples:
        started = time.perf_counter()
        record: dict[str, Any] = {"id": ex.id, "category": ex.category, "answerable": ex.answerable}
        try:
            answer = pipeline.ask(ex.question)
        except RagError as exc:
            record.update(error=str(exc), latency=time.perf_counter() - started)
            records.append(record)
            continue
        record.update(
            latency=time.perf_counter() - started,
            answer=answer.text,
            refused=answer.refused,
            sources=[s.file for s in answer.sources],
        )
        if ex.answerable:
            record["correct"] = (not answer.refused) and answer_is_correct(answer.text, ex.must_include)
            if not answer.refused:
                record["citation_ok"] = any(s.file in ex.gold_sources for s in answer.sources)
        else:
            record["correct_refusal"] = answer.refused
        if judge and not answer.refused and answer.contexts:
            record["faithful"] = judge(ex.question, "\n\n".join(answer.contexts), answer.text)
        records.append(record)
    return records, summarize_generation(records)


def summarize_generation(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if "error" not in r]
    answerable = [r for r in ok if r["answerable"]]
    unanswerable = [r for r in ok if not r["answerable"]]
    correct = sum(r["correct"] for r in answerable)
    refused_ok = sum(r["correct_refusal"] for r in unanswerable)
    answered = [r for r in answerable if not r["refused"]]
    judged = [r["faithful"] for r in ok if r.get("faithful") is not None]
    latencies = [r["latency"] for r in ok]

    by_category: dict[str, Any] = {}
    for category in sorted({r["category"] for r in ok}):
        rs = [r for r in ok if r["category"] == category]
        passed = sum(r.get("correct", False) or r.get("correct_refusal", False) for r in rs)
        by_category[category] = {"n": len(rs), "accuracy": ratio(passed, len(rs))}

    return {
        "n": len(records),
        "errors": len(records) - len(ok),
        "overall_accuracy": ratio(correct + refused_ok, len(ok)),
        "answerable_accuracy": ratio(correct, len(answerable)),
        "false_refusal_rate": ratio(sum(r["refused"] for r in answerable), len(answerable)),
        "refusal_accuracy": ratio(refused_ok, len(unanswerable)),
        "hallucination_rate": ratio(len(unanswerable) - refused_ok, len(unanswerable)),
        "citation_accuracy": ratio(sum(r["citation_ok"] for r in answered), len(answered)),
        "faithfulness": mean([float(v) for v in judged]),
        "faithfulness_judged": len(judged),
        "latency_mean_s": mean(latencies),
        "latency_p95_s": percentile(latencies, 95),
        "by_category": by_category,
    }


# --------------------------------------------------------------------------- report


def build_report(
    settings: Settings,
    mode: str,
    examples: Sequence[Example],
    retrieval: tuple[list[dict[str, Any]], dict[str, Any]],
    generation: tuple[list[dict[str, Any]], dict[str, Any]] | None,
    judge_model: str | None,
) -> dict[str, Any]:
    retrieval_records, retrieval_summary = retrieval
    report: dict[str, Any] = {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode": mode,
            "judge_model": judge_model,
            "settings": {k: getattr(settings, k) for k in REPORT_SETTINGS_KEYS},
            "dataset": {
                "total": len(examples),
                "answerable": sum(e.answerable for e in examples),
                "unanswerable": sum(not e.answerable for e in examples),
            },
        },
        "retrieval": retrieval_summary,
        "threshold_sweep": sweep_thresholds(retrieval_records),
        "retrieval_records": retrieval_records,
    }
    if generation:
        report["generation"] = generation[1]
        report["generation_records"] = generation[0]
    return _round(report)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def _failure_reason(record: dict[str, Any]) -> str | None:
    if "error" in record:
        return "lỗi: " + record["error"]
    if record.get("correct") is False:
        return "sai/từ chối nhầm"
    if record.get("correct_refusal") is False:
        return "không từ chối"
    if record.get("citation_ok") is False:
        return "citation sai"
    return None


def format_summary(report: dict[str, Any]) -> str:
    meta, ret, sweep = report["meta"], report["retrieval"], report["threshold_sweep"]
    top_k = meta["settings"]["top_k"]
    lines = ["== Cấu hình =="]
    lines += [f"  {k}: {v}" for k, v in meta["settings"].items()]
    d = meta["dataset"]
    lines.append(f"  dataset: {d['total']} câu ({d['answerable']} có đáp án, {d['unanswerable']} không có)")

    lines += ["", f"== Retrieval ({ret['n_answerable']} câu có đáp án) =="]
    for key in ("hit@1", "hit@3", f"hit@{top_k}", "mrr", f"source_recall@{top_k}", "context_precision"):
        lines.append(f"  {key:<22}{_fmt(ret.get(key))}")
    lines.append("  theo nhóm câu hỏi:")
    for cat, v in ret["by_category"].items():
        hit = _fmt(v[f"hit@{top_k}"])
        lines.append(f"    {cat:<14} n={v['n']:<3} hit@{top_k}={hit}  mrr={_fmt(v['mrr'])}")

    best = sweep["best"]
    lines += ["", "== Ngưỡng score_threshold =="]
    lines.append(f"  top-1 thấp nhất của câu có đáp án : {_fmt(sweep['min_answerable_top1'])}")
    lines.append(f"  top-1 cao nhất của câu không có    : {_fmt(sweep['max_unanswerable_top1'])}")
    if best:
        lines.append(
            f"  ngưỡng tốt nhất: {best['threshold']} (giữ {_fmt(best['answerable_kept'])} câu có đáp án, "
            f"chặn {_fmt(best['unanswerable_rejected'])} câu không có)"
        )
    lines.append(f"  ngưỡng đang dùng: {meta['settings']['score_threshold']}")

    gen = report.get("generation")
    if gen:
        lines += ["", f"== Generation ({gen['n']} câu, {gen['errors']} lỗi) =="]
        for key in (
            "overall_accuracy",
            "answerable_accuracy",
            "false_refusal_rate",
            "refusal_accuracy",
            "hallucination_rate",
            "citation_accuracy",
            "faithfulness",
            "latency_mean_s",
            "latency_p95_s",
        ):
            lines.append(f"  {key:<22}{_fmt(gen.get(key))}")
        lines.append("  theo nhóm câu hỏi:")
        for cat, v in gen["by_category"].items():
            lines.append(f"    {cat:<14} n={v['n']:<3} accuracy={_fmt(v['accuracy'])}")
        failures = [(r, _failure_reason(r)) for r in report["generation_records"]]
        failures = [(r, why) for r, why in failures if why]
        if failures:
            lines += ["", "== Các câu chưa đạt =="]
            for r, why in failures:
                answer = (r.get("answer") or "").replace("\n", " ")[:90]
                lines.append(f"  [{r['id']}] {why} | nguồn={r.get('sources')} | {answer}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI


def parse_overrides(pairs: Sequence[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or key not in Settings.model_fields:
            raise SystemExit(f"--set không hợp lệ: {pair!r}. Khóa hợp lệ: {', '.join(Settings.model_fields)}")
        overrides[key] = value
    return overrides


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Đánh giá RAG chatbot")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--mode", choices=("retrieval", "full"), default="full")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--judge-model", help="bật chấm faithfulness bằng LLM-judge (cần mode full)")
    parser.add_argument("--out", type=Path, help="ghi báo cáo JSON ra file này")
    args = parser.parse_args(argv)

    settings = Settings(**parse_overrides(args.overrides))
    configure_logging(settings.log_level)
    examples = load_dataset(args.dataset)

    if args.mode == "retrieval":
        pipeline = RagPipeline(settings, load_or_build_index(settings), chain=None)
    else:
        pipeline = build_pipeline(settings)

    unsupported = find_unsupported_examples(examples, pipeline.chunks())
    if unsupported:
        print(f"Dataset lỗi: không chunk nào chứa bằng chứng cho: {', '.join(unsupported)}", file=sys.stderr)
        return 2

    retrieval = evaluate_retrieval(pipeline, examples, settings)
    generation = None
    if args.mode == "full":
        judge = make_judge(settings, args.judge_model) if args.judge_model else None
        generation = evaluate_generation(pipeline, examples, judge)

    report = build_report(settings, args.mode, examples, retrieval, generation, args.judge_model)
    print(format_summary(report))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nĐã ghi báo cáo: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
