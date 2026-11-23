"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>

Đánh giá RAG trên bộ câu hỏi `evals/dataset_legal.jsonl`.

Ba nhóm chỉ số:
  * Retrieval  : hit@k, MRR, recall nguồn  (chỉ cần embedding, không cần Ollama)
  * Ngưỡng     : quét score_threshold để tìm ngưỡng tách câu có/không có đáp án
  * Generation : đúng/sai theo từ khóa bắt buộc, từ chối đúng, citation đúng, latency,
                 và (tùy chọn) faithfulness do LLM-judge chấm

Cách chạy:
    python -m ragbot.evaluate --mode retrieval
    python -m ragbot.evaluate --mode full --out evals/reports/run.json
    python -m ragbot.evaluate --mode full --set llm_model=qwen2.5:7b-instruct --set top_k=3
    python -m ragbot.evaluate --mode full --judge-model qwen2.5:7b-instruct

Với bộ chia dev/test: quét ngưỡng CHỈ trên dev (`--split dev --mode retrieval`), rồi chạy test bằng
`--split test --mode full --threshold-from <báo cáo dev>`. Chỉ câu verified=true được dùng
(`--include-unverified` chỉ để chạy thử).
"""

import argparse
import difflib
import json
import math
import random
import re
import statistics
import sys
import time
import unicodedata
from collections import Counter
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
from ragbot.config import PROJECT_ROOT, Settings, parse_overrides
from ragbot.errors import RagError
from ragbot.indexing import load_or_build_index
from ragbot.pipeline import (
    RagPipeline,
    build_pipeline,
    build_retriever,
    build_rewrite_chain,
    check_ollama,
)

DEFAULT_DATASET = PROJECT_ROOT / "evals" / "dataset_legal.jsonl"
REPORT_SETTINGS_KEYS = (
    "llm_model",
    "embedding_model",
    "chunking",
    "chunk_size",
    "chunk_overlap",
    "retrieval_mode",
    "vector_store",
    "pdf_loader",
    "rewrite_queries",
    "fetch_k",
    "rerank_model",
    "top_k",
    "score_threshold",
    "rerank_threshold",
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


SPLITS = ("dev", "test")
DIFFICULTIES = ("easy", "medium", "hard")


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    category: str
    answerable: bool
    gold_sources: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    must_include: tuple[tuple[str, ...], ...] = ()
    # Các trường dưới đây có sau bộ 53 câu đầu tiên; thiếu thì dùng mặc định để file cũ vẫn đọc được.
    split: str | None = None  # "dev" | "test"; None = chưa chia
    difficulty: str | None = None  # "easy" | "medium" | "hard"
    verified: bool = True  # True khi người đã duyệt tay; câu nháp của LLM luôn là False


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
                split=raw.get("split"),
                difficulty=raw.get("difficulty"),
                verified=raw.get("verified", True),
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{path}:{line_no}: dòng dataset không hợp lệ ({exc!r})") from exc
        if example.id in seen:
            raise ValueError(f"{path}:{line_no}: id trùng lặp: {example.id}")
        seen.add(example.id)
        if example.split not in (None, *SPLITS):
            raise ValueError(f"{example.id}: split phải là một trong {SPLITS}, nhận {example.split!r}")
        if example.difficulty not in (None, *DIFFICULTIES):
            raise ValueError(f"{example.id}: difficulty phải là một trong {DIFFICULTIES}")
        if not isinstance(example.verified, bool):
            raise ValueError(f"{example.id}: verified phải là true/false")
        if example.answerable and not (example.gold_sources and example.evidence and example.must_include):
            raise ValueError(f"{example.id}: câu có đáp án phải có gold_sources, evidence, must_include")
        examples.append(example)
    return examples


def default_split(examples: Sequence[Example]) -> str:
    """Bộ có chia dev/test thì mặc định chỉ chạy dev, để không vô tình "nhìn" tập test."""
    return "dev" if any(e.split for e in examples) else "all"


def select_examples(
    examples: Sequence[Example], split: str, include_unverified: bool = False, limit: int | None = None
) -> list[Example]:
    """Chỉ lấy câu đã duyệt tay (trừ khi chạy thử) và đúng split ("all" = không lọc).

    `limit`: lấy mẫu ngẫu nhiên (seed cố định) để chạy thử nhanh; không công bố số liệu.
    """
    chosen = [e for e in examples if include_unverified or e.verified]
    if split != "all":
        chosen = [e for e in chosen if e.split == split]
    if limit is not None and limit < len(chosen):
        keep = set(random.Random(0).sample(range(len(chosen)), limit))
        chosen = [e for i, e in enumerate(chosen) if i in keep]
    return chosen


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


MIN_PARTIAL_EVIDENCE = 40  # bằng chứng ngắn hơn mức này chỉ được khớp nguyên văn
PARTIAL_EVIDENCE_RATIO = 0.4  # bị cắt đúng giữa thì mỗi chunk giữ khoảng một nửa
MIN_PARTIAL_MATCH = 24  # ký tự khớp liền mạch tối thiểu
EDGE_TOLERANCE = 8  # ký tự


def _evidence_in(content: str, snippet: str) -> bool:
    """Chunk chứa bằng chứng: khớp nguyên văn, hoặc khớp phần lớn bằng chứng ở sát mép chunk.

    Chia theo độ dài có thể cắt ngang câu bằng chứng; không nới lỏng thì ablation "fixed" bị phạt oan.
    """
    if snippet in content:
        return True
    if len(snippet) < MIN_PARTIAL_EVIDENCE or not content:
        return False
    match = difflib.SequenceMatcher(None, content, snippet, autojunk=False).find_longest_match(
        0, len(content), 0, len(snippet)
    )
    if match.size < max(MIN_PARTIAL_MATCH, PARTIAL_EVIDENCE_RATIO * len(snippet)):
        return False
    touches_start = match.a <= EDGE_TOLERANCE
    touches_end = match.a + match.size >= len(content) - EDGE_TOLERANCE
    return touches_start or touches_end


def is_relevant_chunk(doc: Document, example: Example) -> bool:
    """Chunk liên quan = đúng file gold và chứa (hoặc chứa phần lớn, ở mép chunk) một đoạn bằng chứng."""
    if doc.metadata.get("source") not in example.gold_sources:
        return False
    content = normalize(doc.page_content)
    return any(_evidence_in(content, normalize(snippet)) for snippet in example.evidence)


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


def bootstrap_ci(
    values: Sequence[float], n_resamples: int = 2000, seed: int = 0, level: float = 0.95
) -> list[float] | None:
    """Khoảng tin cậy bootstrap (percentile) cho trung bình. Seed cố định để chạy lại ra đúng số cũ."""
    if not values:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples))
    low = means[int((1 - level) / 2 * n_resamples)]
    high = means[min(n_resamples - 1, int((1 + level) / 2 * n_resamples))]
    return [low, high]


def _round(value: Any, digits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {k: _round(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [_round(v, digits) for v in value]
    return value


# --------------------------------------------------------------------------- retrieval


def _meta_fields(ex: Example) -> dict[str, Any]:
    return {"split": ex.split, "difficulty": ex.difficulty}


def evaluate_retrieval(
    pipeline: RagPipeline, examples: Sequence[Example], settings: Settings
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for ex in examples:
        record: dict[str, Any] = {
            "id": ex.id,
            "category": ex.category,
            "answerable": ex.answerable,
            **_meta_fields(ex),
        }
        try:
            # Truy hồi bằng câu độc lập (đã viết lại nếu cần) như lúc chạy thật.
            question = pipeline.standalone_question(ex.question)
        except RagError as exc:
            record["error"] = str(exc)
            records.append(record)
            continue
        if question != ex.question:
            record["rewritten"] = question
        results = pipeline.search(question)
        # Điểm cao nhất trong các chunk trả về: quyết định "có chunk nào qua ngưỡng" chỉ cần nó.
        record["top1_score"] = max((score for _, score in results), default=None)
        if ex.answerable:
            flags = [is_relevant_chunk(doc, ex) for doc, _ in results]
            kept = [
                flag
                for (_, score), flag in zip(results, flags, strict=True)
                if score >= settings.active_threshold
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
    ok = [r for r in records if "error" not in r]
    answerable = [r for r in ok if r["answerable"]]

    def hit_at(rs: Sequence[dict[str, Any]], k: int) -> float | None:
        return ratio(sum(1 for r in rs if r["first_rank"] is not None and r["first_rank"] <= k), len(rs))

    def mrr(rs: Sequence[dict[str, Any]]) -> float | None:
        return mean([1 / r["first_rank"] if r["first_rank"] else 0.0 for r in rs])

    by_category: dict[str, Any] = {}
    for category in sorted({r["category"] for r in answerable}):
        rs = [r for r in answerable if r["category"] == category]
        by_category[category] = {"n": len(rs), f"hit@{top_k}": hit_at(rs, top_k), "mrr": mrr(rs)}
        by_category[category]["hit@1"] = hit_at(rs, 1)

    precisions = [r["context_precision"] for r in answerable if r["context_precision"] is not None]
    hit1 = [1.0 if r["first_rank"] == 1 else 0.0 for r in answerable]
    return {
        "n_answerable": len(answerable),
        "errors": len(records) - len(ok),
        "hit@1": hit_at(answerable, 1),
        "hit@1_ci95": bootstrap_ci(hit1),
        "hit@3": hit_at(answerable, 3),
        f"hit@{top_k}": hit_at(answerable, top_k),
        "mrr": mrr(answerable),
        f"source_recall@{top_k}": mean([r["source_recall"] for r in answerable]),
        "context_precision": mean(precisions),
        "by_category": by_category,
    }


def sweep_thresholds(
    records: Sequence[dict[str, Any]], start: float = 0.0, stop: float = 1.0, step: float = 0.02
) -> dict[str, Any]:
    """Với mỗi ngưỡng: số câu có đáp án còn giữ được chunk, số câu không có đáp án bị chặn.

    Cổng chặn TRƯỚC khi gọi LLM chỉ phụ thuộc điểm top-1.
    """
    scored = [r for r in records if r.get("top1_score") is not None]
    answerable = [r["top1_score"] for r in scored if r["answerable"]]
    unanswerable = [r["top1_score"] for r in scored if not r["answerable"]]
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
        record: dict[str, Any] = {
            "id": ex.id,
            "category": ex.category,
            "answerable": ex.answerable,
            **_meta_fields(ex),
        }
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
        "answerable_accuracy_ci95": bootstrap_ci([float(r["correct"]) for r in answerable]),
        "false_refusal_rate": ratio(sum(r["refused"] for r in answerable), len(answerable)),
        "refusal_accuracy": ratio(refused_ok, len(unanswerable)),
        "refusal_accuracy_ci95": bootstrap_ci([float(r["correct_refusal"]) for r in unanswerable]),
        "hallucination_rate": ratio(len(unanswerable) - refused_ok, len(unanswerable)),
        "citation_accuracy": ratio(sum(r["citation_ok"] for r in answered), len(answered)),
        "faithfulness": mean([float(v) for v in judged]),
        "faithfulness_judged": len(judged),
        "latency_mean_s": mean(latencies),
        "latency_p95_s": percentile(latencies, 95),
        "by_category": by_category,
    }


# --------------------------------------------------------------------------- report


def split_breakdown(
    retrieval_records: Sequence[dict[str, Any]],
    generation_records: Sequence[dict[str, Any]] | None,
    top_k: int,
) -> dict[str, Any] | None:
    """Kết quả tách theo split; None nếu chỉ có một split (không có gì để so sánh)."""
    splits = sorted({r["split"] for r in retrieval_records if r.get("split")})
    if len(splits) < 2:
        return None
    result: dict[str, Any] = {}
    for name in splits:
        part: dict[str, Any] = {
            "retrieval": summarize_retrieval([r for r in retrieval_records if r.get("split") == name], top_k)
        }
        if generation_records:
            part["generation"] = summarize_generation(
                [r for r in generation_records if r.get("split") == name]
            )
        result[name] = part
    return result


def build_report(
    settings: Settings,
    mode: str,
    examples: Sequence[Example],
    retrieval: tuple[list[dict[str, Any]], dict[str, Any]],
    generation: tuple[list[dict[str, Any]], dict[str, Any]] | None,
    judge_model: str | None,
    *,
    label: str | None = None,
    split: str = "all",
    include_unverified: bool = False,
    limit: int | None = None,
    threshold_selection: dict[str, Any] | None = None,
    dataset_name: str | None = None,
    unsupported: Sequence[str] = (),
) -> dict[str, Any]:
    retrieval_records, retrieval_summary = retrieval
    # Ngưỡng chỉ được chọn trên dev. Bộ không chia split (bộ cũ) thì quét trên toàn bộ như trước.
    has_split = any(r.get("split") for r in retrieval_records)
    sweep_records = (
        [r for r in retrieval_records if r.get("split") == "dev"] if has_split else retrieval_records
    )
    sweep = sweep_thresholds(sweep_records) if sweep_records else None
    report: dict[str, Any] = {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode": mode,
            "label": label,
            "split": split,
            "include_unverified": include_unverified,
            "limit": limit,
            "judge_model": judge_model,
            "settings": {k: getattr(settings, k) for k in REPORT_SETTINGS_KEYS},
            "corpus": settings.docs_dir.name,
            "active_threshold": settings.active_threshold,
            "threshold_selection": threshold_selection,
            "unsupported_examples": list(unsupported),
            "dataset": {
                "file": dataset_name,
                "total": len(examples),
                "answerable": sum(e.answerable for e in examples),
                "unanswerable": sum(not e.answerable for e in examples),
                "by_category": dict(sorted(Counter(e.category for e in examples).items())),
            },
        },
        "retrieval": retrieval_summary,
        "threshold_sweep": sweep,
        "retrieval_records": retrieval_records,
    }
    if generation:
        report["generation"] = generation[1]
        report["generation_records"] = generation[0]
    by_split = split_breakdown(retrieval_records, generation[0] if generation else None, settings.top_k)
    if by_split:
        report["by_split"] = by_split
    return _round(report)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def _fmt_ci(value: Any, ci: Sequence[float] | None) -> str:
    return _fmt(value) if not ci else f"{_fmt(value)}  (95% CI {ci[0]:.3f}–{ci[1]:.3f})"


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
    meta, ret, sweep = report["meta"], report["retrieval"], report.get("threshold_sweep")
    top_k = meta["settings"]["top_k"]
    lines = ["== Cấu hình =="]
    if meta.get("label"):
        lines.append(f"  label: {meta['label']}")
    lines.append(
        f"  split: {meta.get('split', 'all')}"
        + ("  (CÓ câu chưa duyệt tay)" if meta.get("include_unverified") else "")
    )
    lines += [f"  {k}: {v}" for k, v in meta["settings"].items()]
    d = meta["dataset"]
    lines.append(f"  dataset: {d['total']} câu ({d['answerable']} có đáp án, {d['unanswerable']} không có)")
    if meta.get("threshold_selection"):
        t = meta["threshold_selection"]
        lines.append(f"  ngưỡng chọn trên dev: {t['key']}={t['value']} (từ {t['from']})")

    lines += ["", f"== Retrieval ({ret['n_answerable']} câu có đáp án) =="]
    lines.append(f"  {'hit@1':<22}{_fmt_ci(ret.get('hit@1'), ret.get('hit@1_ci95'))}")
    for key in ("hit@3", f"hit@{top_k}", "mrr", f"source_recall@{top_k}", "context_precision"):
        lines.append(f"  {key:<22}{_fmt(ret.get(key))}")
    if ret.get("errors"):
        lines.append(f"  (bỏ qua {ret['errors']} câu lỗi khi viết lại câu hỏi)")
    lines.append("  theo nhóm câu hỏi:")
    for cat, v in ret["by_category"].items():
        hit = _fmt(v[f"hit@{top_k}"])
        lines.append(f"    {cat:<14} n={v['n']:<3} hit@{top_k}={hit}  mrr={_fmt(v['mrr'])}")

    lines += ["", "== Ngưỡng score_threshold =="]
    if sweep:
        best = sweep["best"]
        lines.append(f"  top-1 thấp nhất của câu có đáp án : {_fmt(sweep['min_answerable_top1'])}")
        lines.append(f"  top-1 cao nhất của câu không có    : {_fmt(sweep['max_unanswerable_top1'])}")
        if best:
            lines.append(
                f"  ngưỡng tốt nhất: {best['threshold']} (giữ {_fmt(best['answerable_kept'])} câu có đáp án, "
                f"chặn {_fmt(best['unanswerable_rejected'])} câu không có)"
            )
    else:
        lines.append("  (không quét ngưỡng: báo cáo này không có dev; ngưỡng đã chọn trước trên dev)")
    active = meta.get("active_threshold", meta["settings"].get("score_threshold"))
    lines.append(f"  ngưỡng đang dùng: {active}")

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
            lines.append(f"  {key:<22}{_fmt_ci(gen.get(key), gen.get(key + '_ci95'))}")
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
    for name, part in (report.get("by_split") or {}).items():
        r, g = part["retrieval"], part.get("generation") or {}
        lines.append(
            f"\n== Split {name} == hit@1={_fmt(r['hit@1'])} mrr={_fmt(r['mrr'])}"
            + (f" answerable_acc={_fmt(g.get('answerable_accuracy'))}" if g else "")
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- threshold protocol

THRESHOLD_KEYS = ("rerank_threshold", "score_threshold")


def threshold_from_report(path: Path, settings: Settings) -> tuple[str, float, dict[str, Any]]:
    """Ngưỡng tốt nhất do báo cáo DEV chọn, để áp nguyên xi cho test.

    Từ chối báo cáo không phải dev hoặc khác cấu hình reranker (hai thang điểm không mượn nhau được).
    """
    report = json.loads(path.read_text(encoding="utf-8"))
    meta = report["meta"]
    if meta.get("split") != "dev":
        raise SystemExit(
            f"--threshold-from cần báo cáo chạy với --split dev, {path} là split={meta.get('split')!r}"
        )
    best = (report.get("threshold_sweep") or {}).get("best")
    if not best:
        raise SystemExit(
            f"{path} không có ngưỡng tốt nhất (thiếu câu có đáp án hoặc không có đáp án trong dev)"
        )
    if bool(meta["settings"].get("rerank_model")) != bool(settings.rerank_model):
        raise SystemExit("Báo cáo dev và lần chạy này khác nhau về reranker: ngưỡng không dùng chung được")
    key = "rerank_threshold" if settings.rerank_model else "score_threshold"
    info = {
        "key": key,
        "value": best["threshold"],
        "from": path.name,
        "dev_answerable_kept": best["answerable_kept"],
        "dev_unanswerable_rejected": best["unanswerable_rejected"],
    }
    return key, float(best["threshold"]), info


# --------------------------------------------------------------------------- CLI


def build_retrieval_pipeline(settings: Settings) -> RagPipeline:
    """Pipeline chỉ để đo retrieval: không cần LLM trả lời, nhưng cần LLM viết lại nếu bật rewrite."""
    vectorstore = load_or_build_index(settings)
    rewrite_chain = None
    if settings.rewrite_queries:
        check_ollama(settings)
        rewrite_chain = build_rewrite_chain(settings)
    return RagPipeline(
        settings, vectorstore, None, build_retriever(settings, vectorstore), rewrite_chain=rewrite_chain
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Đánh giá RAG chatbot")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--mode", choices=("retrieval", "full"), default="full")
    parser.add_argument(
        "--split",
        choices=("dev", "test", "all"),
        help="mặc định: dev nếu bộ câu hỏi có chia split, ngược lại all",
    )
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "--threshold-from", type=Path, help="báo cáo dev để lấy ngưỡng (bắt buộc với --split test)"
    )
    parser.add_argument("--label", help="tên cấu hình, ghi vào báo cáo")
    parser.add_argument("--include-unverified", action="store_true", help="CHỈ để chạy thử quy trình")
    parser.add_argument("--limit", type=int, help="CHỈ để chạy thử: lấy mẫu ngẫu nhiên N câu")
    parser.add_argument(
        "--allow-unsupported", action="store_true", help="không dừng khi có câu thiếu bằng chứng"
    )
    parser.add_argument("--judge-model", help="bật chấm faithfulness bằng LLM-judge (cần mode full)")
    parser.add_argument("--out", type=Path, help="ghi báo cáo JSON ra file này")
    args = parser.parse_args(argv)

    overrides = parse_overrides(args.overrides)
    settings = Settings(**overrides)
    configure_logging(settings.log_level)
    all_examples = load_dataset(args.dataset)
    split = args.split or default_split(all_examples)

    threshold_selection = None
    if args.threshold_from:
        key, value, threshold_selection = threshold_from_report(args.threshold_from, settings)
        overrides[key] = str(value)
        settings = Settings(**overrides)
    elif split == "test" and not set(THRESHOLD_KEYS) & overrides.keys():
        raise SystemExit(
            "Chạy --split test cần ngưỡng đã chọn trên dev: dùng --threshold-from <báo cáo dev> "
            "(hoặc --set rerank_threshold=... / score_threshold=... nếu bạn tự chọn)."
        )

    examples = select_examples(all_examples, split, args.include_unverified, args.limit)
    if not examples:
        print(
            f"Không có câu hỏi nào (split={split}, đã duyệt tay={not args.include_unverified}).",
            file=sys.stderr,
        )
        return 2

    pipeline = build_pipeline(settings) if args.mode == "full" else build_retrieval_pipeline(settings)
    unsupported = find_unsupported_examples(examples, pipeline.chunks())
    if unsupported and not args.allow_unsupported:
        print(f"Dataset lỗi: không chunk nào chứa bằng chứng cho: {', '.join(unsupported)}", file=sys.stderr)
        return 2

    retrieval = evaluate_retrieval(pipeline, examples, settings)
    generation = None
    if args.mode == "full":
        judge = make_judge(settings, args.judge_model) if args.judge_model else None
        generation = evaluate_generation(pipeline, examples, judge)

    report = build_report(
        settings,
        args.mode,
        examples,
        retrieval,
        generation,
        args.judge_model,
        label=args.label,
        split=split,
        include_unverified=args.include_unverified,
        limit=args.limit,
        threshold_selection=threshold_selection,
        dataset_name=args.dataset.name,
        unsupported=unsupported,
    )
    print(format_summary(report))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nĐã ghi báo cáo: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
