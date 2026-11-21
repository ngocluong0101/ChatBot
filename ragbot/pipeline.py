"""Pipeline RAG: retrieval có ngưỡng điểm -> prompt -> LLM -> câu trả lời kèm nguồn."""

import logging
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import httpx
import ollama
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from ragbot.config import Settings, get_settings
from ragbot.errors import LLMUnavailableError
from ragbot.indexing import load_or_build_index
from ragbot.rerank import CrossEncoderReranker
from ragbot.retrieval import Retriever
from ragbot.telemetry import NOOP, Stage, Telemetry, UsageRecorder, make_usage_handler
from ragbot.vectorstore import all_chunks

logger = logging.getLogger(__name__)

NOT_FOUND_MESSAGE = "Tôi không tìm thấy thông tin này trong tài liệu."

PROMPT_TEMPLATE = (
    "Bạn là trợ lý hỏi đáp cho kho tài liệu nội bộ của công ty.\n"
    "QUY TẮC:\n"
    "1) Chỉ dùng thông tin trong phần NGỮ CẢNH bên dưới để trả lời.\n"
    f'2) Nếu ngữ cảnh không chứa câu trả lời, chỉ trả lời đúng một câu: "{NOT_FOUND_MESSAGE}"\n'
    "3) Không suy đoán, không dùng kiến thức bên ngoài tài liệu.\n"
    "4) Trả lời ngắn gọn, chính xác, bằng ngôn ngữ của câu hỏi.\n"
    "5) Không tự ghi nguồn tài liệu; hệ thống sẽ tự thêm phần nguồn.\n\n"
    "NGỮ CẢNH:\n{context}\n\n"
    "CÂU HỎI: {question}\n"
    "TRẢ LỜI:"
)


REWRITE_PROMPT = (
    "Viết lại CÂU HỎI MỚI thành một câu hỏi độc lập, đầy đủ ngữ cảnh, bằng tiếng Việt có dấu.\n"
    '- Dùng LỊCH SỬ để thay đại từ/ý bỏ lửng ("nó", "còn trường hợp...") bằng nội dung cụ thể.\n'
    "- Nếu câu hỏi là tiếng Việt không dấu, hãy thêm dấu đúng chính tả.\n"
    "- Giữ nguyên ý, số liệu và thuật ngữ; không trả lời câu hỏi, không thêm thông tin.\n"
    "- Nếu câu hỏi đã độc lập và đúng chính tả thì giữ nguyên.\n"
    "- Chỉ in ra đúng một câu hỏi đã viết lại, không giải thích.\n\n"
    "LỊCH SỬ:\n{history}\n\n"
    "CÂU HỎI MỚI: {question}\n"
    "CÂU HỎI ĐỘC LẬP:"
)

MAX_TURN_CHARS = 500  # cắt mỗi lượt trong lịch sử để prompt viết lại không phình to


@dataclass(frozen=True)
class Turn:
    """Một lượt trong lịch sử hội thoại. role: "user" | "assistant"."""

    role: str
    content: str


@dataclass(frozen=True)
class Source:
    file: str
    page: int | None
    score: float
    section: str | None = None
    # Chỉ có với chunk từ văn bản pháp luật (chunking="legal"); xem ragbot/legal.py.
    title: str | None = None  # tên văn bản, ví dụ "Bộ luật Lao động 2019"
    dieu: str | None = None
    khoan: str | None = None
    phu_luc: str | None = None
    kind: str | None = None  # "text" | "table" | "preamble"


@dataclass(frozen=True)
class Answer:
    text: str
    sources: tuple[Source, ...] = ()
    refused: bool = False
    # Các đoạn văn bản đã đưa vào prompt; dùng để kiểm tra/đánh giá tính trung thực (faithfulness).
    contexts: tuple[str, ...] = ()
    # Câu hỏi độc lập thực sự dùng để truy hồi (khác câu người dùng gõ nếu đã được viết lại).
    question: str = ""

    def render(self) -> str:
        if not self.sources:
            return self.text
        lines = "\n".join(_format_source(s) for s in self.sources)
        return f"{self.text}\n\nNguồn:\n{lines}"


@dataclass(frozen=True)
class StreamEvent:
    """kind: "rewrite" (text = câu hỏi đã viết lại) | "token" (text = đoạn mới) | "done" (answer)."""

    kind: str
    text: str = ""
    answer: Answer | None = None


def format_citation(source: Source) -> str:
    """ "Bộ luật Lao động 2019, Điều 113, Khoản 1 (trang 41)"; nguồn cũ: "file.pdf (trang 1, mục 2. Tên)"."""
    page = f"trang {source.page if source.page is not None else '?'}"
    if source.title:
        parts = [source.title]
        if source.dieu:
            parts.append(f"Điều {source.dieu}")
        if source.phu_luc:
            parts.append(f"Phụ lục {source.phu_luc}")
        if source.khoan:
            parts.append(f"Khoản {source.khoan}")
        if source.kind == "table":
            parts.append("bảng")
        return f"{', '.join(parts)} ({page})"
    where = f"{page}, mục {source.section}" if source.section else page
    return f"{source.file} ({where})"


def _format_source(source: Source) -> str:
    return f"- {format_citation(source)}"


# LLM nhỏ hay diễn đạt lại câu từ chối ("Không có thông tin về...", "tài liệu không đề cập...").
REFUSAL_PATTERN = re.compile(
    r"không (?:tìm thấy|có|đề cập|được đề cập|nhắc)[^.\n]{0,40}?(?:thông tin|trong tài liệu|đến)"
    r"|tài liệu không (?:đề cập|nhắc|có thông tin)"
    r"|không (?:thể )?(?:biết|xác định) (?:được )?dựa trên",
    re.IGNORECASE,
)


def is_refusal(answer: str) -> bool:
    return REFUSAL_PATTERN.search(answer) is not None


def looks_unaccented(text: str) -> bool:
    """Câu toàn ký tự ASCII và có từ 3 từ trở lên: nhiều khả năng là tiếng Việt gõ không dấu."""
    return text.isascii() and len(re.findall(r"[A-Za-z]+", text)) >= 3


def format_history(history: Sequence[Turn]) -> str:
    if not history:
        return "(trống)"
    names = {"user": "Người dùng", "assistant": "Trợ lý"}
    return "\n".join(f"{names.get(t.role, t.role)}: {t.content[:MAX_TURN_CHARS]}" for t in history)


def clean_rewrite(raw: str, original: str) -> str:
    """LLM nhỏ đôi khi thêm tiền tố/ngoặc kép hoặc nói lan man; không tin được thì giữ câu gốc."""
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return original
    candidate = re.sub(r"^(câu hỏi độc lập|câu hỏi)\s*:\s*", "", lines[0], flags=re.IGNORECASE)
    candidate = candidate.strip(" \"'`“”")
    if not candidate or len(candidate) > 4 * len(original) + 200:
        return original
    return candidate


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[Tài liệu: {d.metadata.get('doc_title') or d.metadata.get('source', '?')}, "
        f"trang {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )


def select_sources(hits: list[tuple[Document, float]], margin: float) -> tuple[Source, ...]:
    """Nguồn do code tạo từ metadata thật của chunk được truy hồi, không để LLM tự bịa.

    Chỉ giữ chunk có điểm sát top-1: chunk đứng sau chỉ là "ứng viên phụ" được đưa vào context,
    liệt kê hết sẽ khiến người dùng tưởng câu trả lời lấy từ cả những file không liên quan.
    """
    if not hits:
        return ()
    top_score = max(score for _, score in hits)
    best: dict[tuple[Any, ...], Source] = {}
    for doc, score in hits:
        if score < top_score - margin:
            continue
        meta = doc.metadata
        key = (
            meta.get("source", "?"),
            meta.get("page"),
            meta.get("section"),
            meta.get("khoan"),
            meta.get("type"),
        )
        if key not in best or score > best[key].score:
            best[key] = Source(
                file=key[0],
                page=key[1],
                score=score,
                section=key[2],
                title=meta.get("doc_title"),
                dieu=meta.get("dieu"),
                khoan=key[3],
                phu_luc=meta.get("phu_luc"),
                kind=key[4] if meta.get("doc_title") else None,
            )
    return tuple(sorted(best.values(), key=lambda s: s.score, reverse=True))


def check_ollama(settings: Settings) -> None:
    """Fail sớm với thông báo rõ ràng thay vì để traceback nổ ở câu hỏi đầu tiên."""
    try:
        listing = ollama.Client(host=settings.ollama_base_url).list()
    except (ConnectionError, httpx.HTTPError) as exc:
        raise LLMUnavailableError(
            f"Không kết nối được Ollama tại {settings.ollama_base_url}. Hãy chạy `ollama serve`."
        ) from exc

    wanted = settings.llm_model if ":" in settings.llm_model else f"{settings.llm_model}:latest"
    available = {m.model for m in listing.models}
    if wanted not in available:
        raise LLMUnavailableError(
            f"Ollama chưa có model '{settings.llm_model}'. Hãy chạy: ollama pull {settings.llm_model}"
        )


_LLM_ERRORS = (ConnectionError, httpx.HTTPError, ollama.ResponseError)


class RagPipeline:
    def __init__(
        self,
        settings: Settings,
        vectorstore: Any,
        chain: Any,
        retriever: Retriever | None = None,
        rewrite_chain: Any = None,
    ) -> None:
        self._settings = settings
        self._vectorstore = vectorstore
        self._chain = chain
        self._retriever = retriever or Retriever(settings, vectorstore)
        self._rewrite_chain = rewrite_chain
        self._chunk_cache: list[Document] | None = None
        self.usage: UsageRecorder | None = None  # đọc số token của lượt sinh vừa xong (xem telemetry.py)
        self._telemetry: Telemetry = NOOP
        self.telemetry = NOOP

    @property
    def telemetry(self) -> Telemetry:
        return self._telemetry

    @telemetry.setter
    def telemetry(self, value: Telemetry) -> None:
        """Gắn telemetry cho cả pipeline lẫn retriever (retriever dùng nó để tạo span con)."""
        self._telemetry = value
        if hasattr(self._retriever, "telemetry"):
            self._retriever.telemetry = value

    def chunks(self) -> list[Document]:
        """Mọi chunk đang nằm trong index (dùng cho đánh giá/kiểm tra). Chỉ mục không đổi khi đang chạy nên
        chỉ đọc một lần (với Qdrant đây là một lượt đọc toàn bộ, không nên lặp ở mỗi lần gọi /health)."""
        if self._chunk_cache is None:
            self._chunk_cache = all_chunks(self._vectorstore)
        return list(self._chunk_cache)

    def search(self, question: str, parent: Stage | None = None) -> list[tuple[Document, float]]:
        """Top-k chunk thô (chưa áp ngưỡng) kèm điểm; ý nghĩa điểm xem ragbot/retrieval.py."""
        if parent is None:
            return self._retriever.search(question)
        return self._retriever.search(question, parent=parent)

    def retrieve(self, question: str, parent: Stage | None = None) -> list[tuple[Document, float]]:
        """Top-k chunk rồi loại các chunk có điểm dưới ngưỡng đang áp dụng."""
        with self.telemetry.stage("retrieve", parent) as stage:
            results = self.search(question, stage)
            stage.set("retrieve.candidates", len(results))
        hits = [(doc, score) for doc, score in results if score >= self._settings.active_threshold]
        logger.debug("Retrieval: %d/%d chunk đạt ngưỡng", len(hits), len(results))
        return hits

    def standalone_question(
        self, question: str, history: Sequence[Turn] = (), parent: Stage | None = None
    ) -> str:
        """Viết lại câu hỏi nối tiếp / không dấu thành câu độc lập, có dấu (1 lượt gọi LLM nếu cần)."""
        if self._rewrite_chain is None or not self._settings.rewrite_queries:
            return question
        recent = (
            list(history)[-2 * self._settings.max_history_turns :] if self._settings.max_history_turns else []
        )
        if not recent and not looks_unaccented(question):
            return question
        try:
            with self.telemetry.stage("rewrite", parent):
                raw = self._rewrite_chain.invoke({"history": format_history(recent), "question": question})
        except _LLM_ERRORS as exc:
            raise LLMUnavailableError(f"Lỗi khi gọi Ollama: {exc}") from exc
        rewritten = clean_rewrite(raw, question)
        logger.debug("Viết lại câu hỏi: %r -> %r", question, rewritten)
        return rewritten

    def _prepare(
        self, question: str, history: Sequence[Turn], parent: Stage | None = None
    ) -> tuple[str, list[tuple[Document, float]]]:
        standalone = self.standalone_question(question, history, parent)
        return standalone, self.retrieve(standalone, parent)

    def _finish_llm(self, stage: Stage) -> None:
        """Ghi số token (nếu Ollama báo) vào span và bộ đếm /metrics."""
        usage = self.usage.pop() if self.usage else None
        if usage:
            for key, value in usage.items():
                stage.set(f"llm.{key}", value)
            self.telemetry.observe_tokens(usage)

    def _build_answer(self, standalone: str, hits: list[tuple[Document, float]], raw: str) -> Answer:
        text = raw.strip()
        if is_refusal(text):
            return Answer(NOT_FOUND_MESSAGE, refused=True, question=standalone)
        return Answer(
            text,
            sources=select_sources(hits, self._settings.source_score_margin),
            contexts=tuple(doc.page_content for doc, _ in hits),
            question=standalone,
        )

    def ask(self, question: str, history: Sequence[Turn] = ()) -> Answer:
        with self.telemetry.stage("ask", **{"rag.question": question}) as root:
            standalone, hits = self._prepare(question, history, root)
            root.set("rag.standalone_question", standalone)
            if not hits:
                # Không gọi LLM khi retrieval không đủ liên quan: nhanh hơn và không thể bịa.
                root.set("rag.refused", True)
                return Answer(NOT_FOUND_MESSAGE, refused=True, question=standalone)
            inputs = {"context": format_context([doc for doc, _ in hits]), "question": standalone}
            try:
                with self.telemetry.stage("llm", root, **{"llm.model": self._settings.llm_model}) as llm:
                    raw = self._chain.invoke(inputs)
                    self._finish_llm(llm)
            except _LLM_ERRORS as exc:
                raise LLMUnavailableError(f"Lỗi khi gọi Ollama: {exc}") from exc
            answer = self._build_answer(standalone, hits, raw)
            root.set("rag.refused", answer.refused)
            return answer

    def ask_stream(self, question: str, history: Sequence[Turn] = ()) -> Iterator[StreamEvent]:
        """Như `ask` nhưng phát từng đoạn. Sự kiện "done" mang Answer cuối cùng (có nguồn, đã chuẩn hóa
        lời từ chối): client nên dùng nó thay cho phần text đã ghép từ các sự kiện "token"."""
        with self.telemetry.stage("ask", **{"rag.question": question, "rag.stream": True}) as root:
            standalone, hits = self._prepare(question, history, root)
            root.set("rag.standalone_question", standalone)
            if standalone != question:
                yield StreamEvent("rewrite", text=standalone)
            if not hits:
                root.set("rag.refused", True)
                yield StreamEvent("done", answer=Answer(NOT_FOUND_MESSAGE, refused=True, question=standalone))
                return
            inputs = {"context": format_context([doc for doc, _ in hits]), "question": standalone}
            parts: list[str] = []
            try:
                with self.telemetry.stage("llm", root, **{"llm.model": self._settings.llm_model}) as llm:
                    for chunk in self._chain.stream(inputs):
                        if not parts:
                            llm.set("llm.ttft_seconds", round(llm.elapsed(), 4))
                        parts.append(chunk)
                        yield StreamEvent("token", text=chunk)
                    self._finish_llm(llm)
            except _LLM_ERRORS as exc:
                raise LLMUnavailableError(f"Lỗi khi gọi Ollama: {exc}") from exc
            answer = self._build_answer(standalone, hits, "".join(parts))
            root.set("rag.refused", answer.refused)
            yield StreamEvent("done", answer=answer)


def build_retriever(settings: Settings, vectorstore: Any) -> Retriever:
    reranker = (
        CrossEncoderReranker(settings.rerank_model, settings.rerank_device) if settings.rerank_model else None
    )
    return Retriever(settings, vectorstore, reranker)


def build_rewrite_chain(settings: Settings) -> Any:
    # Câu hỏi viết lại chỉ cần một dòng ngắn: giới hạn num_predict để không lan man tốn thời gian.
    rewriter = ChatOllama(
        model=settings.llm_model,
        temperature=0,
        num_ctx=settings.num_ctx,
        num_predict=128,
        base_url=settings.ollama_base_url,
    )
    return ChatPromptTemplate.from_template(REWRITE_PROMPT) | rewriter | StrOutputParser()


def build_pipeline(settings: Settings | None = None) -> RagPipeline:
    settings = settings or get_settings()
    check_ollama(settings)
    vectorstore = load_or_build_index(settings)
    usage = UsageRecorder()
    llm = ChatOllama(
        model=settings.llm_model,
        temperature=settings.temperature,
        num_ctx=settings.num_ctx,
        base_url=settings.ollama_base_url,
        callbacks=[make_usage_handler(usage)],
    )
    chain = ChatPromptTemplate.from_template(PROMPT_TEMPLATE) | llm | StrOutputParser()
    pipeline = RagPipeline(
        settings,
        vectorstore,
        chain,
        build_retriever(settings, vectorstore),
        rewrite_chain=build_rewrite_chain(settings),
    )
    pipeline.usage = usage
    return pipeline
