"""Pipeline RAG: retrieval có ngưỡng điểm -> prompt -> LLM -> câu trả lời kèm nguồn."""

import logging
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import httpx
import ollama
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from ragbot.config import Settings, get_settings
from ragbot.errors import LLMUnavailableError
from ragbot.indexing import load_or_build_index
from ragbot.rerank import CrossEncoderReranker
from ragbot.retrieval import Retriever

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


def _format_source(source: Source) -> str:
    where = f"trang {source.page if source.page is not None else '?'}"
    if source.section:
        where += f", mục {source.section}"
    return f"- {source.file} ({where})"


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
        f"[Tài liệu: {d.metadata.get('source', '?')}, trang {d.metadata.get('page', '?')}]\n{d.page_content}"
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
    best: dict[tuple[str, int | None, str | None], Source] = {}
    for doc, score in hits:
        if score < top_score - margin:
            continue
        key = (doc.metadata.get("source", "?"), doc.metadata.get("page"), doc.metadata.get("section"))
        if key not in best or score > best[key].score:
            best[key] = Source(file=key[0], page=key[1], score=score, section=key[2])
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
        vectorstore: FAISS,
        chain: Any,
        retriever: Retriever | None = None,
        rewrite_chain: Any = None,
    ) -> None:
        self._settings = settings
        self._vectorstore = vectorstore
        self._chain = chain
        self._retriever = retriever or Retriever(settings, vectorstore)
        self._rewrite_chain = rewrite_chain

    def chunks(self) -> list[Document]:
        """Mọi chunk đang nằm trong index (dùng cho đánh giá/kiểm tra)."""
        # InMemoryDocstore chưa có API công khai để liệt kê tài liệu.
        return list(self._vectorstore.docstore._dict.values())

    def search(self, question: str) -> list[tuple[Document, float]]:
        """Top-k chunk thô (chưa áp ngưỡng) kèm điểm; ý nghĩa điểm xem ragbot/retrieval.py."""
        return self._retriever.search(question)

    def retrieve(self, question: str) -> list[tuple[Document, float]]:
        """Top-k chunk rồi loại các chunk có điểm dưới ngưỡng đang áp dụng."""
        results = self.search(question)
        hits = [(doc, score) for doc, score in results if score >= self._settings.active_threshold]
        logger.debug("Retrieval: %d/%d chunk đạt ngưỡng", len(hits), len(results))
        return hits

    def standalone_question(self, question: str, history: Sequence[Turn] = ()) -> str:
        """Viết lại câu hỏi nối tiếp / không dấu thành câu độc lập, có dấu (1 lượt gọi LLM nếu cần)."""
        if self._rewrite_chain is None or not self._settings.rewrite_queries:
            return question
        recent = (
            list(history)[-2 * self._settings.max_history_turns :] if self._settings.max_history_turns else []
        )
        if not recent and not looks_unaccented(question):
            return question
        try:
            raw = self._rewrite_chain.invoke({"history": format_history(recent), "question": question})
        except _LLM_ERRORS as exc:
            raise LLMUnavailableError(f"Lỗi khi gọi Ollama: {exc}") from exc
        rewritten = clean_rewrite(raw, question)
        logger.debug("Viết lại câu hỏi: %r -> %r", question, rewritten)
        return rewritten

    def _prepare(self, question: str, history: Sequence[Turn]) -> tuple[str, list[tuple[Document, float]]]:
        standalone = self.standalone_question(question, history)
        return standalone, self.retrieve(standalone)

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
        standalone, hits = self._prepare(question, history)
        if not hits:
            # Không gọi LLM khi retrieval không đủ liên quan: nhanh hơn và không thể bịa.
            return Answer(NOT_FOUND_MESSAGE, refused=True, question=standalone)
        inputs = {"context": format_context([doc for doc, _ in hits]), "question": standalone}
        try:
            raw = self._chain.invoke(inputs)
        except _LLM_ERRORS as exc:
            raise LLMUnavailableError(f"Lỗi khi gọi Ollama: {exc}") from exc
        return self._build_answer(standalone, hits, raw)

    def ask_stream(self, question: str, history: Sequence[Turn] = ()) -> Iterator[StreamEvent]:
        """Như `ask` nhưng phát từng đoạn. Sự kiện "done" mang Answer cuối cùng (có nguồn, đã chuẩn hóa
        lời từ chối): client nên dùng nó thay cho phần text đã ghép từ các sự kiện "token"."""
        standalone, hits = self._prepare(question, history)
        if standalone != question:
            yield StreamEvent("rewrite", text=standalone)
        if not hits:
            yield StreamEvent("done", answer=Answer(NOT_FOUND_MESSAGE, refused=True, question=standalone))
            return
        inputs = {"context": format_context([doc for doc, _ in hits]), "question": standalone}
        parts: list[str] = []
        try:
            for chunk in self._chain.stream(inputs):
                parts.append(chunk)
                yield StreamEvent("token", text=chunk)
        except _LLM_ERRORS as exc:
            raise LLMUnavailableError(f"Lỗi khi gọi Ollama: {exc}") from exc
        yield StreamEvent("done", answer=self._build_answer(standalone, hits, "".join(parts)))


def build_retriever(settings: Settings, vectorstore: FAISS) -> Retriever:
    reranker = CrossEncoderReranker(settings.rerank_model) if settings.rerank_model else None
    return Retriever(settings, vectorstore, reranker)


def build_pipeline(settings: Settings | None = None) -> RagPipeline:
    settings = settings or get_settings()
    check_ollama(settings)
    vectorstore = load_or_build_index(settings)
    llm = ChatOllama(
        model=settings.llm_model,
        temperature=settings.temperature,
        num_ctx=settings.num_ctx,
        base_url=settings.ollama_base_url,
    )
    chain = ChatPromptTemplate.from_template(PROMPT_TEMPLATE) | llm | StrOutputParser()
    # Câu hỏi viết lại chỉ cần một dòng ngắn: giới hạn num_predict để không lan man tốn thời gian.
    rewriter = ChatOllama(
        model=settings.llm_model,
        temperature=0,
        num_ctx=settings.num_ctx,
        num_predict=128,
        base_url=settings.ollama_base_url,
    )
    rewrite_chain = ChatPromptTemplate.from_template(REWRITE_PROMPT) | rewriter | StrOutputParser()
    return RagPipeline(
        settings, vectorstore, chain, build_retriever(settings, vectorstore), rewrite_chain=rewrite_chain
    )
