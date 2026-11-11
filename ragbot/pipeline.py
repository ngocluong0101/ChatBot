"""Pipeline RAG: retrieval có ngưỡng điểm -> prompt -> LLM -> câu trả lời kèm nguồn."""

import logging
import re
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

    def render(self) -> str:
        if not self.sources:
            return self.text
        lines = "\n".join(_format_source(s) for s in self.sources)
        return f"{self.text}\n\nNguồn:\n{lines}"


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


class RagPipeline:
    def __init__(
        self, settings: Settings, vectorstore: FAISS, chain: Any, retriever: Retriever | None = None
    ) -> None:
        self._settings = settings
        self._vectorstore = vectorstore
        self._chain = chain
        self._retriever = retriever or Retriever(settings, vectorstore)

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

    def ask(self, question: str) -> Answer:
        hits = self.retrieve(question)
        if not hits:
            # Không gọi LLM khi retrieval không đủ liên quan: nhanh hơn và không thể bịa.
            return Answer(NOT_FOUND_MESSAGE, refused=True)

        docs = [doc for doc, _ in hits]
        context = format_context(docs)
        try:
            raw = self._chain.invoke({"context": context, "question": question})
        except (ConnectionError, httpx.HTTPError, ollama.ResponseError) as exc:
            raise LLMUnavailableError(f"Lỗi khi gọi Ollama: {exc}") from exc

        text = raw.strip()
        if is_refusal(text):
            return Answer(NOT_FOUND_MESSAGE, refused=True)
        return Answer(
            text,
            sources=select_sources(hits, self._settings.source_score_margin),
            contexts=tuple(d.page_content for d in docs),
        )


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
    return RagPipeline(settings, vectorstore, chain, build_retriever(settings, vectorstore))
