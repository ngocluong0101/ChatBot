from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

PAPERS_DIR = "./papers"
EMBEDDING_MODEL = "BAAI/bge-m3"
LLM_MODEL = "llama3.2:3b"

# Ollama mặc định dùng context window nhỏ và âm thầm cắt đầu prompt khi vượt quá.
# Đặt tường minh để 5 chunk + prompt luôn nằm gọn trong cửa sổ.
NUM_CTX = 4096

TOP_K = 5
# Ngưỡng cosine similarity (đã chuẩn hóa). Dưới ngưỡng => coi như không có thông tin.
# Hiệu chỉnh thủ công trên 12 câu hỏi: câu trong phạm vi có top-1 >= 0.57, câu ngoài chủ đề <= 0.37,
# câu "gần chủ đề nhưng tài liệu không có" tới ~0.50. Sẽ được tinh chỉnh lại bằng bộ eval.
SCORE_THRESHOLD = 0.50

# Chỉ hiển thị nguồn của các chunk có điểm >= top-1 trừ biên độ này.
SOURCE_SCORE_MARGIN = 0.08

NOT_FOUND_MESSAGE = "Tôi không tìm thấy thông tin này trong tài liệu."

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

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


def load_documents(papers_dir: str = PAPERS_DIR) -> list[Document]:
    """Đọc PDF theo từng trang và chuẩn hóa metadata về {source, page}."""
    loader = DirectoryLoader(
        path=papers_dir,
        glob="**/*.pdf",
        loader_cls=UnstructuredFileLoader,
        # mode="paged" để mỗi trang là một Document có page_number (mode "single" không có).
        loader_kwargs={"mode": "paged", "languages": ["vie"]},
        show_progress=True,
        use_multithreading=True,
    )
    docs = loader.load()
    for doc in docs:
        meta = doc.metadata
        # Loader trả về nhiều metadata nhiễu (tọa độ pixel, parent_id...). Chỉ giữ thứ cần cho citation.
        doc.metadata = {
            "source": meta.get("filename") or meta.get("source", "?"),
            "page": meta.get("page_number"),
        }
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )
    return splitter.split_documents(docs)


def build_vectorstore(splits: list[Document]) -> FAISS:
    # normalize_embeddings=True + MAX_INNER_PRODUCT => điểm số trả về chính là cosine similarity.
    # (DistanceStrategy.COSINE của LangChain thực chất dùng IndexFlatL2 và công thức 1 - distance,
    # nên điểm không phải cosine thật và có thể âm.)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.from_documents(
        documents=splits,
        embedding=embeddings,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )


def retrieve(vectorstore: FAISS, question: str) -> list[tuple[Document, float]]:
    """Lấy top-k chunk rồi loại các chunk có cosine similarity dưới ngưỡng."""
    results = vectorstore.similarity_search_with_score(question, k=TOP_K)
    return [(doc, float(score)) for doc, score in results if score >= SCORE_THRESHOLD]


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[Tài liệu: {d.metadata.get('source', '?')}, trang {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )


def format_sources(hits: list[tuple[Document, float]]) -> str:
    """Nguồn do code tạo từ metadata thật của chunk được truy hồi, không để LLM tự bịa.

    Chỉ liệt kê chunk có điểm sát top-1: chunk đứng sau chỉ là "ứng viên phụ" được đưa vào
    context, liệt kê hết sẽ khiến người dùng tưởng câu trả lời lấy từ cả những file không liên quan.
    """
    top_score = max(score for _, score in hits)
    seen: list[str] = []
    for d, score in hits:
        if score < top_score - SOURCE_SCORE_MARGIN:
            continue
        label = f"{d.metadata.get('source', '?')} (trang {d.metadata.get('page', '?')})"
        if label not in seen:
            seen.append(label)
    return "\n".join(f"- {s}" for s in seen)


def is_refusal(answer: str) -> bool:
    """LLM nhỏ hay diễn đạt lại câu từ chối (bỏ chữ "Tôi", đổi dấu câu...), nên so khớp lỏng."""
    return "không tìm thấy thông tin" in answer.casefold()


def answer_question(vectorstore: FAISS, chain, question: str) -> str:
    hits = retrieve(vectorstore, question)
    if not hits:
        # Không gọi LLM khi retrieval không đủ liên quan: nhanh hơn và không thể bịa.
        return NOT_FOUND_MESSAGE

    docs = [doc for doc, _ in hits]
    answer = chain.invoke({"context": format_docs(docs), "question": question}).strip()

    if is_refusal(answer):
        return NOT_FOUND_MESSAGE
    return f"{answer}\n\nNguồn:\n{format_sources(hits)}"


def rag_chatbot() -> None:
    docs = load_documents()
    splits = split_documents(docs)
    vectorstore = build_vectorstore(splits)

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatOllama(model=LLM_MODEL, temperature=0, num_ctx=NUM_CTX)
    chain = prompt | llm | StrOutputParser()

    while True:
        user_input = input("Question: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the chatbot. Goodbye!")
            break
        if not user_input:
            continue

        print(answer_question(vectorstore, chain, user_input))


if __name__ == "__main__":
    rag_chatbot()
