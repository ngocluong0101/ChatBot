# RAG Chatbot for PDF Documents

## Overview

CLI chatbot hỏi đáp dựa trên nội dung các file PDF trong thư mục `papers/`. Ứng dụng tải tài liệu, chia nhỏ nội dung, lập chỉ mục bằng FAISS và dùng LLM local qua Ollama để tạo câu trả lời từ các đoạn liên quan.

## Features

- Đọc nhiều file PDF trong `papers/`, hỗ trợ nội dung tiếng Việt.
- Chia văn bản bằng `RecursiveCharacterTextSplitter` với `chunk_size=1200` và `chunk_overlap=200`.
- Tìm kiếm tương đồng bằng FAISS với cosine distance, lấy 5 đoạn liên quan nhất.
- Trả lời chỉ dựa trên context được truy hồi; có thể hiển thị nguồn trong prompt.
- Thoát chương trình bằng `exit` hoặc `quit`.

## Tech Stack

- Python
- LangChain: document loading, text splitting, prompt và RAG chain
- HuggingFace Embeddings: `BAAI/bge-m3`
- FAISS: vector store với cosine similarity
- Ollama: LLM `llama3.2:3b`
- Unstructured: đọc và trích xuất nội dung PDF

## Project Structure

```text
.
├── chatbot.py            # entry point: python chatbot.py
├── ragbot/
│   ├── config.py         # Settings (pydantic-settings, đọc .env, tiền tố RAG_)
│   ├── ingest.py         # đọc PDF theo trang, chia chunk
│   ├── indexing.py       # dựng/cache FAISS index (fingerprint theo nội dung PDF)
│   ├── pipeline.py       # retrieval có ngưỡng điểm, prompt, LLM, citation
│   ├── cli.py            # vòng lặp hỏi đáp, xử lý lỗi, logging
│   └── errors.py
├── tests/                # pytest, không cần tải model
├── papers/               # 5 PDF mẫu
├── .env.example          # mẫu cấu hình
├── pyproject.toml        # cấu hình ruff + pytest
├── requirements.txt
└── requirements-dev.txt
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # thêm requirements-dev.txt nếu muốn chạy test/lint
```

Cài Ollama theo hướng dẫn tại [ollama.com](https://ollama.com), sau đó tải model:

```bash
ollama pull llama3.2:3b
```

Model embedding `BAAI/bge-m3` sẽ được HuggingFace tải khi chạy lần đầu.

PDF dạng scan (ảnh) cần OCR: cài thêm Tesseract kèm gói ngôn ngữ `vie` (`apt install tesseract-ocr tesseract-ocr-vie`).

## Configuration

Mọi tham số đều tùy chọn. Sao chép `.env.example` thành `.env` để đổi, ví dụ:

```bash
RAG_LLM_MODEL=qwen2.5:7b-instruct
RAG_TOP_K=4
RAG_SCORE_THRESHOLD=0.5
```

Xem đầy đủ trong `.env.example` và [ragbot/config.py](ragbot/config.py). Đổi PDF hoặc tham số chunking/embedding sẽ tự dựng lại index.

## Evaluation

Bộ đánh giá nằm trong `evals/dataset.jsonl` (53 câu: 39 có đáp án gồm factoid / paraphrase / đa nguồn / không dấu, và 14 không có đáp án gồm ngoài chủ đề / gần chủ đề). Mỗi câu có đáp án đi kèm file nguồn vàng, đoạn bằng chứng và từ khóa bắt buộc; harness tự kiểm tra bằng chứng có tồn tại trong chunk trước khi chạy.

```bash
python -m ragbot.evaluate --mode retrieval                 # chỉ retrieval, không cần Ollama
python -m ragbot.evaluate --mode full --out evals/reports/run.json
python -m ragbot.evaluate --mode full --set llm_model=qwen2.5:7b-instruct --set top_k=3
python -m ragbot.evaluate --mode full --judge-model qwen2.5:7b-instruct   # thêm faithfulness
```

Baseline (`llama3.2:3b`, `bge-m3`, chunk 1200/200, top_k 5, ngưỡng 0.50) lưu ở `evals/baseline.json`:

| Chỉ số | Giá trị |
|---|---|
| Retrieval hit@1 / hit@3 / hit@5 | 0.846 / 0.949 / 0.974 |
| Retrieval MRR | 0.900 |
| Context precision (chunk đạt ngưỡng thực sự liên quan) | 0.429 |
| Accuracy tổng (53 câu) | 0.887 |
| Accuracy câu có đáp án / từ chối đúng câu không có đáp án | 0.872 / 0.929 |
| Citation accuracy | 1.000 |
| Faithfulness (LLM-judge, chỉ mang tính tham khảo) | 0.944 |

Phát hiện từ baseline (đầu vào cho các cải tiến tiếp theo):

- **Câu không dấu** (`d01`, `d02`): accuracy 0, MRR 0.29. `bge-m3` kém khi mất dấu, top-1 chỉ 0.36–0.44 nên bị ngưỡng chặn.
- **Ngưỡng không tách sạch được**: câu có đáp án có top-1 thấp nhất 0.355, câu không có đáp án cao nhất 0.565 (chồng lấn). Ngưỡng tốt nhất theo balanced accuracy là 0.54 nhưng phải đánh đổi.
- **Chunk quá thô**: context precision chỉ 0.43; `f20` bỏ lỡ đúng chunk dù cùng file vì chunk đầu mỗi tài liệu chứa cả khối tiêu đề/metadata.
- **Phát hiện từ chối chưa đủ rộng**: `n04` trả "Không có thông tin…" nhưng không khớp mẫu từ chối nên không bị coi là từ chối.
- **Model 3B**: `f20` chọn nhầm câu, `p06` từ chối nhầm dù retrieval đúng; `f08` đổi "từ 3 lần" thành "trên 3 lần" (judge bắt được).

Lưu ý khi đọc số: LLM ở temperature 0 vẫn không hoàn toàn tất định (hai lần chạy cho accuracy 0.868 và 0.887), nên chênh lệch ±1 câu là nhiễu. Latency trong `baseline.json` bị đội lên do Ollama phải nạp lại model khi xen kẽ judge; chạy không `--judge-model` cho latency thực (~0.7 giây/câu). LLM-judge là phép đo thô: đã phải sửa một lần vì judge không thấy câu hỏi, và vẫn còn false positive.

## Development

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check . && ruff format --check .
```

## How to Run

Đảm bảo Ollama đang chạy, kích hoạt virtual environment, rồi chạy:

```bash
python chatbot.py
```

Nhập câu hỏi tại prompt `Question:`. Gõ `exit` hoặc `quit` để kết thúc.

## How RAG Works

Pipeline hiện tại:

```text
PDF Documents → Document Loading → Text Chunking → Embeddings →
FAISS Vector Store → Retrieval → LLM → Answer
```

`DirectoryLoader` tìm các file `**/*.pdf` trong `papers/` và `UnstructuredFileLoader` trích xuất nội dung. Các đoạn văn được chuyển thành vector bằng `BAAI/bge-m3`, lưu trong FAISS và truy hồi 5 đoạn gần nhất. Những đoạn này được đưa vào prompt cho Ollama `llama3.2:3b`, với yêu cầu không dùng kiến thức bên ngoài tài liệu.

```text
User Question → Embedding → FAISS Retrieval → Relevant Chunks → LLM → Answer
```

## Example Usage

```text
$ python chatbot.py
Question: Quy định làm việc từ xa là gì?
<câu trả lời dựa trên các PDF trong papers/>
Question: exit
Exiting the chatbot. Goodbye!
```

## Limitations

- Chỉ trả lời dựa trên các PDF được đặt trong `papers/`.
- Cần chạy Ollama local và có model `llama3.2:3b`.
- Việc tải model embedding lần đầu cần kết nối mạng và có thể tốn thời gian/dung lượng.
- Index được cache vào `.index/`; khi PDF thay đổi, toàn bộ index được dựng lại (chưa cập nhật tăng dần).
- Chất lượng kết quả phụ thuộc vào nội dung PDF, quá trình trích xuất và model local.

## Future Improvements

- Thêm giao diện web và lịch sử hội thoại.
- Bộ đánh giá (eval) cho retrieval và generation.
