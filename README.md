# RAG Chatbot for PDF Documents

## Overview

Chatbot hỏi đáp tiếng Việt trên các file PDF nội bộ trong `papers/`, chạy hoàn toàn local, dùng được qua CLI, giao diện web hoặc REST API (có streaming). Pipeline: đọc PDF theo trang → chia chunk theo mục → tìm kiếm lai (vector + BM25) → rerank bằng cross-encoder → LLM qua Ollama trả lời kèm nguồn (file, trang, mục). Có bộ đánh giá tự động để mọi thay đổi đều được đo bằng số liệu.

## Features

- Hội thoại nhiều lượt: câu hỏi nối tiếp ("còn nghỉ ốm thì sao?") và câu gõ không dấu được viết lại thành câu độc lập, có dấu trước khi truy hồi.
- REST API FastAPI (`/chat`, `/chat/stream` dạng SSE, `/health`), API key tùy chọn, giao diện web một trang có streaming; Docker + docker-compose; CI bằng GitHub Actions.
- Chunk theo cấu trúc tài liệu: mỗi mục đánh số ("1. …", "2. …") là một chunk, mang kèm tên công ty và tiêu đề tài liệu; trang không có mục rơi về cách cắt theo độ dài.
- Hybrid retrieval: FAISS (`BAAI/bge-m3`, cosine) + BM25 trên văn bản bỏ dấu, gộp bằng Reciprocal Rank Fusion; hỗ trợ cả câu hỏi gõ không dấu.
- Rerank bằng cross-encoder `BAAI/bge-reranker-v2-m3`; điểm reranker dùng làm ngưỡng "không đủ liên quan" nên câu ngoài phạm vi bị từ chối trước khi gọi LLM.
- Trả lời chỉ dựa trên ngữ cảnh; nguồn (file, trang, mục) do code lấy từ metadata của chunk, không để LLM tự ghi.
- Cache FAISS index theo fingerprint nội dung PDF + tham số; cấu hình qua `.env`; xử lý lỗi và logging rõ ràng.
- Bộ đánh giá retrieval / generation / ngưỡng / faithfulness (`python -m ragbot.evaluate`).

## Tech Stack

- Python, LangChain (loaders, vector store, prompt, Ollama), pydantic-settings
- FastAPI + uvicorn (API, SSE streaming), giao diện web HTML/JS thuần, Docker, GitHub Actions
- Embedding `BAAI/bge-m3`, reranker `BAAI/bge-reranker-v2-m3` (sentence-transformers)
- FAISS, BM25 tự cài (`ragbot/lexical.py`)
- Ollama: LLM `qwen2.5:7b-instruct` (mặc định)
- Unstructured: trích xuất nội dung PDF
- pytest, ruff

## Project Structure

```text
.
├── chatbot.py            # entry point: python chatbot.py
├── ragbot/
│   ├── config.py         # Settings (pydantic-settings, đọc .env, tiền tố RAG_)
│   ├── ingest.py         # đọc PDF theo trang; chunk theo mục hoặc theo độ dài
│   ├── indexing.py       # dựng/cache FAISS index (fingerprint theo nội dung PDF + tham số)
│   ├── lexical.py        # BM25 + tokenizer bỏ dấu tiếng Việt
│   ├── rerank.py         # cross-encoder reranker
│   ├── retrieval.py      # dense | hybrid (RRF) + rerank, trả về điểm để áp ngưỡng
│   ├── pipeline.py       # viết lại câu hỏi, ngưỡng, prompt, LLM, streaming, nhận diện từ chối, citation
│   ├── api.py            # FastAPI: /chat, /chat/stream (SSE), /health, API key
│   ├── static/index.html # giao diện web (phục vụ ở /)
│   ├── evaluate.py       # harness đánh giá
│   ├── cli.py            # vòng lặp hỏi đáp có lịch sử, xử lý lỗi, logging
│   └── errors.py
├── evals/
│   ├── dataset.jsonl     # 53 câu hỏi có nhãn
│   ├── baseline.json     # kết quả trước Bước 4 (llama3.2:3b, chunk cố định, dense)
│   ├── after-step4.json  # sau Bước 4 (chunk theo mục, hybrid, rerank, qwen2.5:7b)
│   └── after-step5.json  # sau Bước 5 (thêm viết lại câu hỏi) = cấu hình mặc định hiện tại
├── tests/                # pytest, không cần tải model
├── papers/               # 5 PDF mẫu
├── Dockerfile, docker-compose.yml, .dockerignore
├── .github/workflows/    # ci.yml (lint + test), docker.yml (build image)
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
ollama pull qwen2.5:7b-instruct        # mặc định, ~4.7GB; có thể dùng llama3.2:3b nhẹ hơn
```

Lần chạy đầu sẽ tự tải từ HuggingFace `BAAI/bge-m3` (embedding) và `BAAI/bge-reranker-v2-m3` (~2GB). Muốn nhẹ hơn, đặt `RAG_RERANK_MODEL=` (rỗng) để tắt reranker.

PDF dạng scan (ảnh) cần OCR: cài thêm Tesseract kèm gói ngôn ngữ `vie` (`apt install tesseract-ocr tesseract-ocr-vie`).

Yêu cầu tài nguyên tham khảo: máy phát triển có GPU 8GB chạy được cả ba model (embedding, reranker fp16, LLM 7B). Không có GPU vẫn chạy được nhưng chậm hơn.

## Configuration

Mọi tham số đều tùy chọn. Sao chép `.env.example` thành `.env` để đổi, ví dụ:

```bash
RAG_LLM_MODEL=llama3.2:3b
RAG_RETRIEVAL_MODE=dense
RAG_RERANK_MODEL=
```

Xem đầy đủ trong `.env.example` và [ragbot/config.py](ragbot/config.py). Đổi PDF hoặc tham số chunking/embedding sẽ tự dựng lại index.

## How to Run

Đảm bảo Ollama đang chạy, kích hoạt virtual environment, rồi chạy:

```bash
python chatbot.py
```

Nhập câu hỏi tại prompt `Question:`. Gõ `exit` hoặc `quit` để kết thúc.

```text
Question: Quà tặng giá trị bao nhiêu thì phải khai báo?
Quà tặng có giá trị ước tính trên 1.000.000 VNĐ phải khai báo với quản lý và bộ phận Compliance.

Nguồn:
- 05_quy_tac_ung_xu_doanh_nghiep.pdf (trang 1, mục 3. Quà tặng và tiếp khách)
```

## Web UI & API

```bash
uvicorn ragbot.api:app --host 0.0.0.0 --port 8000
```

Mở `http://localhost:8000` để dùng giao diện web (streaming, nguồn kèm mục, nút "Cuộc trò chuyện mới"). API:

| Endpoint | Mô tả |
|---|---|
| `POST /chat` | Trả lời trọn vẹn: `{text, refused, question, sources[]}` |
| `POST /chat/stream` | Server-Sent Events: `rewrite` (câu hỏi đã viết lại), `token`, `done` (câu trả lời cuối + nguồn), `error` |
| `GET /health` | Trạng thái, model đang dùng, số chunk |

```bash
curl -X POST localhost:8000/chat -H 'content-type: application/json' -d '{
  "question": "còn nghỉ khẩn cấp thì sao?",
  "history": [
    {"role": "user", "content": "Nghỉ ốm cần báo trước mấy giờ?"},
    {"role": "assistant", "content": "Trước 09:00."}
  ]
}'
```

Client giữ lịch sử và gửi lại mỗi lần (server không lưu phiên). Câu hỏi tối đa 1000 ký tự, lịch sử tối đa 40 tin nhắn. Đặt `RAG_API_KEY` để yêu cầu header `X-API-Key` cho `/chat*` (`/health` và trang chủ luôn mở). Sự kiện `done` là câu trả lời có thẩm quyền: nếu model vừa stream một câu từ chối, `done` sẽ thay bằng lời từ chối chuẩn và bỏ nguồn.

Các model dùng chung một GPU và không đảm bảo thread-safe nên mọi lượt hỏi được xếp hàng qua một khóa: phù hợp cho một nhóm nhỏ trong nội bộ, chưa phải dịch vụ đồng thời cao.

## Docker

```bash
docker compose up -d --build
docker compose exec ollama ollama pull qwen2.5:7b-instruct   # lần đầu; app tự khởi động lại khi model đã có
```

Mở `http://localhost:8000`. Compose chạy Ollama và app, mount `./papers` (chỉ đọc) và một volume `/data` giữ model HuggingFace + FAISS index. Image mặc định dùng torch CPU (~850MB, chậm hơn GPU); bỏ comment khối `deploy` trong `docker-compose.yml` để Ollama dùng GPU NVIDIA. `RAG_API_KEY` truyền qua biến môi trường.

## How RAG Works

```text
Câu hỏi + lịch sử ─ (viết lại thành câu độc lập, có dấu, khi cần) ─┐
                                                                     ▼
Câu hỏi ─┬─ vector search (bge-m3, FAISS) ──┐
         └─ BM25 (bỏ dấu)  ────────────────┴─ RRF ─ ứng viên ─ reranker ─ lọc theo ngưỡng ─┬─ rỗng → "không tìm thấy" (không gọi LLM)
                                                                                            └─ top-k chunk → LLM → trả lời + nguồn
```

Điểm dùng để áp ngưỡng là điểm reranker (xác suất [0, 1]) khi bật reranker, hoặc cosine similarity của vector khi tắt. Nguồn hiển thị chỉ gồm các chunk có điểm sát top-1.

## Evaluation

Bộ đánh giá `evals/dataset.jsonl` gồm 53 câu: 39 có đáp án (factoid, paraphrase, đa nguồn, không dấu) và 14 không có đáp án (ngoài chủ đề, gần chủ đề). Mỗi câu có đáp án đi kèm file nguồn vàng, đoạn bằng chứng và từ khóa bắt buộc; harness tự kiểm tra bằng chứng tồn tại trong chunk trước khi chạy.

```bash
python -m ragbot.evaluate --mode retrieval                 # chỉ retrieval, không cần Ollama
python -m ragbot.evaluate --mode full --out evals/reports/run.json
python -m ragbot.evaluate --mode full --set llm_model=llama3.2:3b --set rerank_model=
python -m ragbot.evaluate --mode full --judge-model qwen2.5:7b-instruct   # thêm faithfulness
```

### Trước và sau

| Chỉ số | Baseline (Bước 3) | Hiện tại |
|---|---|---|
| Retrieval hit@1 / hit@5 | 0.846 / 0.974 | 0.974 / 1.000 |
| Retrieval MRR | 0.900 | 0.983 |
| Context precision | 0.429 | 0.905 |
| Accuracy tổng (53 câu) | 0.887 | 0.962 |
| Từ chối đúng câu không có đáp án | 0.929 | 1.000 |
| Tỉ lệ bịa câu trả lời | 0.071 | 0.000 |
| Citation accuracy | 1.000 | 1.000 |
| Latency trung bình / câu | ~0.6 giây (3B) | ~1.3 giây (7B) |

### Ablation retrieval (cùng bộ câu hỏi, chỉ đổi từng thành phần)

| Cấu hình | hit@1 | hit@5 | MRR | Context precision | Không dấu hit@5 |
|---|---|---|---|---|---|
| A. chunk cố định + dense (baseline) | 0.846 | 0.974 | 0.900 | 0.429 | 1.0 (MRR 0.29) |
| B. chunk theo mục + dense | 0.949 | 0.949 | 0.949 | 0.355 | 0.0 |
| C. B + hybrid BM25 | 0.923 | 0.974 | 0.941 | 0.389 | 1.0 |
| D. B + reranker | 0.949 | 0.974 | 0.962 | 0.958 | 0.5 |
| E. C + reranker (**mặc định**) | 0.974 | 1.000 | 0.983 | 0.958 | 1.0 |

Context precision của D/E đo ở ngưỡng reranker 0.30; ở ngưỡng mặc định 0.20 là 0.905.

### So sánh LLM (cấu hình E, 3 lần chạy mỗi model)

| Model | Accuracy 3 lần | Latency |
|---|---|---|
| `llama3.2:3b` | 0.943 / 0.962 / 0.943 (dao động ở `f17`, `f23`) | ~0.45 giây |
| `qwen2.5:7b-instruct` | 0.962 / 0.962 / 0.962 (ổn định) | ~1.4 giây |

Hai model chỉ chênh một câu nên không đủ để kết luận về độ chính xác; `qwen2.5:7b` được chọn vì ổn định hơn giữa các lần chạy. `qwen2.5:7b` chỉ sai hai câu không dấu; `llama3.2:3b` ngoài hai câu đó còn sai ngẫu nhiên thêm 0–1 câu mỗi lần.

### Sau Bước 5 (viết lại câu hỏi)

Thêm bước viết lại câu hỏi (`evals/after-step5.json`) đưa accuracy lên **53/53 ở cả 3 lần chạy liên tiếp**, 0 bịa đáp án, citation 1.000; hai câu không dấu `d01`, `d02` đã đúng. Prompt viết lại được viết một lần theo thiết kế chung, không chỉnh theo hai câu này. Retrieval-only vẫn đo trên câu gốc nên hit@k không đổi; phần cải thiện nằm ở generation. Latency ~1.5 giây/câu (câu không dấu hoặc nối tiếp tốn thêm một lượt LLM).

Bộ eval hiện **đã bão hòa**: chạm 100% nghĩa là không còn phân biệt được các cải tiến tiếp theo, và đây là tập phát triển mà các quyết định (ngưỡng, prompt, model) đã dựa vào, nên 100% không đồng nghĩa với 100% trên dữ liệu thật. Việc cần làm tiếp là thêm câu hỏi mới, khó hơn, chưa từng được dùng để chỉnh hệ thống. Bộ eval cũng chưa đo hội thoại nhiều lượt.

### Lưu ý khi đọc số

- LLM ở temperature 0 vẫn không hoàn toàn tất định; chênh ±1 câu giữa các lần chạy là nhiễu (ví dụ `n08` lúc từ chối lúc bịa "4 ngày" ở `llama3.2:3b`).
- Bộ 53 câu nhỏ và ngưỡng reranker 0.20 được chọn trên chính bộ này, nên có nguy cơ khớp quá mức; cần thêm câu hỏi để kiểm chứng.
- Faithfulness 1.000 của lần chạy cuối do chính `qwen2.5:7b` chấm câu trả lời của chính nó, nên có thiên lệch tự đánh giá; xem như tham khảo. Judge từng mắc lỗi thiết kế (không thấy câu hỏi) và đã được sửa.
- Mỗi lần chạy `evaluate.py` với `--judge-model` làm Ollama nạp lại model xen kẽ nên latency bị đội lên; đo latency khi không dùng judge.

## Development

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check . && ruff format --check .
```

## Limitations

- Câu hỏi không dấu chỉ đúng nhờ bước viết lại bằng LLM (reranker và embedding đều yếu với văn bản mất dấu). Nếu `RAG_REWRITE_QUERIES=false` hoặc LLM viết lại sai thì kết quả quay về như trước. Câu tiếng Anh toàn ASCII cũng bị đưa qua bước này (prompt yêu cầu giữ nguyên).
- Lịch sử hội thoại do client giữ, server không lưu phiên; không có xác thực người dùng, chỉ có một API key dùng chung.
- Chunk theo mục giả định tài liệu có tiêu đề đánh số "N. Tiêu đề"; tài liệu khác cấu trúc rơi về cách cắt theo độ dài.
- Index được dựng lại toàn bộ khi PDF thay đổi (chưa cập nhật tăng dần). `FAISS.load_local` dùng pickle nên không được trỏ `RAG_INDEX_DIR` tới thư mục không tin cậy.
- Khởi động mất vài chục giây vì nạp ba model.
- Chỉ trả lời dựa trên các PDF trong `papers/`.
- Workflow CI và Docker mới được kiểm tra cú pháp và build image ở máy phát triển, chưa từng chạy trên GitHub Actions.

## Future Improvements

- Mở rộng bộ eval (câu khó hơn, hội thoại nhiều lượt) và hiệu chỉnh ngưỡng trên tập kiểm chứng riêng.
- Cập nhật index tăng dần khi thêm PDF; phiên hội thoại phía server, xác thực người dùng.
- Phục vụ mô hình qua một server suy luận riêng để xử lý đồng thời.
