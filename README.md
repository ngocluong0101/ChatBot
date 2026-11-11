# RAG Chatbot – Hỏi đáp tài liệu nội bộ bằng tiếng Việt

Chatbot giúp nhân viên **đặt câu hỏi bằng ngôn ngữ tự nhiên** về các tài liệu quy định của công ty (nội quy, chính sách bảo mật, làm việc từ xa, nghỉ phép, quy tắc ứng xử…) và nhận câu trả lời **kèm nguồn trích dẫn** (tên file, trang, mục).

Toàn bộ hệ thống chạy **trên máy của bạn**: tài liệu và câu hỏi không bị gửi lên dịch vụ bên ngoài.

## Ví dụ

```text
Bạn:    Quà tặng giá trị bao nhiêu thì phải khai báo?
Chatbot: Quà tặng có giá trị ước tính trên 1.000.000 VNĐ phải khai báo với quản lý
         và bộ phận Compliance.

         Nguồn:
         - 05_quy_tac_ung_xu_doanh_nghiep.pdf (trang 1, mục 3. Quà tặng và tiếp khách)

Bạn:    Công ty có thưởng Tết không?
Chatbot: Tôi không tìm thấy thông tin này trong tài liệu.
```

Chatbot chỉ trả lời dựa trên tài liệu được cung cấp. Nếu tài liệu không có thông tin, nó nói thẳng là không tìm thấy thay vì đoán.

## Tính năng

- **Trả lời có nguồn:** luôn cho biết thông tin lấy từ file, trang và mục nào để người dùng tự kiểm tra.
- **Từ chối khi không có thông tin:** câu hỏi ngoài phạm vi tài liệu được từ chối thay vì bịa ra câu trả lời.
- **Hỏi tiếp tự nhiên:** hiểu các câu nối tiếp như "còn nghỉ khẩn cấp thì sao?" dựa trên cuộc trò chuyện trước đó.
- **Gõ không dấu vẫn hiểu:** ví dụ "mat khau toi thieu bao nhieu ky tu".
- **Ba cách dùng:** giao diện web (câu trả lời hiện dần theo thời gian thực), dòng lệnh, hoặc REST API.
- **Chạy local:** dùng mô hình ngôn ngữ chạy trên máy qua Ollama; có thể đóng gói bằng Docker.

## Công nghệ sử dụng

| Nhóm | Công nghệ |
|---|---|
| Ngôn ngữ, framework | Python, LangChain, FastAPI |
| Mô hình ngôn ngữ | Ollama (mặc định `qwen2.5:7b-instruct`) |
| Tìm kiếm | Embedding `BAAI/bge-m3`, FAISS, BM25 (tự cài), reranker `BAAI/bge-reranker-v2-m3` |
| Đọc tài liệu | Unstructured (đọc PDF, hỗ trợ tiếng Việt) |
| Triển khai | Docker, Docker Compose, GitHub Actions |
| Chất lượng mã | pytest (hơn 100 bài kiểm thử), ruff |

## Cách hoạt động

```text
Câu hỏi ──► (nếu cần) viết lại thành câu hỏi đầy đủ, có dấu
        ──► tìm đoạn liên quan: tìm theo ý nghĩa + tìm theo từ khóa
        ──► chấm lại độ liên quan, loại đoạn không phù hợp
        ──► không còn đoạn nào phù hợp?  → trả lời "không tìm thấy" (không gọi mô hình)
        ──► còn đoạn phù hợp?            → mô hình ngôn ngữ trả lời + hiển thị nguồn
```

Một vài điểm đáng chú ý:

- **Chia tài liệu theo từng mục** ("1. …", "2. …") thay vì cắt theo độ dài, để mỗi đoạn là một ý trọn vẹn.
- **Kết hợp hai cách tìm kiếm:** tìm theo ý nghĩa (hiểu câu diễn đạt khác) và tìm theo từ khóa (chính xác với số liệu, thuật ngữ, câu không dấu).
- **Chấm lại bằng một mô hình riêng** để chọn đúng đoạn cần dùng; điểm này cũng dùng để quyết định câu hỏi có nằm trong phạm vi tài liệu hay không.
- **Nguồn trích dẫn do chương trình tự lấy** từ dữ liệu của tài liệu, không để mô hình tự ghi (mô hình nhỏ dễ ghi sai nguồn).
- **Lưu sẵn chỉ mục tìm kiếm** trên đĩa; chỉ xây dựng lại khi tài liệu hoặc cấu hình thay đổi.

## Bắt đầu nhanh

### Cách 1: Docker (khuyên dùng khi triển khai)

```bash
docker compose up -d --build
docker compose exec ollama ollama pull qwen2.5:7b-instruct   # chỉ cần lần đầu
```

Mở trình duyệt tại **http://localhost:8000**.

Lưu ý:
- Lần đầu cần mạng để tải mô hình (khoảng 10GB gồm mô hình ngôn ngữ và hai mô hình tìm kiếm).
- Mặc định chạy bằng CPU nên sẽ chậm. Nếu có GPU NVIDIA, bỏ dấu `#` ở khối `deploy` trong `docker-compose.yml`. Nếu máy yếu, đặt `RAG_LLM_MODEL=llama3.2:3b` và `RAG_RERANK_MODEL=` (để trống) để nhẹ hơn.
- Nếu cổng 8000 đã bị chiếm, đổi `"8000:8000"` trong `docker-compose.yml` thành cổng khác, ví dụ `"8080:8000"`.

### Cách 2: Chạy trực tiếp trên máy

Cần Python 3.10 và [Ollama](https://ollama.com).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

ollama pull qwen2.5:7b-instruct

python chatbot.py                                   # dùng qua dòng lệnh
uvicorn ragbot.api:app --host 0.0.0.0 --port 8000   # hoặc mở giao diện web ở http://localhost:8000
```

Lần chạy đầu sẽ tự tải mô hình tìm kiếm từ HuggingFace (khoảng 4GB). Trong dòng lệnh, gõ `reset` để bắt đầu cuộc trò chuyện mới và `exit` để thoát.

Để dùng tài liệu của bạn, bỏ các file PDF vào thư mục `papers/` rồi chạy lại; chỉ mục sẽ tự được xây dựng lại.

PDF dạng ảnh scan cần OCR: cài thêm Tesseract kèm gói tiếng Việt (`apt install tesseract-ocr tesseract-ocr-vie`).

## Dùng qua API

| Đường dẫn | Mô tả |
|---|---|
| `POST /chat` | Trả lời đầy đủ: `{text, refused, question, sources[]}` |
| `POST /chat/stream` | Trả lời hiện dần (Server-Sent Events): `rewrite`, `token`, `done`, `error` |
| `GET /health` | Kiểm tra trạng thái, mô hình đang dùng, số đoạn tài liệu |

```bash
curl -X POST localhost:8000/chat -H 'content-type: application/json' -d '{
  "question": "còn nghỉ khẩn cấp thì sao?",
  "history": [
    {"role": "user", "content": "Nghỉ ốm cần báo trước mấy giờ?"},
    {"role": "assistant", "content": "Trước 09:00."}
  ]
}'
```

- Phía người dùng giữ lịch sử và gửi lại mỗi lần; máy chủ không lưu phiên.
- Giới hạn: câu hỏi tối đa 1000 ký tự, lịch sử tối đa 40 tin nhắn.
- Đặt biến `RAG_API_KEY` để yêu cầu header `X-API-Key` cho `/chat` và `/chat/stream`.
- Các lượt hỏi được xử lý lần lượt (xếp hàng), phù hợp một nhóm nhỏ dùng nội bộ.

## Cấu hình

Mọi tham số đều có giá trị mặc định. Muốn đổi, sao chép `.env.example` thành `.env` rồi sửa, ví dụ:

```bash
RAG_LLM_MODEL=llama3.2:3b     # dùng mô hình nhẹ hơn
RAG_RERANK_MODEL=             # tắt mô hình chấm lại để tiết kiệm bộ nhớ
RAG_API_KEY=mat-khau-cua-ban  # bật khóa API
```

Danh sách đầy đủ có trong [.env.example](.env.example) và [ragbot/config.py](ragbot/config.py).

## Cấu trúc thư mục

```text
.
├── chatbot.py              # chạy chatbot trên dòng lệnh
├── ragbot/
│   ├── config.py           # cấu hình (đọc từ .env)
│   ├── ingest.py           # đọc PDF và chia tài liệu theo mục
│   ├── indexing.py         # xây dựng và lưu sẵn chỉ mục tìm kiếm
│   ├── lexical.py          # tìm kiếm theo từ khóa (BM25, bỏ dấu tiếng Việt)
│   ├── rerank.py           # chấm lại độ liên quan
│   ├── retrieval.py        # kết hợp các cách tìm kiếm
│   ├── pipeline.py         # luồng hỏi đáp: viết lại câu hỏi, gọi mô hình, nguồn trích dẫn
│   ├── api.py              # REST API (FastAPI)
│   ├── static/index.html   # giao diện web
│   └── cli.py              # giao diện dòng lệnh
├── papers/                 # tài liệu PDF mẫu
├── tests/                  # kiểm thử tự động
├── Dockerfile, docker-compose.yml
└── .github/workflows/      # kiểm tra tự động khi đẩy mã lên GitHub
```

## Kiểm thử

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check . && ruff format --check .
```

Các bài kiểm thử không cần tải mô hình và không cần Ollama.

## Hạn chế hiện tại

- Câu hỏi gõ không dấu chỉ hoạt động tốt nhờ việc viết lại câu hỏi bằng mô hình ngôn ngữ; nếu tắt tính năng này (`RAG_REWRITE_QUERIES=false`) hoặc mô hình viết lại sai thì độ chính xác giảm.
- Việc chia theo mục giả định tài liệu có tiêu đề đánh số dạng "1. Tiêu đề"; tài liệu cấu trúc khác sẽ được cắt theo độ dài.
- Khi thêm hoặc sửa PDF, toàn bộ chỉ mục được xây dựng lại (chưa cập nhật từng phần).
- Thư mục chỉ mục `.index/` dùng định dạng `pickle`, chỉ nên dùng thư mục do chính ứng dụng tạo ra.
- Khởi động mất vài chục giây vì phải nạp ba mô hình.
- Chưa có đăng nhập theo từng người dùng; chỉ có một khóa API dùng chung.

## Hướng phát triển

- Cập nhật chỉ mục từng phần khi thêm tài liệu mới.
- Lưu phiên hội thoại phía máy chủ và xác thực theo người dùng.
- Tách phần chạy mô hình ra máy chủ riêng để phục vụ nhiều người cùng lúc.
