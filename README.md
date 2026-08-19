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
├── chatbot.py
├── papers/
│   ├── 01_noi_quy_nhan_su.pdf
│   ├── 02_chinh_sach_bao_mat_thong_tin.pdf
│   ├── 03_quy_dinh_lam_viec_tu_xa.pdf
│   ├── 04_quy_trinh_nghi_phep_cong_tac.pdf
│   └── 05_quy_tac_ung_xu_doanh_nghiep.pdf
├── .env
└── README.md
```

## Installation

Tạo môi trường ảo và cài các package được source sử dụng:

```bash
python -m venv .venv
source .venv/bin/activate
pip install langchain langchain-community langchain-experimental \
	langchain-text-splitters langchain-core langchain-huggingface \
	langchain-ollama faiss-cpu sentence-transformers "unstructured[pdf]"
```

Cài Ollama theo hướng dẫn tại [ollama.com](https://ollama.com), sau đó tải model:

```bash
ollama pull llama3.2:3b
```

Model embedding `BAAI/bge-m3` sẽ được HuggingFace tải khi chạy lần đầu. File `.env` hiện không chứa cấu hình bắt buộc và source không đọc biến môi trường.

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
- Vector store được tạo lại mỗi lần khởi động; chưa có cơ chế lưu index hoặc cập nhật tài liệu tăng dần.
- Chất lượng kết quả phụ thuộc vào nội dung PDF, quá trình trích xuất và model local.

## Future Improvements

- Lưu và tái sử dụng FAISS index thay vì tạo lại sau mỗi lần chạy.
- Thêm giao diện web và lịch sử hội thoại.
- Cải thiện citation để hiển thị chính xác trang và tên tài liệu.
- Bổ sung xử lý lỗi, logging và cấu hình model qua biến môi trường.
