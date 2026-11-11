FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/data/hf \
    RAG_INDEX_DIR=/data/index \
    RAG_PAPERS_DIR=/app/papers

# poppler/tesseract: unstructured đọc PDF (tesseract-ocr-vie cho PDF scan tiếng Việt).
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils tesseract-ocr tesseract-ocr-vie libmagic1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# Cài torch/torchvision bản CPU trước để pip không kéo bản CUDA (vài GB). Hai gói phải khớp phiên
# bản với nhau (và với requirements.txt) nếu không sentence-transformers không import được.
# Muốn dùng GPU trong container thì bỏ dòng này và dùng base image có CUDA.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.10.0 torchvision==0.25.0 \
    && pip install -r requirements.txt

COPY ragbot ./ragbot
COPY chatbot.py .
COPY papers ./papers

# /data giữ model HuggingFace đã tải và FAISS index: mount volume để không tải/dựng lại mỗi lần.
RUN useradd --create-home app && mkdir -p /data && chown -R app /data
USER app
VOLUME /data
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

CMD ["uvicorn", "ragbot.api:app", "--host", "0.0.0.0", "--port", "8000"]
