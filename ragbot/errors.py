"""Các lỗi nghiệp vụ mà CLI/API có thể bắt và hiển thị thân thiện cho người dùng."""


class RagError(Exception):
    """Lớp cha của mọi lỗi do ứng dụng chủ động ném ra."""


class NoDocumentsError(RagError):
    """Không tìm thấy (hoặc không đọc được nội dung của) tài liệu PDF nào."""


class LLMUnavailableError(RagError):
    """Không gọi được Ollama: chưa chạy, sai địa chỉ hoặc chưa tải model."""
