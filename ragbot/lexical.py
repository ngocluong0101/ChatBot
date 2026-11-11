"""BM25 tối giản với tokenizer bỏ dấu tiếng Việt.

Embedding (bge-m3) kém khi người dùng gõ không dấu ("mat khau toi thieu"), còn so khớp từ khóa trên
văn bản đã bỏ dấu thì không bị ảnh hưởng. BM25 bổ sung đúng điểm yếu đó cho tìm kiếm vector.
"""

import math
import re
import unicodedata
from collections import Counter

from langchain_core.documents import Document


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return stripped.replace("đ", "d").replace("Đ", "D")


def tokenize(text: str) -> list[str]:
    """Unigram + bigram trên văn bản bỏ dấu. Bigram giữ được cụm như "mat khau", "lam viec"."""
    words = re.findall(r"\w+", strip_accents(text).casefold())
    return words + [f"{a} {b}" for a, b in zip(words, words[1:], strict=False)]


class BM25Index:
    def __init__(self, docs: list[Document], k1: float = 1.5, b: float = 0.75) -> None:
        self._k1, self._b = k1, b
        self._term_freqs = [Counter(tokenize(d.page_content)) for d in docs]
        self._lengths = [sum(tf.values()) for tf in self._term_freqs]
        self._avg_len = (sum(self._lengths) / len(docs)) if docs else 0.0
        doc_freq: Counter[str] = Counter()
        for tf in self._term_freqs:
            doc_freq.update(tf.keys())
        n = len(docs)
        self._idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in doc_freq.items()}

    def scores(self, query: str) -> list[float]:
        terms = [t for t in set(tokenize(query)) if t in self._idf]
        result = []
        for tf, length in zip(self._term_freqs, self._lengths, strict=True):
            norm = self._k1 * (1 - self._b + self._b * length / self._avg_len) if self._avg_len else self._k1
            result.append(
                sum(self._idf[t] * tf[t] * (self._k1 + 1) / (tf[t] + norm) for t in terms if t in tf)
            )
        return result

    def top(self, query: str, k: int) -> list[tuple[int, float]]:
        """(chỉ số tài liệu, điểm) của k tài liệu tốt nhất, bỏ các tài liệu điểm 0."""
        ranked = sorted(enumerate(self.scores(query)), key=lambda item: item[1], reverse=True)
        return [(i, s) for i, s in ranked[:k] if s > 0]
