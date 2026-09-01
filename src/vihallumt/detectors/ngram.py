"""Bộ phát hiện ảo giác dao động bằng đếm lặp n-gram (Raunak et al., 2021).

Ảo giác *dao động* (oscillatory) là bản dịch lặp đi lặp lại một cụm từ. P2 §6.4
đo được rằng **58–86% ảo giác trong dữ liệu của họ thuộc loại này**, nên một
bộ đếm lặp rất đơn giản đã là baseline mạnh mà không tốn một phép nhân ma trận
nào.

Luật (theo P2 §6.4): gắn cờ ảo giác nếu số lần xuất hiện của n-gram phổ biến
nhất trong bản dịch **vượt** số lần của n-gram phổ biến nhất trong câu nguồn
quá một ngưỡng cho trước. P2 dùng n = 4 và ngưỡng = 2.

Bộ này không phát hiện được ảo giác *tách rời* (bản dịch trôi chảy nhưng nội
dung không liên quan) — đó chính là điểm mù được dùng trong phần Phân tích lỗi.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

from vihallumt.detectors.base import Detector, Pair

# Tách token thô: giữ chữ và số theo Unicode, nên hoạt động với cả tiếng Việt
# có dấu lẫn các hệ chữ viết khác trong HalOmi.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

#: Tham số của P2 §6.4
DEFAULT_N = 4
DEFAULT_THRESHOLD = 2


def tokenize(text: str) -> list[str]:
    """Tách token đơn giản, không phân biệt hoa thường."""
    return _TOKEN_RE.findall(text.lower())


def top_ngram_count(text: str, n: int = DEFAULT_N) -> int:
    """Số lần xuất hiện của n-gram phổ biến nhất.

    Câu ngắn hơn n token thì không có n-gram nào; quy ước trả về 0.
    """
    tokens = tokenize(text)
    if len(tokens) < n:
        return 0
    grams = Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return max(grams.values())


def repetition_excess(src_text: str, mt_text: str, n: int = DEFAULT_N) -> int:
    """Mức lặp thừa của bản dịch so với câu nguồn.

    Trừ đi mức lặp của câu nguồn để không phạt oan những câu nguồn vốn đã lặp
    (ví dụ danh sách liệt kê).
    """
    return top_ngram_count(mt_text, n) - top_ngram_count(src_text, n)


class NGramRepetitionDetector(Detector):
    """Phát hiện ảo giác dao động bằng mức lặp n-gram thừa.

    Điểm trả về là **số nguyên** (mức lặp thừa), không phải xác suất. Nó vẫn
    xếp hạng được nên tính ROC-AUC hợp lệ, nhưng nhiều giá trị trùng nhau —
    phần lớn câu bình thường đều có điểm 0.
    """

    produces_continuous_score = True

    def __init__(self, n: int = DEFAULT_N, threshold: int = DEFAULT_THRESHOLD) -> None:
        self.n = n
        self.threshold = threshold
        self.name = f"top-{n}-gram repetition"

    def score(self, pairs: list[Pair]) -> np.ndarray:
        return np.array(
            [repetition_excess(p.src_text, p.mt_text, self.n) for p in pairs],
            dtype=float,
        )

    def predict(self, pairs: list[Pair]) -> np.ndarray:
        """Dự đoán nhị phân dùng thẳng ngưỡng của P2, không cần hiệu chỉnh."""
        return (self.score(pairs) > self.threshold).astype(int)
