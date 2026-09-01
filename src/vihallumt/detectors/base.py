"""Giao diện chung cho mọi bộ phát hiện ảo giác.

Quy ước: `score()` trả về **điểm ảo giác** — càng cao càng nhiều khả năng có
ảo giác. Quy ước này thống nhất với `vihallumt.eval.metrics` và với cách
HalOmi lưu sẵn các cột `score_*`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Pair:
    """Một cặp câu cần đánh giá."""

    src_text: str
    mt_text: str
    src_lang: str = "EN"
    tgt_lang: str = "VI"


def pairs_from_frame(df: pd.DataFrame) -> list[Pair]:
    """Chuyển DataFrame kiểu HalOmi / ViHalluMT thành danh sách `Pair`."""
    required = {"src_text", "mt_text"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Thieu cot: {sorted(missing)}")

    src_langs = df["src"] if "src" in df.columns else ["EN"] * len(df)
    tgt_langs = df["tgt"] if "tgt" in df.columns else ["VI"] * len(df)

    return [
        Pair(str(s), str(m), str(sl), str(tl))
        for s, m, sl, tl in zip(df["src_text"], df["mt_text"], src_langs, tgt_langs)
    ]


class Detector(ABC):
    """Lớp cơ sở cho bộ phát hiện ảo giác."""

    #: Tên hiển thị trong bảng kết quả.
    name: str = "detector"

    #: True nếu `score()` trả về điểm liên tục (tính được ROC-AUC).
    #: False nếu chỉ trả về 0/1 — khi đó ROC-AUC vô nghĩa.
    produces_continuous_score: bool = True

    @abstractmethod
    def score(self, pairs: list[Pair]) -> np.ndarray:
        """Chấm điểm ảo giác cho từng cặp. Cao = nhiều khả năng ảo giác."""

    def score_frame(self, df: pd.DataFrame) -> np.ndarray:
        """Tiện ích: chấm điểm trực tiếp trên DataFrame."""
        return self.score(pairs_from_frame(df))

    def __repr__(self) -> str:  # pragma: no cover - chỉ để gỡ lỗi
        return f"{type(self).__name__}(name={self.name!r})"
