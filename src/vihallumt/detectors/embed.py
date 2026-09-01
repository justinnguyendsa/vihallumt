"""Bộ phát hiện ảo giác bằng độ tương đồng embedding đa ngữ (P1 §2.4).

Ý tưởng: bản dịch bị ảo giác nằm xa câu nguồn trong không gian embedding đa
ngữ. Chấm điểm bằng ``-cos(emb(src), emb(mt))``, ngưỡng hiệu chỉnh trên tập
validation bằng cách cực đại hoá F1.

P1 dùng các không gian embedding thương mại (OpenAI, Cohere, Mistral) cùng với
SONAR. Đồ án thay bằng các mô hình mở tương đương, và bổ sung embedding chuyên
tiếng Việt — phần P1 không có.
"""

from __future__ import annotations

import numpy as np

from vihallumt.detectors.base import Detector, Pair

#: Các không gian embedding dùng trong đồ án.
#: Cột "vai trò" ghi rõ mô hình nào thay cho mô hình nào của P1.
EMBEDDING_MODELS: dict[str, str] = {
    # Đa ngữ, mở — thay cho các embedding thương mại của P1
    "LaBSE": "sentence-transformers/LaBSE",
    "mE5-large": "intfloat/multilingual-e5-large",
    "mpnet-multilingual": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    # Chuyên tiếng Việt — bổ sung của đồ án
    "vi-bi-encoder": "bkai-foundation-models/vietnamese-bi-encoder",
    "vi-embedding": "dangvantuan/vietnamese-embedding",
}

#: Mô hình yêu cầu thêm tiền tố vào câu đầu vào (họ E5).
_PREFIX_REQUIRED = {"intfloat/multilingual-e5-large": "query: "}

#: Mô hình được huấn luyện trên văn bản tiếng Việt đã tách từ.
_NEEDS_WORD_SEGMENTATION = {
    "bkai-foundation-models/vietnamese-bi-encoder",
    "dangvantuan/vietnamese-embedding",
}


def segment_vietnamese(text: str) -> str:
    """Tách từ tiếng Việt bằng `underthesea`, nối bằng dấu gạch dưới.

    PhoBERT và các mô hình dẫn xuất được huấn luyện trên văn bản đã tách từ
    ("hà_nội" thay vì "hà nội"), nên bỏ qua bước này sẽ làm chất lượng
    embedding giảm rõ rệt. Nếu chưa cài `underthesea` thì trả nguyên văn bản
    và để lời gọi bên ngoài tự quyết định.
    """
    try:
        from underthesea import word_tokenize
    except ImportError:
        return text
    return " ".join(word_tokenize(text, format="text").split())


class EmbeddingDetector(Detector):
    """Chấm điểm ảo giác bằng ``-cos(emb(src), emb(mt))``.

    Args:
        model_id: định danh mô hình trên HuggingFace.
        display_name: tên hiển thị trong bảng kết quả.
        segment_vi: tách từ tiếng Việt trước khi mã hoá. Mặc định `None` =
            tự bật cho các mô hình cần.
        batch_size: kích thước lô khi mã hoá.
    """

    produces_continuous_score = True

    def __init__(
        self,
        model_id: str = "sentence-transformers/LaBSE",
        display_name: str | None = None,
        segment_vi: bool | None = None,
        batch_size: int = 64,
        model: object | None = None,
    ) -> None:
        self.model_id = model_id
        self.name = display_name or model_id.split("/")[-1]
        self.batch_size = batch_size
        self.segment_vi = (
            model_id in _NEEDS_WORD_SEGMENTATION if segment_vi is None else segment_vi
        )
        self.prefix = _PREFIX_REQUIRED.get(model_id, "")
        self._model = model

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id)

    def _prepare(self, text: str, lang: str) -> str:
        if self.segment_vi and lang.upper() == "VI":
            text = segment_vietnamese(text)
        return self.prefix + text

    def score(self, pairs: list[Pair]) -> np.ndarray:
        if not pairs:
            return np.array([], dtype=float)
        self._ensure_loaded()

        src = [self._prepare(p.src_text, p.src_lang) for p in pairs]
        mt = [self._prepare(p.mt_text, p.tgt_lang) for p in pairs]

        emb_src = self._model.encode(
            src, batch_size=self.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        emb_mt = self._model.encode(
            mt, batch_size=self.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )

        cosine = np.sum(emb_src * emb_mt, axis=1)
        return -cosine  # cao = ảo giác
