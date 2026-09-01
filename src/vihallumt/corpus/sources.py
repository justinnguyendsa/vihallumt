"""Nạp ngữ liệu song ngữ Anh–Việt làm nguyên liệu cho ViHalluMT.

Tình trạng truy cập các nguồn (kiểm tra ngày 2026-09-01)
-------------------------------------------------------

======================  ==========  ==========================================
Nguồn                   Trạng thái  Ghi chú
======================  ==========  ==========================================
OPUS-100 ``en-vi``      mở          1M train / 2k dev / 2k test. Phụ đề, web —
                                    nhiễu, nhiều câu ngắn, **dễ gây ảo giác**.
IWSLT'15 ``en-vi``      mở          133k cặp. TED Talks — câu sạch, trang trọng.
FLORES-200 / plus       **gated**   Cần đăng nhập HF. Xem hướng dẫn bên dưới.
PhoMT                   thủ công    Phải điền form của VinAI.
======================  ==========  ==========================================

Mở khoá FLORES (khuyến khích, không bắt buộc)
---------------------------------------------
FLORES đáng dùng vì HalOmi cũng lấy câu nguồn từ đó, nên kết quả tiếng Việt so
sánh trực tiếp được với 18 hướng dịch của HalOmi. Cách mở khoá:

1. Vào https://huggingface.co/datasets/openlanguagedata/flores_plus và bấm
   đồng ý điều khoản (miễn phí, duyệt ngay).
2. Lấy token ở https://huggingface.co/settings/tokens.
3. ``huggingface-cli login`` hoặc đặt biến môi trường ``HF_TOKEN``.

Không có FLORES thì đường ống vẫn chạy đầy đủ với OPUS-100 + IWSLT; chỉ mất
tính so sánh trực tiếp với HalOmi, và điều này phải ghi vào mục *Limitations*.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

#: Định danh dataset trên HuggingFace.
OPUS100_EN_VI = ("Helsinki-NLP/opus-100", "en-vi")
IWSLT15_EN_VI = "thainq107/iwslt2015-en-vi"
FLORES_PLUS = "openlanguagedata/flores_plus"

#: Giới hạn độ dài (số từ) cho câu đưa vào ngữ liệu.
#: Câu quá ngắn không có chỗ cho ảo giác xảy ra; câu quá dài thì gán nhãn tay
#: rất mệt và làm chậm mọi thứ.
MIN_WORDS = 5
MAX_WORDS = 60

_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class SourceStats:
    """Thống kê quá trình lọc, để đưa vào phần Dataset của báo cáo."""

    name: str
    n_raw: int
    n_after_filter: int

    @property
    def kept_pct(self) -> float:
        return 100.0 * self.n_after_filter / max(1, self.n_raw)


def normalise(text: str) -> str:
    """Chuẩn hoá khoảng trắng và Unicode cho tiếng Việt.

    Dùng NFC vì tiếng Việt có hai cách mã hoá cùng một chữ (dựng sẵn so với tổ
    hợp). Nếu không chuẩn hoá, hai chuỗi trông hệt nhau trên màn hình lại không
    bằng nhau khi so sánh, làm hỏng cả việc khử trùng lặp lẫn việc tách token.
    """
    import unicodedata

    return _WS.sub(" ", unicodedata.normalize("NFC", str(text))).strip()


def is_usable(en: str, vi: str) -> bool:
    """Cặp câu có dùng được không."""
    if not en or not vi:
        return False
    n_en, n_vi = len(en.split()), len(vi.split())
    if not (MIN_WORDS <= n_en <= MAX_WORDS):
        return False
    if not (MIN_WORDS <= n_vi <= MAX_WORDS):
        return False
    # Lệch độ dài quá lớn thường là cặp căn chỉnh sai, không phải bản dịch thật
    if not (0.4 <= n_vi / n_en <= 2.5):
        return False
    # Câu giống hệt nhau ở hai phía: gần như chắc chắn là lỗi căn chỉnh
    if en.lower() == vi.lower():
        return False
    return True


def _to_frame(rows: list[tuple[str, str]], source: str) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["en", "vi"]).assign(source=source)


def clean_pairs(
    pairs: list[tuple[str, str]],
    source: str,
) -> tuple[pd.DataFrame, SourceStats]:
    """Chuẩn hoá, lọc và khử trùng lặp một danh sách cặp câu."""
    n_raw = len(pairs)
    cleaned = [(normalise(en), normalise(vi)) for en, vi in pairs]
    kept = [(en, vi) for en, vi in cleaned if is_usable(en, vi)]

    df = _to_frame(kept, source).drop_duplicates(subset=["en", "vi"])
    # Khử luôn trùng lặp phía tiếng Anh: cùng một câu nguồn xuất hiện nhiều lần
    # sẽ làm lệch việc lấy mẫu và tạo rò rỉ giữa tập dev và test.
    df = df.drop_duplicates(subset=["en"]).reset_index(drop=True)

    return df, SourceStats(source, n_raw, len(df))


def load_opus100(split: str = "test", limit: int | None = None) -> tuple[pd.DataFrame, SourceStats]:
    """Nạp OPUS-100 ``en-vi``. Miền nhiễu -> tỉ lệ ảo giác tự nhiên cao hơn."""
    from datasets import load_dataset

    name, config = OPUS100_EN_VI
    ds = load_dataset(name, config, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    pairs = [(r["translation"]["en"], r["translation"]["vi"]) for r in ds]
    return clean_pairs(pairs, "opus100")


def load_iwslt15(split: str = "train", limit: int | None = None) -> tuple[pd.DataFrame, SourceStats]:
    """Nạp IWSLT'15 ``en-vi`` (TED Talks). Câu sạch, văn phong trang trọng."""
    from datasets import load_dataset

    ds = load_dataset(IWSLT15_EN_VI, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    pairs = [(r["en"], r["vi"]) for r in ds]
    return clean_pairs(pairs, "iwslt15")


def load_flores(split: str = "dev", limit: int | None = None) -> tuple[pd.DataFrame, SourceStats]:
    """Nạp FLORES-200 ``eng_Latn`` / ``vie_Latn`` và ghép theo ``id``.

    Raises:
        RuntimeError: kèm hướng dẫn mở khoá, nếu chưa đăng nhập HuggingFace.
    """
    from datasets import load_dataset

    try:
        en = load_dataset(FLORES_PLUS, "eng_Latn", split=split)
        vi = load_dataset(FLORES_PLUS, "vie_Latn", split=split)
    except Exception as exc:  # gated, chưa có token, hoặc lỗi mạng
        raise RuntimeError(
            "Khong nap duoc FLORES-200. Day la dataset gated: hay dong y dieu "
            f"khoan tai https://huggingface.co/datasets/{FLORES_PLUS} roi dang "
            "nhap bang `huggingface-cli login`. Duong ong van chay duoc voi "
            "OPUS-100 + IWSLT neu bo qua FLORES."
        ) from exc

    en_by_id = {r["id"]: r["text"] for r in en}
    pairs = [(en_by_id[r["id"]], r["text"]) for r in vi if r["id"] in en_by_id]
    if limit:
        pairs = pairs[:limit]
    return clean_pairs(pairs, "flores")


def build_source_pool(
    n_per_source: int = 1500,
    use_flores: bool = True,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[SourceStats]]:
    """Gộp các nguồn thành một kho câu nguồn duy nhất.

    Trộn nhiều miền văn bản là có chủ ý: OPUS-100 (phụ đề, nhiễu) sinh ra
    nhiều ảo giác hơn, IWSLT (TED Talks, sạch) cho lớp âm khó, FLORES (Wikipedia,
    đa song song) bảo đảm so sánh được với HalOmi. Chỉ dùng một miền sẽ khiến
    kết luận không khái quát hoá được.

    Args:
        n_per_source: số cặp lấy từ mỗi nguồn sau khi lọc.
        use_flores: thử nạp FLORES; bỏ qua trong im lặng nếu bị chặn.
        seed: hạt giống để lấy mẫu tái lập được.
    """
    frames: list[pd.DataFrame] = []
    stats: list[SourceStats] = []

    loaders = [
        ("opus100", lambda: load_opus100("test", limit=n_per_source * 4)),
        ("iwslt15", lambda: load_iwslt15("train", limit=n_per_source * 4)),
    ]
    if use_flores:
        loaders.append(("flores", lambda: load_flores("dev")))

    for name, loader in loaders:
        try:
            df, st = loader()
        except Exception as exc:
            stats.append(SourceStats(f"{name} (that bai: {type(exc).__name__})", 0, 0))
            continue
        if len(df) > n_per_source:
            df = df.sample(n=n_per_source, random_state=seed)
        frames.append(df)
        stats.append(st)

    if not frames:
        raise RuntimeError("Khong nap duoc nguon nao. Kiem tra ket noi mang.")

    pool = pd.concat(frames, ignore_index=True)
    pool = pool.drop_duplicates(subset=["en"]).reset_index(drop=True)
    pool.insert(0, "sent_id", [f"s{i:06d}" for i in range(len(pool))])
    return pool, stats
