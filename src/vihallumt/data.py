"""Nạp và chuẩn hoá bộ dữ liệu HalOmi (Dale et al., 2023).

HalOmi là benchmark gốc mà paper P1 (Benkirane et al., 2024) đánh giá trên đó.
Module này tái tạo đúng quy trình tiền xử lý của P1:

  1. Chỉ giữ bản dịch *natural* (bỏ `perturbed`) — P1 §2.1.
  2. Nhị phân hoá nhãn: `No` -> 0, {`Small`, `Partial`, `Full`} -> 1 — P1 §2.2.
  3. Chia tập: DE<->EN làm `validation` (chọn prompt + hiệu chỉnh ngưỡng),
     16 hướng còn lại làm `test` — P1 §2.1.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# --- Ánh xạ mã ngôn ngữ FLORES-200 sang chữ viết tắt dùng trong P1 (Appendix A.1) ---
LANG_ACRONYM: dict[str, str] = {
    "arb_Arab": "AR",
    "zho_Hans": "ZH",
    "eng_Latn": "EN",
    "deu_Latn": "DE",
    "kas_Deva": "KS",
    "mni_Beng": "MN",
    "rus_Cyrl": "RU",
    "spa_Latn": "ES",
    "yor_Latn": "YO",
}

HRL: frozenset[str] = frozenset({"AR", "ZH", "DE", "RU", "ES"})
LRL: frozenset[str] = frozenset({"KS", "MN", "YO"})

# Thứ tự 4 mức nghiêm trọng của HalOmi (dùng cho ablation severity ranking, P1 Appendix B)
SEVERITY_ORDER: tuple[str, ...] = (
    "1_No_hallucination",
    "2_Small_hallucination",
    "3_Partial_hallucination",
    "4_Full_hallucination",
)
SEVERITY_TO_INT: dict[str, int] = {name: i for i, name in enumerate(SEVERITY_ORDER)}

# Cặp ngôn ngữ dùng làm tập validation trong P1 (§2.1)
VALIDATION_DIRECTIONS: frozenset[str] = frozenset({"DE-EN", "EN-DE"})

# Các cột điểm số đã được HalOmi tính sẵn. Quan trọng: nhờ có sẵn
# `score_blaser2_qe` mà ta tái lập được baseline SOTA của P1 mà không cần
# cài `fairseq2`/`sonar-space` (vốn hay xung đột phiên bản trên Colab).
#
# CẢNH BÁO VỀ DẤU: mọi cột `score_*` trong HalOmi đã được lưu sẵn dưới dạng
# *điểm ảo giác* — càng CAO càng nhiều khả năng ảo giác. Với các độ đo tương
# đồng thì đó là giá trị đã đảo dấu, ví dụ `score_labse` nằm trong khoảng
# [-1, 0.216] chứ không phải cosine thô. Quy ước này trùng khớp với quy ước
# của `vihallumt.eval.metrics`, nên dùng thẳng được, KHÔNG đảo dấu lần nữa.
# Muốn lấy lại BLASER thô (thang 1..5) thì gọi `halomi_blaser_raw()`.
PRECOMPUTED_SCORES: tuple[str, ...] = (
    "score_blaser2_qe",
    "score_labse",
    "score_laser",
    "score_sonar_cosine",
    "score_comet_qe",
    "score_xnli",
    "score_log_loss",
    "score_alti_mean",
    "score_alti_t_mean",
    "score_attn_ot",
)

HALOMI_URL = "https://dl.fbaipublicfiles.com/nllb/halomi_release_v2.zip"


def _direction_group(src: str, tgt: str) -> str:
    """Nhóm hướng dịch theo cách P1 gộp trong Figure 2."""
    if src == "EN" and tgt in HRL:
        return "EN->HRL"
    if src in HRL and tgt == "EN":
        return "HRL->EN"
    if src == "EN" and tgt in LRL:
        return "EN->LRL"
    if src in LRL and tgt == "EN":
        return "LRL->EN"
    if src == "ES" and tgt in LRL:
        return "ES->LRL"
    if src in LRL and tgt == "ES":
        return "LRL->ES"
    return f"{src}->{tgt}"


def _resource_level(src: str, tgt: str) -> str:
    """HRL/LRL xác định theo phía *không phải* tiếng Anh của cặp dịch.

    Hai hướng phi-Anh (ES<->YO) được P1 xếp vào LRL vì phía YO là ngôn ngữ
    tài nguyên thấp.
    """
    non_en = {src, tgt} - {"EN"}
    if non_en & LRL:
        return "LRL"
    return "HRL"


def resolve_halomi_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Tìm `halomi_full.tsv`, thử các vị trí thông dụng nếu không được chỉ định."""
    if path is not None:
        p = Path(path)
        if p.is_dir():
            p = p / "halomi_full.tsv"
        if not p.exists():
            raise FileNotFoundError(f"Khong tim thay HalOmi tai {p}")
        return p

    candidates = [
        Path("data/raw/halomi_full.tsv"),
        Path("../data/raw/halomi_full.tsv"),
        Path(__file__).resolve().parents[2] / "data" / "raw" / "halomi_full.tsv",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Khong tim thay halomi_full.tsv. Chay `bash scripts/download_halomi.sh` truoc, "
        f"hoac tai tay tu {HALOMI_URL}"
    )


def load_halomi(
    path: str | os.PathLike[str] | None = None,
    natural_only: bool = True,
) -> pd.DataFrame:
    """Nạp HalOmi và bổ sung các cột chuẩn hoá dùng xuyên suốt đồ án.

    Args:
        path: đường dẫn tới `halomi_full.tsv` hoặc thư mục chứa nó.
        natural_only: chỉ giữ bản dịch tự nhiên, bỏ bản đã bị nhiễu loạn.
            P1 §2.1 làm đúng như vậy vì kết luận rút từ dữ liệu nhiễu loạn
            nhân tạo có thể không áp dụng được cho ảo giác tự nhiên.

    Returns:
        DataFrame với các cột bổ sung:
          `src`, `tgt`        — viết tắt ngôn ngữ (EN, DE, ...)
          `direction`         — ví dụ "EN-DE"
          `direction_group`   — ví dụ "EN->HRL"
          `resource_level`    — "HRL" hoặc "LRL"
          `severity`          — 0..3
          `label`             — 0 = không ảo giác, 1 = có ảo giác
          `split`             — "validation" (DE<->EN) hoặc "test"
    """
    tsv = resolve_halomi_path(path)
    df = pd.read_csv(tsv, sep="\t")

    if natural_only:
        df = df[df["perturbation"] == "natural"].copy()

    df["src"] = df["src_lang"].map(LANG_ACRONYM)
    df["tgt"] = df["tgt_lang"].map(LANG_ACRONYM)
    unknown = df[df["src"].isna() | df["tgt"].isna()]
    if len(unknown):
        raise ValueError(f"Ma ngon ngu chua biet: {set(unknown['src_lang']) | set(unknown['tgt_lang'])}")

    df["direction"] = df["src"] + "-" + df["tgt"]
    df["direction_group"] = [_direction_group(s, t) for s, t in zip(df["src"], df["tgt"])]
    df["resource_level"] = [_resource_level(s, t) for s, t in zip(df["src"], df["tgt"])]

    df["severity"] = df["class_hall"].map(SEVERITY_TO_INT)
    if df["severity"].isna().any():
        raise ValueError(f"Nhan class_hall la: {sorted(df['class_hall'].unique())}")
    df["severity"] = df["severity"].astype(int)

    # Nhị phân hoá: mọi mức ảo giác đều tính là dương (P1 §2.2)
    df["label"] = (df["severity"] > 0).astype(int)

    df["split"] = ["validation" if d in VALIDATION_DIRECTIONS else "test" for d in df["direction"]]

    return df.reset_index(drop=True)


def split_halomi(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tách thành (validation, test) theo đúng quy ước của P1."""
    val = df[df["split"] == "validation"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    return val, test


def halomi_blaser_raw(df: pd.DataFrame) -> pd.Series:
    """Khôi phục điểm BLASER-2.0-QE thô (thang 1..5) từ cột đã đảo dấu.

    Cần đến khi muốn áp công thức HS = 1 - BLASER/5 với ngưỡng T = 0.5 của P2,
    vì công thức đó định nghĩa trên thang gốc chứ không phải giá trị đã đảo dấu.
    """
    return -df["score_blaser2_qe"]
