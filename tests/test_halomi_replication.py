"""Kiểm chứng bộ nạp HalOmi tái tạo đúng số liệu công bố trong P1.

Đây là *cổng nghiệm thu* cho phần "cài đặt được paper": nếu mọi con số dưới
đây khớp thì ta chắc chắn đang tiền xử lý dữ liệu giống hệt Benkirane et al.
(2024), và mọi kết quả về sau đều so sánh được trực tiếp với bảng của họ.

Nguồn đối chiếu: P1 Appendix A — Bảng 1 (mức nghiêm trọng), Bảng 2 (phân bố
nhị phân tập validation), Bảng 3 (phân bố nhị phân tập test).
"""

from __future__ import annotations

import pytest

from sklearn.metrics import roc_auc_score

from vihallumt.data import (
    halomi_blaser_raw,
    load_halomi,
    resolve_halomi_path,
    split_halomi,
)


def _data_available() -> bool:
    try:
        resolve_halomi_path()
        return True
    except FileNotFoundError:
        return False


pytestmark = pytest.mark.skipif(
    not _data_available(),
    reason="Chua tai HalOmi — chay `bash scripts/download_halomi.sh`",
)


@pytest.fixture(scope="module")
def halomi():
    return load_halomi()


@pytest.fixture(scope="module")
def splits(halomi):
    return split_halomi(halomi)


# --------------------------------------------------------------------------
# Kích thước tập dữ liệu
# --------------------------------------------------------------------------

def test_natural_subset_size(halomi):
    """HalOmi có 2.865 cặp tự nhiên (P1 §2.1 lọc bỏ bản nhiễu loạn)."""
    assert len(halomi) == 2865


def test_perturbed_rows_are_excluded(halomi):
    assert (halomi["perturbation"] == "natural").all()


def test_eighteen_translation_directions(halomi):
    assert halomi["direction"].nunique() == 18


def test_validation_and_test_sizes(splits):
    """P1 §2.1: validation = DE<->EN (301 câu), test = 16 hướng còn lại."""
    val, test = splits
    assert len(val) == 301
    assert len(test) == 2564  # P1 báo 2.558 sau khi bỏ 6 câu bị lọc nội dung nhạy cảm
    assert test["direction"].nunique() == 16


def test_validation_is_exactly_german_english(splits):
    val, _ = splits
    assert set(val["direction"]) == {"DE-EN", "EN-DE"}
    assert (val["direction"] == "DE-EN").sum() == 155
    assert (val["direction"] == "EN-DE").sum() == 146


# --------------------------------------------------------------------------
# Bảng 1 của P1 — phân bố mức nghiêm trọng
# --------------------------------------------------------------------------

def test_validation_severity_distribution(splits):
    """P1 Bảng 1, khối trên: 272 No / 5 Small / 4 Partial / 20 Full."""
    val, _ = splits
    counts = val["severity"].value_counts().sort_index().tolist()
    assert counts == [272, 5, 4, 20]


def test_test_severity_distribution(splits):
    """P1 Bảng 1, khối dưới: 1859 No / 220 Small / 287 Partial / 198 Full."""
    _, test = splits
    counts = test["severity"].value_counts().sort_index().tolist()
    assert counts == [1859, 220, 287, 198]


# --------------------------------------------------------------------------
# Bảng 2 của P1 — phân bố nhị phân, tập validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "direction, n_total, n_no, n_hall",
    [
        ("DE-EN", 155, 140, 15),
        ("EN-DE", 146, 132, 14),
    ],
)
def test_validation_binary_distribution(splits, direction, n_total, n_no, n_hall):
    val, _ = splits
    sub = val[val["direction"] == direction]
    assert len(sub) == n_total
    assert (sub["label"] == 0).sum() == n_no
    assert (sub["label"] == 1).sum() == n_hall


def test_validation_binary_totals(splits):
    val, _ = splits
    assert (val["label"] == 0).sum() == 272
    assert (val["label"] == 1).sum() == 29


# --------------------------------------------------------------------------
# Bảng 3 của P1 — phân bố nhị phân, tập test (toàn bộ 16 hướng)
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "direction, n_total, n_no, n_hall",
    [
        ("EN-AR", 144, 136, 8),
        ("AR-EN", 156, 132, 24),
        ("EN-RU", 146, 141, 5),
        ("RU-EN", 158, 146, 12),
        ("EN-ES", 153, 131, 22),
        ("ES-EN", 160, 127, 33),
        ("EN-ZH", 160, 131, 29),
        ("ZH-EN", 159, 127, 32),
        ("EN-KS", 184, 111, 73),
        ("KS-EN", 151, 89, 62),
        ("EN-YO", 195, 166, 29),
        ("YO-EN", 146, 124, 22),
        ("EN-MN", 197, 78, 119),
        ("MN-EN", 152, 43, 109),
        ("ES-YO", 151, 97, 54),
        ("YO-ES", 152, 80, 72),
    ],
)
def test_test_binary_distribution(splits, direction, n_total, n_no, n_hall):
    _, test = splits
    sub = test[test["direction"] == direction]
    assert len(sub) == n_total, f"{direction}: so cau lech"
    assert (sub["label"] == 0).sum() == n_no, f"{direction}: so cau khong ao giac lech"
    assert (sub["label"] == 1).sum() == n_hall, f"{direction}: so cau ao giac lech"


def test_test_binary_totals(splits):
    """P1 Bảng 3, dòng Total: 1859 không ảo giác / 705 có ảo giác."""
    _, test = splits
    assert (test["label"] == 0).sum() == 1859
    assert (test["label"] == 1).sum() == 705


def test_class_imbalance_matches_paper_limitations(splits):
    """P1 mục Limitations: HRL lệch 79–94% về phía 'không ảo giác'."""
    _, test = splits
    hrl = test[test["resource_level"] == "HRL"]
    share_no = hrl.groupby("direction")["label"].apply(lambda s: (s == 0).mean())
    assert 0.79 <= share_no.min() <= 0.99
    assert share_no.max() >= 0.94


# --------------------------------------------------------------------------
# Điểm số tính sẵn — nền tảng cho baseline BLASER-2.0-QE
# --------------------------------------------------------------------------

def test_blaser_scores_are_complete(halomi):
    """Không có NaN -> tái lập được baseline SOTA của P1 mà không cần fairseq2."""
    assert halomi["score_blaser2_qe"].notna().all()


def test_blaser_raw_scores_in_valid_range(halomi):
    """Sau khi khôi phục dấu, BLASER 2.0 nằm trên thang 1..5."""
    raw = halomi_blaser_raw(halomi)
    assert 1.0 <= raw.min() <= 1.5
    assert 5.0 <= raw.max() <= 5.5


@pytest.mark.parametrize(
    "col",
    ["score_blaser2_qe", "score_labse", "score_laser", "score_sonar_cosine",
     "score_comet_qe", "score_xnli"],
)
def test_precomputed_scores_are_oriented_as_hallucination_scores(halomi, col):
    """Mọi cột `score_*` của HalOmi đã đảo dấu sẵn: cao = ảo giác nhiều.

    Chốt quy ước này bằng test để tránh lỗi đảo dấu hai lần — loại lỗi âm thầm
    biến một detector tốt thành detector tệ hơn cả đoán bừa.
    """
    s = halomi[col]
    ok = s.notna()
    assert ok.sum() > 0, f"{col} rong"
    assert roc_auc_score(halomi.loc[ok, "label"], s[ok]) > 0.5, (
        f"{col} co ve bi dao dau: AUC < 0.5 khi dung nguyen trang"
    )
    assert s[halomi["label"] == 1].mean() > s[halomi["label"] == 0].mean()


def test_labse_is_competitive_with_blaser_on_natural_data(halomi):
    """Kiểm chứng sớm luận điểm của P1: embedding cạnh tranh được với BLASER-QE."""
    auc_labse = roc_auc_score(halomi["label"], halomi["score_labse"])
    auc_blaser = roc_auc_score(halomi["label"], halomi["score_blaser2_qe"])
    assert auc_labse > 0.75
    assert auc_blaser > 0.75


# --------------------------------------------------------------------------
# Phân nhóm hướng dịch theo mức tài nguyên (P1 Figure 2)
# --------------------------------------------------------------------------

def test_direction_groups_cover_paper_figure2(splits):
    _, test = splits
    assert set(test["direction_group"]) == {
        "EN->HRL", "HRL->EN", "EN->LRL", "LRL->EN", "ES->LRL", "LRL->ES",
    }


def test_spanish_yoruba_counted_as_low_resource(halomi):
    """Hai hướng phi-Anh ES<->YO được P1 xếp vào nhóm LRL."""
    sub = halomi[halomi["direction"].isin({"ES-YO", "YO-ES"})]
    assert (sub["resource_level"] == "LRL").all()
