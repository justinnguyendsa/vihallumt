"""Kiểm thử lấy mẫu phân tầng và bộ probe tiếng Việt."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vihallumt.corpus.probe_vi import (
    ALL_PROBE_ITEMS,
    PHENOMENA,
    build_probe_frame,
    probe_delta,
    probe_summary,
)
from vihallumt.corpus.sampling import (
    DEFAULT_SHARES,
    aggregate_score,
    selection_report,
    stratified_sample,
)
from vihallumt.data import resolve_halomi_path


@pytest.fixture
def pool() -> pd.DataFrame:
    """Kho ứng viên giả với hai detector có thang đo rất khác nhau."""
    rng = np.random.default_rng(0)
    n = 400
    latent = rng.random(n)  # mức ảo giác tiềm ẩn
    return pd.DataFrame({
        "src_text": [f"source {i}" for i in range(n)],
        "mt_text": [f"target {i}" for i in range(n)],
        # thang [-5, -1], giống BLASER đã đảo dấu
        "score_a": -5 + 4 * latent + rng.normal(0, 0.1, n),
        # thang [-1, 0], giống cosine đã đảo dấu
        "score_b": -1 + latent + rng.normal(0, 0.05, n),
        "label": (latent > 0.75).astype(int),
    })


# ==========================================================================
# aggregate_score
# ==========================================================================

def test_aggregate_score_is_in_unit_range(pool):
    s = aggregate_score(pool, ["score_a", "score_b"])
    assert s.min() >= 0.0 and s.max() <= 1.0


def test_aggregate_score_is_not_dominated_by_widest_scale(pool):
    """Trung bình thứ hạng, không phải trung bình giá trị thô.

    Nếu lấy trung bình thô thì detector thang [-5,-1] sẽ lấn át hoàn toàn
    detector thang [-1,0]; dùng thứ hạng nên hai detector đóng góp ngang nhau.
    """
    both = aggregate_score(pool, ["score_a", "score_b"])
    only_a = aggregate_score(pool, ["score_a"])
    only_b = aggregate_score(pool, ["score_b"])
    # Điểm gộp phải tương quan mạnh với CẢ HAI, không chỉ với thang rộng
    assert np.corrcoef(both, only_a)[0, 1] > 0.9
    assert np.corrcoef(both, only_b)[0, 1] > 0.9


def test_aggregate_score_correlates_with_ground_truth(pool):
    s = aggregate_score(pool, ["score_a", "score_b"])
    assert s[pool["label"] == 1].mean() > s[pool["label"] == 0].mean()


def test_aggregate_score_handles_missing_values(pool):
    pool = pool.copy()
    pool.loc[:50, "score_a"] = np.nan
    s = aggregate_score(pool, ["score_a", "score_b"])
    assert np.isfinite(s).all()


def test_aggregate_score_requires_columns(pool):
    with pytest.raises(KeyError):
        aggregate_score(pool, ["khong_ton_tai"])
    with pytest.raises(ValueError):
        aggregate_score(pool, [])


# ==========================================================================
# stratified_sample
# ==========================================================================

def test_sample_has_requested_size(pool):
    out = stratified_sample(pool, ["score_a", "score_b"], n_total=100)
    assert len(out) == 100


def test_sample_covers_all_three_strata(pool):
    out = stratified_sample(pool, ["score_a", "score_b"], n_total=100)
    assert set(out["selection"]) == {"uniform", "biased", "worst"}


def test_sample_respects_share_proportions(pool):
    out = stratified_sample(pool, ["score_a", "score_b"], n_total=100)
    counts = out["selection"].value_counts()
    assert counts["worst"] == int(round(100 * DEFAULT_SHARES["worst"]))
    assert counts["biased"] == int(round(100 * DEFAULT_SHARES["biased"]))


def test_sample_never_repeats_a_row(pool):
    out = stratified_sample(pool, ["score_a", "score_b"], n_total=200)
    assert out["src_text"].is_unique


def test_worst_stratum_has_highest_scores(pool):
    out = stratified_sample(pool, ["score_a", "score_b"], n_total=150)
    means = out.groupby("selection")["agg_score"].mean()
    assert means["worst"] > means["biased"] > means["uniform"]


def test_worst_stratum_enriches_hallucinations(pool):
    """Mục đích tồn tại của tầng `worst`: tăng mật độ mẫu dương."""
    out = stratified_sample(pool, ["score_a", "score_b"], n_total=150)
    rates = out.groupby("selection")["label"].mean()
    assert rates["worst"] > rates["uniform"]


def test_sample_is_reproducible(pool):
    a = stratified_sample(pool, ["score_a", "score_b"], n_total=100, seed=3)
    b = stratified_sample(pool, ["score_a", "score_b"], n_total=100, seed=3)
    assert a["src_text"].tolist() == b["src_text"].tolist()


def test_different_seeds_change_random_strata(pool):
    a = stratified_sample(pool, ["score_a", "score_b"], n_total=100, seed=1)
    b = stratified_sample(pool, ["score_a", "score_b"], n_total=100, seed=2)
    assert a["src_text"].tolist() != b["src_text"].tolist()


def test_sample_larger_than_pool_raises(pool):
    with pytest.raises(ValueError, match="kho chi co"):
        stratified_sample(pool, ["score_a"], n_total=len(pool) + 1)


def test_invalid_shares_raise(pool):
    with pytest.raises(ValueError, match="Tong ti trong"):
        stratified_sample(pool, ["score_a"], n_total=10,
                          shares={"uniform": 0.5, "worst": 0.2})
    with pytest.raises(ValueError, match="Tang khong hop le"):
        stratified_sample(pool, ["score_a"], n_total=10,
                          shares={"uniform": 0.5, "khac": 0.5})


def test_single_stratum_configuration_works(pool):
    out = stratified_sample(pool, ["score_a"], n_total=50,
                            shares={"uniform": 0.0, "biased": 0.0, "worst": 1.0})
    assert set(out["selection"]) == {"worst"}


def test_selection_report_shape(pool):
    out = stratified_sample(pool, ["score_a", "score_b"], n_total=120)
    rep = selection_report(out)
    assert set(rep.index) == {"uniform", "biased", "worst"}
    assert rep["n"].sum() == 120
    assert "pct_hallucination" in rep.columns


def test_selection_report_requires_selection_column(pool):
    with pytest.raises(KeyError):
        selection_report(pool)


# -- xác nhận chiến lược trên dữ liệu thật ----------------------------------

def _halomi_available() -> bool:
    try:
        resolve_halomi_path()
        return True
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not _halomi_available(), reason="Chua tai HalOmi")
def test_strategy_works_on_real_halomi_data():
    """Kiểm chứng chiến lược lấy mẫu trên dữ liệu có nhãn thật.

    HalOmi có sẵn cả nhãn người lẫn điểm detector, nên ta xác nhận được rằng
    ba tầng thật sự tạo ra chênh lệch mật độ ảo giác **trước khi** đem chiến
    lược này áp lên dữ liệu tiếng Việt chưa có nhãn.
    """
    from vihallumt.data import load_halomi

    df = load_halomi()
    out = stratified_sample(
        df, ["score_labse", "score_blaser2_qe", "score_sonar_cosine"],
        n_total=700, seed=42,
    )
    rates = out.groupby("selection")["label"].mean()
    assert rates["worst"] > rates["biased"] > rates["uniform"]
    # Tầng worst phải giàu mẫu dương hơn hẳn phân bố gốc
    assert rates["worst"] > df["label"].mean() * 1.5


# ==========================================================================
# probe_vi
# ==========================================================================

def test_probe_covers_all_six_phenomena():
    assert {i.phenomenon for i in ALL_PROBE_ITEMS} == set(PHENOMENA)


def test_probe_has_enough_items_per_phenomenon():
    from collections import Counter
    counts = Counter(i.phenomenon for i in ALL_PROBE_ITEMS)
    for phenomenon in PHENOMENA:
        assert counts[phenomenon] >= 5, f"{phenomenon} chi co {counts[phenomenon]} muc"


def test_probe_frame_has_two_rows_per_item():
    df = build_probe_frame()
    assert len(df) == 2 * len(ALL_PROBE_ITEMS)
    assert set(df["variant"]) == {"correct", "corrupted"}


def test_probe_frame_is_balanced():
    """Cặp tối thiểu -> đúng 50% mẫu dương, trừ các cặp đối chứng âm."""
    df = build_probe_frame()
    assert (df["variant"] == "correct").sum() == (df["variant"] == "corrupted").sum()


def test_correct_variant_always_has_label_zero():
    df = build_probe_frame()
    assert (df.loc[df["variant"] == "correct", "label"] == 0).all()


def test_corrupted_and_correct_texts_differ():
    for item in ALL_PROBE_ITEMS:
        assert item.mt_correct != item.mt_corrupted, item.note


def test_every_item_has_an_explanatory_note():
    """Ghi chú là thứ đưa thẳng vào mục Phân tích lỗi của báo cáo."""
    for item in ALL_PROBE_ITEMS:
        assert len(item.note) > 20, item.mt_correct


def test_negative_control_pair_exists():
    """Cần ít nhất một cặp mà 'bản sai' thật ra vẫn đúng nghĩa.

    Nếu thiếu, ta không phân biệt được detector đang thật sự hiểu ngữ nghĩa
    hay chỉ đơn thuần phạt mọi thay đổi bề mặt.
    """
    controls = [i for i in ALL_PROBE_ITEMS if i.severity == 0]
    assert controls, "Thieu cap doi chung am"


def test_severity_within_halomi_scale():
    for item in ALL_PROBE_ITEMS:
        assert 0 <= item.severity <= 3


def test_all_sources_are_english_and_targets_vietnamese():
    df = build_probe_frame()
    assert (df["src"] == "EN").all()
    assert (df["tgt"] == "VI").all()
    assert (df["direction"] == "EN-VI").all()


# -- phân tích Δ ------------------------------------------------------------

def test_probe_delta_computes_difference():
    df = build_probe_frame()
    # Detector giả: chấm bản sai cao hơn đúng 0.3 điểm
    df["score"] = np.where(df["variant"] == "corrupted", 0.8, 0.5)
    deltas = probe_delta(df, "score")
    assert len(deltas) == len(ALL_PROBE_ITEMS)
    assert np.allclose(deltas["delta"], 0.3)


def test_probe_summary_flags_a_blind_spot():
    """Detector cho điểm y hệt hai bản -> Δ = 0 -> điểm mù."""
    df = build_probe_frame()
    df["score"] = 0.5  # hoàn toàn mù
    summary = probe_summary(df, "score")
    assert (summary["mean_delta"] == 0).all()
    assert (summary["pct_ranked_correctly"] == 0).all()


def test_probe_summary_reports_per_phenomenon():
    df = build_probe_frame()
    df["score"] = np.where(df["variant"] == "corrupted", 1.0, 0.0)
    summary = probe_summary(df, "score")
    assert set(summary.index) == set(PHENOMENA)
    assert (summary["pct_ranked_correctly"] == 100.0).all()


def test_probe_delta_requires_both_variants():
    df = build_probe_frame()
    df = df[df["variant"] == "correct"].copy()
    df["score"] = 0.5
    with pytest.raises(ValueError, match="Thieu bien the"):
        probe_delta(df, "score")
