"""Kiểm thử module độ đo trên nhãn giả có kết quả tính tay được."""

from __future__ import annotations

import numpy as np
import pytest

from vihallumt.eval import (
    apply_threshold,
    binary_metrics,
    blaser_to_hallucination_score,
    bootstrap_ci,
    mcnemar_test,
    severity_roc_auc,
    similarity_to_hallucination_score,
    tune_threshold,
)


# --------------------------------------------------------------------------
# binary_metrics
# --------------------------------------------------------------------------

def test_perfect_prediction():
    y = [0, 0, 1, 1]
    m = binary_metrics(y, y, score=[0.0, 0.1, 0.9, 1.0])
    assert m.mcc == pytest.approx(1.0)
    assert m.f1 == pytest.approx(1.0)
    assert m.roc_auc == pytest.approx(1.0)
    assert (m.tn, m.fp, m.fn, m.tp) == (2, 0, 0, 2)


def test_inverted_prediction_gives_mcc_minus_one():
    y_true = [0, 0, 1, 1]
    y_pred = [1, 1, 0, 0]
    assert binary_metrics(y_true, y_pred).mcc == pytest.approx(-1.0)


def test_mcc_matches_hand_computation():
    # TP=2, FN=1, TN=2, FP=1
    # MCC = (2*2 - 1*1) / sqrt(3*3*3*3) = 3/9
    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [1, 1, 0, 0, 0, 1]
    m = binary_metrics(y_true, y_pred)
    assert (m.tp, m.fn, m.tn, m.fp) == (2, 1, 2, 1)
    assert m.mcc == pytest.approx(1.0 / 3.0)
    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == pytest.approx(2 / 3)
    assert m.accuracy == pytest.approx(4 / 6)


def test_mcc_is_zero_for_constant_prediction():
    """Đoán bừa toàn 'không ảo giác' phải cho MCC = 0 dù accuracy rất cao.

    Đây chính là lý do P1 chọn MCC thay vì accuracy: HalOmi mất cân bằng tới
    mức 94% ở một số hướng dịch.
    """
    y_true = [0] * 95 + [1] * 5
    y_pred = [0] * 100
    m = binary_metrics(y_true, y_pred)
    assert m.accuracy == pytest.approx(0.95)
    assert m.mcc == pytest.approx(0.0)


def test_roc_auc_none_when_single_class():
    m = binary_metrics([0, 0, 0], [0, 0, 1], score=[0.1, 0.2, 0.3])
    assert m.roc_auc is None
    assert m.auprc is None


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        binary_metrics([0, 1], [0, 1, 1])


# --------------------------------------------------------------------------
# Chuyển đổi điểm
# --------------------------------------------------------------------------

def test_blaser_to_hallucination_score():
    """Công thức (1) của P2: HS = 1 - BLASER/5."""
    hs = blaser_to_hallucination_score([5.0, 2.5, 1.0])
    assert hs == pytest.approx([0.0, 0.5, 0.8])


def test_blaser_threshold_half_corresponds_to_blaser_2_5():
    """P2 dùng ngưỡng T = 0.5, tương đương BLASER = 2.5."""
    assert blaser_to_hallucination_score([2.5])[0] == pytest.approx(0.5)


def test_similarity_inversion_preserves_ranking():
    sim = [0.9, 0.5, 0.1]
    hs = similarity_to_hallucination_score(sim)
    assert list(np.argsort(hs)) == [0, 1, 2]  # tương đồng cao -> ảo giác thấp


# --------------------------------------------------------------------------
# tune_threshold
# --------------------------------------------------------------------------

def test_tune_threshold_finds_clean_separation():
    y_true = [0, 0, 0, 1, 1, 1]
    score = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    thr, f1 = tune_threshold(y_true, score)
    assert f1 == pytest.approx(1.0)
    assert 0.3 < thr <= 0.7
    assert list(apply_threshold(score, thr)) == y_true


def test_tune_threshold_requires_both_classes():
    with pytest.raises(ValueError):
        tune_threshold([1, 1, 1], [0.1, 0.2, 0.3])


def test_apply_threshold_is_inclusive_lower_bound():
    assert list(apply_threshold([0.4, 0.5, 0.6], 0.5)) == [0, 1, 1]


# --------------------------------------------------------------------------
# bootstrap_ci
# --------------------------------------------------------------------------

def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=300)
    y_pred = np.where(rng.random(300) < 0.8, y_true, 1 - y_true)
    point, lo, hi = bootstrap_ci(y_true, y_pred, n_resamples=300, seed=1)
    assert lo <= point <= hi
    assert lo < hi


def test_bootstrap_ci_narrows_with_more_data():
    rng = np.random.default_rng(0)

    def width(n: int) -> float:
        y_true = rng.integers(0, 2, size=n)
        y_pred = np.where(rng.random(n) < 0.8, y_true, 1 - y_true)
        _, lo, hi = bootstrap_ci(y_true, y_pred, n_resamples=300, seed=1)
        return hi - lo

    assert width(2000) < width(100)


def test_bootstrap_ci_is_deterministic_given_seed():
    y_true = [0, 1] * 50
    y_pred = [0, 1] * 40 + [1, 0] * 10
    a = bootstrap_ci(y_true, y_pred, n_resamples=200, seed=7)
    b = bootstrap_ci(y_true, y_pred, n_resamples=200, seed=7)
    assert a == b


# --------------------------------------------------------------------------
# mcnemar_test
# --------------------------------------------------------------------------

def test_mcnemar_identical_predictions():
    y = [0, 1] * 20
    r = mcnemar_test(y, y, y)
    assert r.p_value == pytest.approx(1.0)
    assert r.method == "identical"


def test_mcnemar_detects_clear_difference():
    """A đúng hết, B sai 30 mẫu -> phải rất có ý nghĩa thống kê."""
    y_true = [0] * 50 + [1] * 50
    pred_a = list(y_true)
    pred_b = [1] * 30 + [0] * 20 + [1] * 50
    r = mcnemar_test(y_true, pred_a, pred_b)
    assert r.n_only_a_correct == 30
    assert r.n_only_b_correct == 0
    assert r.p_value < 0.001


def test_mcnemar_uses_exact_test_for_small_discordant_counts():
    y_true = [0] * 50 + [1] * 50
    pred_a = list(y_true)
    pred_b = [1] * 3 + [0] * 47 + [1] * 50
    r = mcnemar_test(y_true, pred_a, pred_b)
    assert r.method == "exact-binomial"
    assert r.n_only_a_correct == 3


def test_mcnemar_symmetric_difference_is_not_significant():
    y_true = [0] * 50 + [1] * 50
    # A và B sai lệch nhau đối xứng -> không có bằng chứng ai hơn ai
    pred_a = [1] * 15 + [0] * 35 + [1] * 50
    pred_b = [0] * 50 + [0] * 15 + [1] * 35
    r = mcnemar_test(y_true, pred_a, pred_b)
    assert r.p_value > 0.05


# --------------------------------------------------------------------------
# severity_roc_auc
# --------------------------------------------------------------------------

def test_severity_perfect_ranking():
    sev = [0, 1, 2, 3]
    assert severity_roc_auc(sev, [0.1, 0.2, 0.3, 0.4]) == pytest.approx(1.0)


def test_severity_inverted_ranking():
    sev = [0, 1, 2, 3]
    assert severity_roc_auc(sev, [0.4, 0.3, 0.2, 0.1]) == pytest.approx(0.0)


def test_severity_all_ties_is_half():
    sev = [0, 1, 2, 3]
    assert severity_roc_auc(sev, [0.5] * 4) == pytest.approx(0.5)


def test_severity_requires_varied_labels():
    with pytest.raises(ValueError):
        severity_roc_auc([2, 2, 2], [0.1, 0.2, 0.3])


# --------------------------------------------------------------------------
# macro_average
# --------------------------------------------------------------------------

def test_macro_average_equals_pooled_when_groups_identical():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    groups = ["a", "a", "b", "b"]
    from vihallumt.eval import macro_average
    assert macro_average(y_true, y_pred, groups) == pytest.approx(1.0)


def test_macro_average_differs_from_pooled_under_group_imbalance():
    """Nhóm nhỏ được cân bằng như nhóm lớn -> khác hẳn MCC gộp.

    Đây chính là lý do con số của ta lệch với P1 cho tới khi phát hiện họ
    dùng trung bình vĩ mô theo hướng dịch.
    """
    from sklearn.metrics import matthews_corrcoef
    from vihallumt.eval import macro_average

    # Nhóm A: 100 mẫu, đoán hoàn hảo. Nhóm B: 10 mẫu, đoán ngược hoàn toàn.
    y_true = [0, 1] * 50 + [0, 1] * 5
    y_pred = [0, 1] * 50 + [1, 0] * 5
    groups = ["A"] * 100 + ["B"] * 10

    pooled = matthews_corrcoef(y_true, y_pred)
    macro = macro_average(y_true, y_pred, groups)
    assert macro == pytest.approx(0.0)      # (1 + (-1)) / 2
    assert pooled > 0.8                      # nhóm lớn lấn át
    assert macro < pooled


def test_macro_average_treats_single_class_group_as_zero():
    from vihallumt.eval import macro_average
    y_true = [0, 1, 0, 1] + [0, 0]
    y_pred = [0, 1, 0, 1] + [0, 0]
    groups = ["a"] * 4 + ["b"] * 2   # nhóm b chỉ có một lớp -> MCC = 0
    assert macro_average(y_true, y_pred, groups) == pytest.approx(0.5)


def test_macro_average_length_mismatch_raises():
    from vihallumt.eval import macro_average
    with pytest.raises(ValueError):
        macro_average([0, 1], [0, 1], ["a"])
