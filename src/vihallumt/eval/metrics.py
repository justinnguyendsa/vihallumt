"""Độ đo đánh giá cho bài toán phát hiện ảo giác.

Quy ước xuyên suốt đồ án
------------------------
* `y_true`: 0 = không ảo giác, 1 = có ảo giác.
* `score` : **điểm ảo giác** — càng CAO càng nhiều khả năng ảo giác.
  Các độ đo tương đồng (cosine, BLASER) đi ngược chiều nên phải đảo dấu
  trước khi đưa vào đây; dùng `similarity_to_hallucination_score()`.

Độ đo chính là **MCC** (Matthews Correlation Coefficient), theo P1 §2.2: bộ
dữ liệu mất cân bằng nặng (79–94% không ảo giác ở HRL) nên accuracy và F1
đều dễ gây hiểu nhầm.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

ArrayLike = Sequence[int] | Sequence[float] | np.ndarray


# --------------------------------------------------------------------------
# Chuyển đổi điểm
# --------------------------------------------------------------------------

def similarity_to_hallucination_score(sim: ArrayLike) -> np.ndarray:
    """Đảo chiều điểm tương đồng thành điểm ảo giác (cao = ảo giác nhiều)."""
    return -np.asarray(sim, dtype=float)


def blaser_to_hallucination_score(blaser: ArrayLike) -> np.ndarray:
    """HS(x, y) = 1 - BLASER(x, y) / 5 — công thức (1) của P2.

    BLASER cho điểm trong khoảng 1–5 (5 = tương đương ngữ nghĩa hoàn toàn),
    nên HS nằm trong khoảng 0–0.8 với đầu vào hợp lệ.
    """
    return 1.0 - np.asarray(blaser, dtype=float) / 5.0


# --------------------------------------------------------------------------
# Độ đo nhị phân
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BinaryMetrics:
    """Bộ độ đo cho phân loại nhị phân. `mcc` là độ đo chính."""

    mcc: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    roc_auc: float | None
    auprc: float | None
    tn: int
    fp: int
    fn: int
    tp: int
    n: int
    n_positive: int

    def to_dict(self) -> dict:
        return asdict(self)


def binary_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    score: ArrayLike | None = None,
) -> BinaryMetrics:
    """Tính toàn bộ độ đo nhị phân.

    Args:
        y_true: nhãn đúng (0/1).
        y_pred: nhãn dự đoán (0/1).
        score:  điểm ảo giác liên tục, cao = ảo giác. Nếu None thì bỏ qua
            ROC-AUC và AUPRC. P1 chỉ lấy nhãn cứng từ LLM nên không tính
            được hai độ đo này; ta khắc phục bằng logit-scoring.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Kich thuoc lech nhau: {y_true.shape} vs {y_pred.shape}")

    # labels=[0,1] để ma trận luôn 2x2 kể cả khi một lớp vắng mặt
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    roc_auc: float | None = None
    auprc: float | None = None
    if score is not None:
        score = np.asarray(score, dtype=float)
        # ROC-AUC không xác định khi chỉ có một lớp
        if len(np.unique(y_true)) == 2:
            roc_auc = float(roc_auc_score(y_true, score))
            auprc = float(average_precision_score(y_true, score))

    return BinaryMetrics(
        mcc=float(matthews_corrcoef(y_true, y_pred)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        roc_auc=roc_auc,
        auprc=auprc,
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        n=int(len(y_true)),
        n_positive=int(y_true.sum()),
    )


def macro_average(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    groups: Sequence,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = matthews_corrcoef,
    min_group_size: int = 1,
) -> float:
    """Trung bình vĩ mô một độ đo trên từng nhóm (mặc định: MCC theo hướng dịch).

    **P1 báo cáo MCC theo kiểu này**, chứ không phải MCC gộp toàn tập test.
    Điều đó không được nói rõ trong phần thân bài, chỉ ngụ ý ở caption Figure 2
    ("MCC average score across ... directions"). Kiểm chứng thực nghiệm trên
    BLASER-2.0-QE: trung bình vĩ mô cho 0.374 tổng thể / 0.466 riêng HRL, khớp
    số công bố 0.38 / 0.46; trong khi MCC gộp chỉ cho 0.317 — lệch hẳn.

    Vì vậy **mọi so sánh với bảng của P1 phải dùng hàm này**, không dùng MCC gộp.

    Nhóm nào chỉ còn một lớp nhãn thì MCC không xác định, quy ước tính là 0 —
    giống cách `sklearn` xử lý.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    groups = np.asarray(groups)
    if not (len(y_true) == len(y_pred) == len(groups)):
        raise ValueError("y_true, y_pred va groups phai cung do dai")

    scores: list[float] = []
    for g in np.unique(groups):
        mask = groups == g
        if mask.sum() < min_group_size:
            continue
        yt, yp = y_true[mask], y_pred[mask]
        scores.append(0.0 if len(np.unique(yt)) < 2 else float(metric_fn(yt, yp)))

    if not scores:
        raise ValueError("Khong co nhom nao du lon de tinh trung binh")
    return float(np.mean(scores))


# --------------------------------------------------------------------------
# Hiệu chỉnh ngưỡng
# --------------------------------------------------------------------------

def tune_threshold(y_true: ArrayLike, score: ArrayLike) -> tuple[float, float]:
    """Tìm ngưỡng cực đại hoá F1 trên đường precision-recall.

    Đây đúng là quy trình P1 §2.4 dùng cho các bộ phát hiện bằng embedding:
    ngưỡng được chọn trên tập validation (DE<->EN) rồi áp nguyên si sang test.

    Returns:
        (ngưỡng, F1 đạt được tại ngưỡng đó). Dự đoán là `score >= ngưỡng`.
    """
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    if len(np.unique(y_true)) < 2:
        raise ValueError("Can ca hai lop de hieu chinh nguong")

    precision, recall, thresholds = precision_recall_curve(y_true, score)
    # precision/recall dài hơn thresholds đúng 1 phần tử
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 2 * precision * recall / (precision + recall)
    f1 = np.nan_to_num(f1[:-1], nan=0.0)
    if len(f1) == 0:
        raise ValueError("Duong precision-recall rong")

    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def apply_threshold(score: ArrayLike, threshold: float) -> np.ndarray:
    """Nhị phân hoá điểm ảo giác: `score >= threshold` -> 1."""
    return (np.asarray(score, dtype=float) >= threshold).astype(int)


# --------------------------------------------------------------------------
# Khoảng tin cậy bootstrap
# --------------------------------------------------------------------------

def bootstrap_ci(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = matthews_corrcoef,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Khoảng tin cậy percentile bootstrap cho một độ đo.

    P1 không báo cáo khoảng tin cậy, nên chênh lệch vài điểm MCC giữa các mô
    hình trong bảng của họ khó biết có ý nghĩa hay không. Ta bổ sung phần này.

    Returns:
        (giá trị điểm, cận dưới, cận trên).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    point = float(metric_fn(y_true, y_pred))

    stats_boot = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        # Mẫu bootstrap có thể chỉ còn một lớp -> MCC không xác định, coi là 0
        if len(np.unique(yt)) < 2:
            stats_boot[i] = 0.0
        else:
            stats_boot[i] = metric_fn(yt, yp)

    alpha = (1.0 - confidence) / 2.0
    lo = float(np.percentile(stats_boot, 100 * alpha))
    hi = float(np.percentile(stats_boot, 100 * (1 - alpha)))
    return point, lo, hi


# --------------------------------------------------------------------------
# Kiểm định ý nghĩa thống kê
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class McNemarResult:
    """Kết quả kiểm định McNemar giữa hai bộ phân loại trên cùng tập dữ liệu."""

    n_only_a_correct: int   # b: A đúng, B sai
    n_only_b_correct: int   # c: A sai, B đúng
    statistic: float
    p_value: float
    method: str

    def to_dict(self) -> dict:
        return asdict(self)


def mcnemar_test(
    y_true: ArrayLike,
    pred_a: ArrayLike,
    pred_b: ArrayLike,
    exact_threshold: int = 25,
) -> McNemarResult:
    """Kiểm định McNemar: A và B có khác nhau đáng kể không?

    Dùng để trả lời "hệ tốt nhất có thực sự hơn baseline, hay chỉ là nhiễu".
    Với b + c nhỏ thì dùng kiểm định nhị thức chính xác; ngược lại dùng
    xấp xỉ chi-bình-phương có hiệu chỉnh liên tục.
    """
    y_true = np.asarray(y_true, dtype=int)
    a_ok = np.asarray(pred_a, dtype=int) == y_true
    b_ok = np.asarray(pred_b, dtype=int) == y_true

    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))

    if b + c == 0:
        return McNemarResult(b, c, 0.0, 1.0, "identical")

    if b + c < exact_threshold:
        p = float(stats.binomtest(b, b + c, 0.5).pvalue)
        return McNemarResult(b, c, float(min(b, c)), p, "exact-binomial")

    statistic = (abs(b - c) - 1.0) ** 2 / (b + c)
    p = float(stats.chi2.sf(statistic, df=1))
    return McNemarResult(b, c, float(statistic), p, "chi2-continuity-corrected")


# --------------------------------------------------------------------------
# Severity ranking (ablation của P1, Appendix B)
# --------------------------------------------------------------------------

def severity_roc_auc(y_severity: ArrayLike, score: ArrayLike) -> float:
    """ROC-AUC thích ứng cho xếp hạng mức nghiêm trọng nhiều lớp.

    P1 Appendix B: tính tỉ lệ cặp câu có nhãn khác nhau bị xếp hạng SAI, rồi
    lấy 1 trừ đi. Cài đặt ở đây tương đương và chính là hệ số Somers' D chuẩn
    hoá về [0, 1] — đúng bằng thống kê c của Harrell.
    """
    y = np.asarray(y_severity, dtype=float)
    s = np.asarray(score, dtype=float)
    if len(y) != len(s):
        raise ValueError("Kich thuoc lech nhau")

    # So sánh mọi cặp có mức nghiêm trọng khác nhau
    dy = y[:, None] - y[None, :]
    ds = s[:, None] - s[None, :]
    mask = dy != 0
    if not mask.any():
        raise ValueError("Can it nhat hai muc nghiem trong khac nhau")

    concordant = np.sum((dy * ds > 0) & mask)
    ties = np.sum((ds == 0) & mask)
    total = np.sum(mask)
    return float((concordant + 0.5 * ties) / total)
