"""Lấy mẫu phân tầng để chọn câu đưa đi gán nhãn tay.

Vì sao không lấy mẫu ngẫu nhiên
-------------------------------
Ảo giác là hiện tượng hiếm. Nếu bốc ngẫu nhiên 700 cặp từ đầu ra của một hệ
dịch bình thường, ta sẽ được khoảng 690 mẫu âm và vài mẫu dương — không đủ để
ước lượng bất kỳ độ đo nào một cách đáng tin, mà vẫn tốn trọn 10 giờ công gán
nhãn.

HalOmi giải bài toán này bằng ba tầng lấy mẫu (P1 Appendix A.3, Figure 4):

  ``uniform``  bốc ngẫu nhiên  -> giữ tính đại diện của phân bố thật
  ``biased``   thiên về vùng điểm cao -> tăng mật độ ca khó, ca biên
  ``worst``    lấy phần tệ nhất -> bảo đảm có đủ mẫu dương

Module này tái tạo đúng ba tầng đó.

Cảnh báo khi diễn giải kết quả
------------------------------
Tỉ lệ ảo giác trên tập đã lấy mẫu phân tầng **không phải** tỉ lệ ảo giác thật
của hệ dịch — nó cao hơn nhiều theo chủ ý. Muốn ước lượng tỉ lệ thật thì chỉ
được dùng tầng ``uniform``. Điều này phải nêu rõ trong báo cáo; P1 cũng liệt kê
đây là một hạn chế của HalOmi.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

#: Tỉ trọng mặc định của ba tầng.
DEFAULT_SHARES: dict[str, float] = {
    "uniform": 0.30,
    "biased": 0.40,
    "worst": 0.30,
}

#: Vùng phân vị dùng cho tầng `biased` — khoảng giữa-trên, nơi tập trung các ca
#: mà detector phân vân. Đây chính là chỗ nhãn người có giá trị nhất, vì nếu
#: detector đã chắc chắn thì gán nhãn tay chẳng thêm được thông tin gì.
BIASED_QUANTILE_RANGE: tuple[float, float] = (0.55, 0.95)


def aggregate_score(df: pd.DataFrame, score_cols: Sequence[str]) -> np.ndarray:
    """Gộp nhiều điểm detector thành một điểm ảo giác duy nhất.

    Dùng **trung bình thứ hạng** thay vì trung bình giá trị thô, vì các detector
    có thang đo hoàn toàn khác nhau (BLASER ~ [-5, -1], cosine ~ [-1, 0]); lấy
    trung bình trực tiếp sẽ để detector có thang rộng nhất lấn át tất cả.

    Returns:
        Mảng trong [0, 1]; 1 = bị mọi detector đánh giá là nhiều ảo giác nhất.
    """
    if not score_cols:
        raise ValueError("Can it nhat mot cot diem")
    missing = [c for c in score_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Thieu cot diem: {missing}")

    ranks = []
    for col in score_cols:
        s = df[col]
        if s.notna().sum() == 0:
            continue
        # rank trung bình cho các giá trị bằng nhau; NaN đẩy về giữa thang
        r = s.rank(method="average", pct=True, na_option="keep")
        ranks.append(r.fillna(0.5).to_numpy(dtype=float))

    if not ranks:
        raise ValueError("Moi cot diem deu rong")
    return np.mean(ranks, axis=0)


def stratified_sample(
    df: pd.DataFrame,
    score_cols: Sequence[str],
    n_total: int,
    shares: dict[str, float] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Chọn `n_total` dòng theo ba tầng uniform / biased / worst.

    Args:
        df: kho câu ứng viên, đã có sẵn điểm của các detector rẻ tiền.
        score_cols: các cột điểm dùng để xếp hạng (quy ước cao = ảo giác).
        n_total: tổng số dòng cần lấy.
        shares: tỉ trọng ba tầng; mặc định `DEFAULT_SHARES`.
        seed: hạt giống ngẫu nhiên.

    Returns:
        DataFrame con, thêm hai cột `selection` (tên tầng) và `agg_score`.
        Không có dòng nào bị lấy hai lần.

    Raises:
        ValueError: nếu `n_total` lớn hơn số dòng có sẵn, hoặc tỉ trọng sai.
    """
    if n_total <= 0:
        raise ValueError("n_total phai duong")
    if n_total > len(df):
        raise ValueError(f"Can {n_total} dong nhung kho chi co {len(df)}")

    shares = shares or DEFAULT_SHARES
    if abs(sum(shares.values()) - 1.0) > 1e-6:
        raise ValueError(f"Tong ti trong phai bang 1, dang la {sum(shares.values())}")
    unknown = set(shares) - {"uniform", "biased", "worst"}
    if unknown:
        raise ValueError(f"Tang khong hop le: {sorted(unknown)}")

    rng = np.random.default_rng(seed)

    work = df.copy().reset_index(drop=True)
    work["agg_score"] = aggregate_score(work, score_cols)

    n_worst = int(round(n_total * shares.get("worst", 0.0)))
    n_biased = int(round(n_total * shares.get("biased", 0.0)))
    n_uniform = n_total - n_worst - n_biased

    taken: set[int] = set()
    chunks: list[pd.DataFrame] = []

    # -- tầng worst: điểm cao nhất ------------------------------------------
    if n_worst > 0:
        idx = work["agg_score"].nlargest(n_worst).index
        chunk = work.loc[idx].copy()
        chunk["selection"] = "worst"
        chunks.append(chunk)
        taken.update(idx)

    # -- tầng biased: dải phân vị giữa-trên ---------------------------------
    if n_biased > 0:
        remaining = work.drop(index=list(taken))
        lo, hi = remaining["agg_score"].quantile(BIASED_QUANTILE_RANGE).to_list()
        band = remaining[(remaining["agg_score"] >= lo) & (remaining["agg_score"] <= hi)]
        # Dải quá hẹp (nhiều điểm bằng nhau) thì mở rộng ra toàn bộ phần còn lại
        pool = band if len(band) >= n_biased else remaining
        idx = rng.choice(pool.index.to_numpy(), size=min(n_biased, len(pool)),
                         replace=False)
        chunk = work.loc[idx].copy()
        chunk["selection"] = "biased"
        chunks.append(chunk)
        taken.update(idx)

    # -- tầng uniform: ngẫu nhiên phần còn lại ------------------------------
    if n_uniform > 0:
        remaining = work.drop(index=list(taken))
        idx = rng.choice(remaining.index.to_numpy(),
                         size=min(n_uniform, len(remaining)), replace=False)
        chunk = work.loc[idx].copy()
        chunk["selection"] = "uniform"
        chunks.append(chunk)

    out = pd.concat(chunks).sort_index()
    if out.index.duplicated().any():
        raise AssertionError("Co dong bi lay hai lan — loi logic lay mau")
    return out.reset_index(drop=True)


def selection_report(sample: pd.DataFrame) -> pd.DataFrame:
    """Thống kê mô tả cho từng tầng, để đưa vào phần Dataset của báo cáo."""
    if "selection" not in sample.columns:
        raise KeyError("Thieu cot 'selection' — hay chay stratified_sample truoc")

    agg: dict[str, tuple] = {
        "n": ("agg_score", "size"),
        "mean_agg_score": ("agg_score", "mean"),
        "min_agg_score": ("agg_score", "min"),
        "max_agg_score": ("agg_score", "max"),
    }
    if "label" in sample.columns:
        agg["pct_hallucination"] = ("label", lambda s: 100.0 * s.mean())

    return sample.groupby("selection").agg(**agg).round(4)
