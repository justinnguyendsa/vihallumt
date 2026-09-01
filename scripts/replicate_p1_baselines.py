"""Tái lập các baseline phi-LLM của P1 trên HalOmi.

Quy trình đúng theo P1 §2.4:
  1. Hiệu chỉnh ngưỡng trên tập validation (DE<->EN) bằng cách cực đại hoá F1
     trên đường precision-recall.
  2. Áp NGUYÊN ngưỡng đó sang tập test (16 hướng còn lại).
  3. Báo cáo MCC **trung bình vĩ mô theo hướng dịch** — xem ghi chú bên dưới.

GHI CHÚ QUAN TRỌNG VỀ CÁCH TỔNG HỢP
-----------------------------------
P1 không nói rõ trong thân bài rằng MCC của họ là trung bình theo hướng dịch;
chỉ caption Figure 2 ngụ ý ("MCC average score across ... directions"). Ta xác
định được điều này bằng thực nghiệm trên BLASER-2.0-QE:

    cách tổng hợp        tổng thể   riêng HRL
    MCC gộp (pooled)       0.317      0.477
    MCC trung bình vĩ mô   0.374      0.466     <- khớp paper
    P1 công bố             0.38       0.46

Vì vậy cột `mcc_macro` là cột dùng để so với P1; `mcc_pooled` giữ lại để tham
khảo vì nó phản ánh hiệu năng thực tế trên một tập dữ liệu trộn lẫn.

Số đối chiếu trong P1:
  BLASER-2.0-QE  tổng thể MCC = 0.38, riêng HRL = 0.46
  (mô hình tốt nhất của họ, Llama3-70B, đạt 0.43 tổng thể và 0.63 ở HRL)

Chạy:  python scripts/replicate_p1_baselines.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from vihallumt.data import load_halomi, split_halomi
from vihallumt.eval import (
    apply_threshold,
    binary_metrics,
    bootstrap_ci,
    macro_average,
    mcnemar_test,
    tune_threshold,
)

RESULTS = ROOT / "results"

# Các detector có điểm tính sẵn trong HalOmi (đã theo quy ước cao = ảo giác).
DETECTORS: dict[str, str] = {
    "score_blaser2_qe": "BLASER-2.0-QE",
    "score_sonar_cosine": "SONAR cosine",
    "score_labse": "LaBSE cosine",
    "score_laser": "LASER cosine",
    "score_comet_qe": "COMET-QE",
    "score_xnli": "XNLI",
    "score_alti_mean": "ALTI+",
    "score_log_loss": "Seq log-loss",
}

# Con số P1 công bố, dùng để đối chiếu (MCC).
PAPER_MCC: dict[str, dict[str, float]] = {
    "BLASER-2.0-QE": {"overall": 0.38, "HRL": 0.46},
}


def evaluate_detector(
    val: pd.DataFrame,
    test: pd.DataFrame,
    score_col: str,
) -> tuple[dict, pd.Series]:
    """Hiệu chỉnh ngưỡng trên val, đánh giá trên test. Trả về (tổng hợp, dự đoán)."""
    v_ok = val[score_col].notna()
    threshold, val_f1 = tune_threshold(val.loc[v_ok, "label"], val.loc[v_ok, score_col])

    pred = pd.Series(apply_threshold(test[score_col].fillna(test[score_col].median()),
                                     threshold), index=test.index)

    m = binary_metrics(test["label"], pred, score=test[score_col])
    _, lo, hi = bootstrap_ci(test["label"], pred, n_resamples=1000, seed=42)

    row = {
        "threshold": round(threshold, 4),
        "val_f1": round(val_f1, 4),
        # Cột chính để so với P1
        "mcc_macro": round(macro_average(test["label"], pred, test["direction"]), 4),
        # Cột tham khảo: hiệu năng trên tập trộn lẫn
        "mcc_pooled": round(m.mcc, 4),
        "mcc_pooled_ci_low": round(lo, 4),
        "mcc_pooled_ci_high": round(hi, 4),
        "f1": round(m.f1, 4),
        "precision": round(m.precision, 4),
        "recall": round(m.recall, 4),
        "accuracy": round(m.accuracy, 4),
        "roc_auc": round(m.roc_auc, 4) if m.roc_auc is not None else None,
    }

    # MCC trung bình vĩ mô tách theo mức tài nguyên và nhóm hướng dịch (P1 Figure 2)
    for level in ("HRL", "LRL"):
        mask = (test["resource_level"] == level).to_numpy()
        row[f"mcc_{level}"] = round(
            macro_average(test.loc[mask, "label"], pred[mask], test.loc[mask, "direction"]), 4
        )

    for group in ("EN->HRL", "HRL->EN", "EN->LRL", "LRL->EN", "ES->LRL", "LRL->ES"):
        mask = (test["direction_group"] == group).to_numpy()
        if mask.sum() > 0:
            row[f"mcc_{group}"] = round(
                macro_average(test.loc[mask, "label"], pred[mask], test.loc[mask, "direction"]), 4
            )

    return row, pred


def main() -> int:
    RESULTS.mkdir(exist_ok=True)

    df = load_halomi()
    val, test = split_halomi(df)

    print("=" * 84)
    print("TÁI LẬP BASELINE PHI-LLM CỦA P1 TRÊN HALOMI")
    print("=" * 84)
    print(f"Hiệu chỉnh ngưỡng trên: {len(val)} câu DE<->EN  |  Đánh giá trên: {len(test)} câu")
    print(f"Tỉ lệ ảo giác trong tập test: {100 * test['label'].mean():.1f}%")
    print()

    rows, preds = [], {}
    for col, name in DETECTORS.items():
        if col not in test.columns or test[col].isna().all():
            print(f"  [bỏ qua] {name}: không có dữ liệu")
            continue
        row, pred = evaluate_detector(val, test, col)
        row["detector"] = name
        rows.append(row)
        preds[name] = pred

    res = pd.DataFrame(rows).set_index("detector").sort_values("mcc_macro", ascending=False)
    res.to_csv(RESULTS / "halomi_baselines.csv")

    print("-" * 84)
    print("KẾT QUẢ CHÍNH — MCC trung bình vĩ mô theo hướng dịch (cách P1 tổng hợp)")
    print("-" * 84)
    show = res[["mcc_macro", "mcc_HRL", "mcc_LRL", "mcc_pooled",
                "f1", "roc_auc", "threshold"]]
    print(show.to_string())
    print()

    print("-" * 84)
    print("ĐỐI CHIẾU VỚI SỐ CÔNG BỐ TRONG P1")
    print("-" * 84)
    for name, expected in PAPER_MCC.items():
        if name not in res.index:
            continue
        got_all = res.loc[name, "mcc_macro"]
        got_hrl = res.loc[name, "mcc_HRL"]
        pooled = res.loc[name, "mcc_pooled"]
        d_all = abs(got_all - expected["overall"])
        d_hrl = abs(got_hrl - expected["HRL"])
        print(f"  {name}")
        print(f"    tổng thể (macro) : ta {got_all:.3f}  |  paper {expected['overall']:.2f}"
              f"  |  lệch {d_all:.3f}  {'KHỚP' if d_all <= 0.02 else 'LỆCH'}")
        print(f"    riêng HRL (macro): ta {got_hrl:.3f}  |  paper {expected['HRL']:.2f}"
              f"  |  lệch {d_hrl:.3f}  {'KHỚP' if d_hrl <= 0.02 else 'LỆCH'}")
        print(f"    (MCC gộp để tham khảo: {pooled:.3f} — KHÔNG dùng để so với P1)")
    print()

    print("-" * 84)
    print("MCC theo nhóm hướng dịch (đối chiếu P1 Figure 2)")
    print("-" * 84)
    group_cols = [c for c in res.columns if c.startswith("mcc_") and "->" in c]
    print(res[group_cols].to_string())
    print()

    # Kiểm định: LaBSE có thực sự khác BLASER-QE không?
    if "LaBSE cosine" in preds and "BLASER-2.0-QE" in preds:
        r = mcnemar_test(test["label"], preds["LaBSE cosine"], preds["BLASER-2.0-QE"])
        print("-" * 84)
        print("KIỂM ĐỊNH McNEMAR: LaBSE cosine  vs  BLASER-2.0-QE")
        print("-" * 84)
        print(f"  chỉ LaBSE đúng : {r.n_only_a_correct}")
        print(f"  chỉ BLASER đúng: {r.n_only_b_correct}")
        print(f"  p = {r.p_value:.2e}  ({r.method})")
        verdict = "khác biệt có ý nghĩa" if r.p_value < 0.05 else "chưa đủ bằng chứng khác biệt"
        print(f"  -> {verdict} (alpha = 0.05)")
        print()

    print(f"Đã ghi: {RESULTS / 'halomi_baselines.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
