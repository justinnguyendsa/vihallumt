"""Chạy bộ phát hiện ảo giác bằng LLM trên HalOmi (hoặc ViHalluMT).

Hai chế độ dùng
---------------

1. **Quét chọn prompt** trên tập validation — tái hiện Bảng 6 của P1::

     python scripts/run_llm_detector.py --model Qwen/Qwen2.5-7B-Instruct --grid

2. **Chạy prompt tốt nhất** trên tập test::

     python scripts/run_llm_detector.py --model Qwen/Qwen2.5-7B-Instruct \
         --split test --prompt-id p3 --cot none

Ghi ra hai file:
  results/llm_<model>_<split>_scores.csv   điểm thô từng câu (để phân tích lỗi)
  results/llm_<model>_<split>_summary.csv  bảng độ đo từng biến thể prompt

Trên máy không có GPU, hãy dùng mô hình nhỏ (`Qwen/Qwen2.5-0.5B-Instruct`) và
`--limit` để kiểm tra đường ống chạy thông trước khi đưa lên Kaggle.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from vihallumt.data import load_halomi, split_halomi
from vihallumt.detectors.base import pairs_from_frame
from vihallumt.detectors.llm import LLMConfig, LLMDetector
from vihallumt.eval import (
    apply_threshold,
    binary_metrics,
    macro_average,
    tune_threshold,
)
from vihallumt.prompts.binary import PROMPT_GRID, variant_name

RESULTS = ROOT / "results"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Dinh danh mo hinh tren HuggingFace")
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--grid", action="store_true",
                    help="Quet toan bo luoi bien the prompt (P1 Bang 6)")
    ap.add_argument("--prompt-id", default="p2", choices=["p1", "p2", "p3"])
    ap.add_argument("--cot", default="none", choices=["none", "cot1", "cot2"])
    ap.add_argument("--language", default="en", choices=["en", "vi"])
    ap.add_argument("--mode", default="logit", choices=["logit", "generate"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="Chi chay N cau dau (de kiem tra nhanh)")
    ap.add_argument("--no-4bit", action="store_true",
                    help="Tat luong tu hoa 4-bit (dung cho mo hinh nho / chay CPU)")
    ap.add_argument("--tag", default=None, help="Hau to cho ten file ket qua")
    return ap.parse_args()


def variants_to_run(args: argparse.Namespace) -> list[dict]:
    """Danh sách biến thể prompt cần chạy."""
    if not args.grid:
        return [{"prompt_id": args.prompt_id, "cot": args.cot, "language": args.language}]

    out = []
    for v in PROMPT_GRID:
        # CoT bắt buộc phải sinh chuỗi; nếu người dùng chọn logit thì bỏ qua
        # các biến thể CoT thay vì làm hỏng cả lượt chạy.
        if args.mode == "logit" and v["cot"] != "none":
            continue
        out.append(dict(v))
    return out


def evaluate(
    df: pd.DataFrame,
    scores: np.ndarray,
    threshold: float | None,
) -> tuple[dict, float]:
    """Tính độ đo. Nếu `threshold` là None thì tự hiệu chỉnh trên chính `df`."""
    if threshold is None:
        threshold, _ = tune_threshold(df["label"], scores)
    pred = apply_threshold(scores, threshold)

    m = binary_metrics(df["label"], pred, score=scores)
    row = {
        "threshold": round(float(threshold), 4),
        "mcc_macro": round(macro_average(df["label"], pred, df["direction"]), 4),
        "mcc_pooled": round(m.mcc, 4),
        "f1": round(m.f1, 4),
        "precision": round(m.precision, 4),
        "recall": round(m.recall, 4),
        "accuracy": round(m.accuracy, 4),
        "roc_auc": round(m.roc_auc, 4) if m.roc_auc is not None else None,
        "pred_positive_rate": round(float(pred.mean()), 4),
        # Với mode="generate", điểm 0.5 nghĩa là không bóc được nhãn.
        "unparsed_rate": round(float(np.mean(np.isclose(scores, 0.5))), 4),
    }
    return row, float(threshold)


def main() -> int:
    args = parse_args()
    RESULTS.mkdir(exist_ok=True)

    df = load_halomi()
    val, test = split_halomi(df)
    data = val if args.split == "validation" else test
    if args.limit:
        # Lấy mẫu phân tầng theo nhãn để tập nhỏ vẫn có cả hai lớp.
        # Không dùng groupby().apply() vì pandas 3.0 loại cột nhóm khỏi kết quả.
        per_class = max(1, args.limit // 2)
        data = pd.concat(
            [data[data["label"] == c].head(per_class) for c in (0, 1)]
        ).reset_index(drop=True)
    pairs = pairs_from_frame(data)

    print("=" * 78)
    print(f"LLM DETECTOR — {args.model}")
    print("=" * 78)
    print(f"Tập      : {args.split} ({len(data)} câu, {100*data['label'].mean():.1f}% ảo giác)")
    print(f"Chế độ   : {args.mode}")
    print()

    summary_rows, score_cols = [], {}

    for variant in variants_to_run(args):
        name = variant_name(variant["prompt_id"], variant["cot"], variant["language"])
        cfg = LLMConfig(
            **variant,
            mode=args.mode,
            batch_size=args.batch_size,
            load_in_4bit=not args.no_4bit,
        )
        det = LLMDetector(args.model, cfg)

        t0 = time.perf_counter()
        scores = det.score(pairs)
        elapsed = time.perf_counter() - t0

        row, _ = evaluate(data, scores, threshold=None)
        row["variant"] = name
        row["prompt_id"] = variant["prompt_id"]
        row["cot"] = variant["cot"]
        row["language"] = variant["language"]
        row["seconds"] = round(elapsed, 1)
        row["sec_per_sentence"] = round(elapsed / max(1, len(pairs)), 3)
        summary_rows.append(row)
        score_cols[name] = scores

        print(f"  {name:<14} MCC(macro) {row['mcc_macro']:>6.3f}  "
              f"F1 {row['f1']:>5.3f}  AUC {str(row['roc_auc']):>6}  "
              f"{row['sec_per_sentence']:>5.2f}s/câu")

    summary = pd.DataFrame(summary_rows).set_index("variant").sort_values(
        "mcc_macro", ascending=False)

    slug = args.model.split("/")[-1].replace(".", "_")
    tag = f"_{args.tag}" if args.tag else ""
    summary_path = RESULTS / f"llm_{slug}_{args.split}{tag}_summary.csv"
    scores_path = RESULTS / f"llm_{slug}_{args.split}{tag}_scores.csv"

    summary.to_csv(summary_path)

    raw = data[["direction", "label", "severity", "src_text", "mt_text"]].copy()
    for name, s in score_cols.items():
        raw[f"score::{name}"] = s
    raw.to_csv(scores_path, index=False)

    print()
    print("-" * 78)
    print("TỔNG HỢP (sắp theo MCC trung bình vĩ mô)")
    print("-" * 78)
    print(summary[["mcc_macro", "mcc_pooled", "f1", "roc_auc",
                   "pred_positive_rate", "sec_per_sentence"]].to_string())
    print()
    best = summary.index[0]
    print(f"Prompt tốt nhất: {best}  (MCC macro = {summary.loc[best, 'mcc_macro']:.3f})")
    print(f"Đã ghi: {summary_path}")
    print(f"Đã ghi: {scores_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
