"""Kiểm định ngữ liệu ViHalluMT trước khi bỏ công gán nhãn.

Chạy sau khi notebook Kaggle sinh xong dữ liệu, TRƯỚC khi bắt đầu gán nhãn.
Mục đích là phát hiện sớm mọi vấn đề khiến 10 giờ công gán nhãn thành lãng phí.

Bốn câu hỏi phải trả lời được
-----------------------------
1. Cấu trúc có đúng kế hoạch không (hướng dịch, hệ dịch, nguồn văn bản)?
2. Có bao nhiêu mẫu dương dự kiến? Quá ít thì mọi độ đo đều vô nghĩa.
3. Có lỗi dữ liệu nào không (bản dịch rỗng, chép nguyên câu nguồn)?
4. Các loại ảo giác có đủ đa dạng để phân tích lỗi không?

Cách ước lượng số mẫu dương
---------------------------
Chưa có nhãn người, nên ta **hiệu chuẩn trên HalOmi**: đo P(ảo giác | cosine
LaBSE) trên dữ liệu có nhãn thật của HalOmi, rồi áp phân phối cosine của
ViHalluMT vào. Cách này đáng tin hơn nhiều so với đoán, dù vẫn là ước lượng —
LaBSE có thể hiệu chuẩn khác nhau giữa tiếng Việt và các ngôn ngữ của HalOmi.

Chạy:  python scripts/audit_corpus.py
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from vihallumt.data import load_halomi

DATA = ROOT / "data" / "vihallumt"
RESULTS = ROOT / "results"

#: Ngưỡng cosine dùng để phân khoảng khi hiệu chuẩn.
COSINE_BINS = [-1.0, 0.30, 0.45, 0.60, 0.70, 0.80, 0.90, 1.01]
COSINE_LABELS = ["<0.30", "0.30-0.45", "0.45-0.60", "0.60-0.70",
                 "0.70-0.80", "0.80-0.90", ">0.90"]

#: Chữ viết phi Latinh — dấu hiệu bản dịch ra sai ngôn ngữ.
NON_LATIN = re.compile(r"[一-鿿぀-ヿ฀-๿Ѐ-ӿ؀-ۿ]")


def has_vietnamese_diacritics(text: str) -> bool:
    """Văn bản có dấu tiếng Việt không."""
    decomposed = unicodedata.normalize("NFD", str(text))
    if any(unicodedata.category(ch) == "Mn" for ch in decomposed):
        return True
    return "đ" in str(text).lower()


def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Gắn cờ các dấu hiệu ảo giác đọc được bằng máy."""
    df = df.copy()
    df["cos"] = -df["score_labse"]

    df["empty"] = df["mt_text"].astype(str).str.strip().str.len() < 3
    df["oscillatory"] = df["score_ngram"] > 2          # luật của P2 §6.4
    df["non_latin"] = df["mt_text"].astype(str).apply(lambda t: bool(NON_LATIN.search(t)))

    to_vi = df["tgt"] == "VI"
    has_diacritics = df["mt_text"].apply(has_vietnamese_diacritics)
    # Dịch sang VI mà không có dấu, hoặc dịch sang EN mà lại có dấu VI
    df["wrong_language"] = ((to_vi & ~has_diacritics) | (~to_vi & has_diacritics)
                            | df["non_latin"])
    df["copied_source"] = (df["mt_text"].astype(str).str.strip().str.lower()
                           == df["src_text_original"].astype(str).str.strip().str.lower())
    df["low_similarity"] = df["cos"] < 0.45
    df["length_blowup"] = df["score_length_ratio"] > 1.5

    flags = ["empty", "oscillatory", "wrong_language", "copied_source",
             "low_similarity", "length_blowup"]
    df["any_flag"] = df[flags].any(axis=1)
    return df


def calibrate_positive_rate(sample: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Ước lượng tỉ lệ mẫu dương bằng cách hiệu chuẩn trên HalOmi."""
    halomi = load_halomi()
    halomi["cos"] = -halomi["score_labse"]
    halomi["bin"] = pd.cut(halomi["cos"], COSINE_BINS, labels=COSINE_LABELS)

    sample = sample.copy()
    sample["bin"] = pd.cut(sample["cos"], COSINE_BINS, labels=COSINE_LABELS)

    rate = halomi.groupby("bin", observed=True)["label"].agg(["mean", "size"])
    counts = sample.groupby("bin", observed=True).size()

    rows, expected_total = [], 0.0
    for label in COSINE_LABELS:
        if label not in rate.index:
            continue
        p = float(rate.loc[label, "mean"])
        n_vi = int(counts.get(label, 0))
        expected = p * n_vi
        expected_total += expected
        rows.append({
            "cosine_bin": label,
            "p_hallucination_halomi": round(p, 4),
            "n_halomi": int(rate.loc[label, "size"]),
            "n_vihallumt": n_vi,
            "expected_positive": round(expected, 1),
        })
    return pd.DataFrame(rows), expected_total


def report_labse_blind_spot(candidates: pd.DataFrame) -> pd.DataFrame | None:
    """Đo mức độ LaBSE bỏ sót ảo giác sai ngôn ngữ đích.

    Nhánh ép sai ngôn ngữ đích cho ta **nhãn chắc chắn đúng**: mọi câu ở đó
    đều là ảo giác toàn phần loại `off_target`. Vì vậy nó là phép thử sạch cho
    câu hỏi: bộ phát hiện bằng embedding có bắt được loại lỗi này không?
    """
    is_off = candidates["gen_tag"].astype(str).str.contains("off", na=False)
    off, normal = candidates[is_off], candidates[~is_off]
    if off.empty:
        return None

    return pd.DataFrame([
        {"group": "ep sai ngon ngu dich", "n": len(off),
         "median_cosine": round(off["cos"].median(), 3),
         "pct_below_0.45": round(100 * (off["cos"] < 0.45).mean(), 1)},
        {"group": "dich binh thuong", "n": len(normal),
         "median_cosine": round(normal["cos"].median(), 3),
         "pct_below_0.45": round(100 * (normal["cos"] < 0.45).mean(), 1)},
    ])


def main() -> int:
    RESULTS.mkdir(exist_ok=True)

    cand_path, ann_path = DATA / "candidates.jsonl", DATA / "to_annotate.jsonl"
    for p in (cand_path, ann_path):
        if not p.exists():
            print(f"Khong tim thay {p}")
            print("Chay notebook Kaggle truoc roi tai ket qua ve day.")
            return 1

    candidates = add_quality_flags(pd.read_json(cand_path, lines=True))
    sample = add_quality_flags(pd.read_json(ann_path, lines=True))
    natural = candidates[candidates["perturbation"] == "natural"]

    print("=" * 78)
    print("KIEM DINH NGU LIEU ViHalluMT")
    print("=" * 78)
    print(f"Ung vien     : {len(candidates)}  (natural {len(natural)})")
    print(f"Cho gan nhan : {len(sample)}")
    print()

    print("-" * 78)
    print("1. CAU TRUC")
    print("-" * 78)
    print(candidates.groupby(["direction", "perturbation"]).size().to_string())
    print()
    print(candidates["gen_tag"].value_counts().to_string())
    print()
    print(candidates["source"].value_counts().to_string())
    print()

    print("-" * 78)
    print("2. UOC LUONG SO MAU DUONG (hieu chuan tren HalOmi co nhan that)")
    print("-" * 78)
    calib, expected = calibrate_positive_rate(sample)
    print(calib.to_string(index=False))
    print()
    rate = 100 * expected / len(sample)
    print(f"  Du kien: {expected:.0f}/{len(sample)} = {rate:.1f}% mau duong")
    print(f"  (HalOmi that: {100 * load_halomi()['label'].mean():.1f}%)")
    print()
    for s in ("uniform", "biased", "worst"):
        sub = sample[sample["selection"] == s]
        if sub.empty:
            continue
        _, e = calibrate_positive_rate(sub)
        print(f"    {s:<9} {e:5.0f}/{len(sub):3d} = {100 * e / len(sub):5.1f}%")
    print()

    print("-" * 78)
    print("3. DAU HIEU CHAT LUONG (theo tang lay mau)")
    print("-" * 78)
    flags = ["empty", "copied_source", "oscillatory", "wrong_language",
             "low_similarity", "length_blowup", "any_flag"]
    table = sample.groupby("selection")[flags].mean().mul(100).round(1)
    print(table.to_string())
    print()

    print("-" * 78)
    print("4. DIEM MU CUA LaBSE VOI AO GIAC SAI NGON NGU DICH")
    print("-" * 78)
    blind = report_labse_blind_spot(natural)
    if blind is None:
        print("  (khong co nhanh ep sai ngon ngu dich trong du lieu)")
    else:
        print(blind.to_string(index=False))
        print()
        print("  Nhanh ep sai ngon ngu dich la ao giac toan phan theo cau tao,")
        print("  nhung LaBSE cham diem chung NGANG voi ban dich dung. Ly do co")
        print("  tinh cau truc: LaBSE la embedding BAT BIEN NGON NGU, nen ban")
        print("  dich sang tieng Trung van tuong duong ngu nghia voi cau nguon")
        print("  tieng Anh. Moi bo phat hien dua tren tuong dong embedding deu")
        print("  chung diem mu nay — ke ca SONAR va BLASER.")
    print()

    print("-" * 78)
    print("5. KET LUAN")
    print("-" * 78)
    problems = []
    if rate < 15:
        problems.append(f"ti le mau duong du kien qua thap ({rate:.1f}%)")
    if sample["empty"].any():
        problems.append(f"{int(sample['empty'].sum())} ban dich rong")
    if sample["copied_source"].mean() > 0.05:
        problems.append("nhieu ban dich chep nguyen cau nguon")
    if sample["selection"].nunique() < 3:
        problems.append("thieu tang lay mau")
    if int(sample["oscillatory"].sum()) < 5:
        problems.append(f"chi co {int(sample['oscillatory'].sum())} ca dao dong — "
                        f"khong du de phan tich rieng loai nay")

    if problems:
        print("  Van de can luu y:")
        for p in problems:
            print(f"    - {p}")
    else:
        print("  Khong phat hien van de.")
    print()
    blocking = [p for p in problems if "qua thap" in p or "rong" in p or "thieu tang" in p]
    print("  => " + ("DUNG LAI, xu ly truoc khi gan nhan." if blocking
                     else "SAN SANG GAN NHAN."))

    calib.to_csv(RESULTS / "corpus_calibration.csv", index=False)
    table.to_csv(RESULTS / "corpus_quality_flags.csv")
    if blind is not None:
        blind.to_csv(RESULTS / "labse_offtarget_blindspot.csv", index=False)
    print()
    print(f"Da ghi: {RESULTS / 'corpus_calibration.csv'}")
    print(f"Da ghi: {RESULTS / 'corpus_quality_flags.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
