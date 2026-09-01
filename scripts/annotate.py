"""Công cụ gán nhãn ảo giác cho ViHalluMT.

Chạy::

    python scripts/annotate.py                       # gán nhãn Tập A
    python scripts/annotate.py --review              # xem lại nhãn đã gán
    python scripts/annotate.py --retest --n 100      # gán lại để đo kappa

Nguyên tắc thiết kế
-------------------
* **Tự lưu sau mỗi nhãn.** Mất điện giữa buổi thì chỉ mất đúng một mục.
* **Ẩn điểm của máy.** Người gán nhãn không được nhìn `score_labse`,
  `selection`... vì nhìn vào sẽ khiến nhãn người bị kéo theo dự đoán của máy,
  và bộ dữ liệu mất giá trị làm chuẩn đối chiếu (xem guideline §7.3).
* **Xáo thứ tự.** Nếu gán theo đúng thứ tự tệp thì các mục cùng tầng lấy mẫu
  sẽ nằm liền nhau, tạo hiệu ứng mỏ neo: gặp mười câu ảo giác liên tiếp rồi sẽ
  có xu hướng gán câu thứ mười một là ảo giác.
* **Cho phép bỏ qua.** Ca thật sự không quyết định được thì bỏ qua còn hơn
  đoán bừa; các mục bỏ qua được thống kê riêng.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

DATA = ROOT / "data" / "vihallumt"

SEVERITY_LABELS = {
    "0": (0, "Không có ảo giác"),
    "1": (1, "Nhỏ — 1–2 từ ảo giác"),
    "2": (2, "Một phần — ≥3 từ, chưa toàn bộ"),
    "3": (3, "Toàn phần — gần như cả câu"),
}

TYPE_LABELS = {
    "o": "oscillatory",
    "d": "detached",
    "t": "off_target",
    "f": "fabricated",
    "m": "mixed",
}

TYPE_HELP = {
    "oscillatory": "lặp cụm từ",
    "detached": "trôi chảy nhưng lạc đề hoàn toàn",
    "off_target": "ra sai ngôn ngữ",
    "fabricated": "bịa tên/số/ngày",
    "mixed": "nhiều loại, không loại nào nổi trội",
}

#: Cột phải giấu khỏi người gán nhãn (guideline §7.3)
HIDDEN_COLUMNS = {
    "score_labse", "score_ngram", "score_length_ratio", "agg_score",
    "selection", "translator", "decoding", "gen_tag", "reference",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(DATA / "to_annotate.jsonl"))
    ap.add_argument("--output", default=str(DATA / "annotations.jsonl"))
    ap.add_argument("--review", action="store_true",
                    help="Xem lai cac nhan da gan thay vi gan moi")
    ap.add_argument("--retest", action="store_true",
                    help="Gan lai N muc dau tien de do kappa test-retest")
    ap.add_argument("--n", type=int, default=None, help="So muc toi da trong phien nay")
    ap.add_argument("--seed", type=int, default=42, help="Hat giong xao thu tu")
    ap.add_argument("--show-reference", action="store_true",
                    help="Hien ban dich tham chieu (CHI dung cho ca kho)")
    return ap.parse_args()


def load_done(path: Path) -> dict[str, dict]:
    """Nạp các nhãn đã gán, khoá theo `cand_id`."""
    if not path.exists():
        return {}
    done = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                done[rec["cand_id"]] = rec
    return done


def append_record(path: Path, record: dict) -> None:
    """Ghi ngay một nhãn xuống đĩa. Tự lưu sau mỗi mục."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def wrap(text: str, width: int = 76, indent: str = "    ") -> str:
    import textwrap

    return "\n".join(textwrap.wrap(str(text), width=width,
                                   initial_indent=indent, subsequent_indent=indent))


def show_item(row: pd.Series, done: int, total: int, show_reference: bool) -> None:
    clear_screen()
    pct = 100.0 * done / max(1, total)
    bar_len = 40
    filled = int(bar_len * done / max(1, total))
    bar = "#" * filled + "." * (bar_len - filled)

    print("=" * 80)
    print(f" ViHalluMT — gán nhãn ảo giác        {done}/{total}  ({pct:.1f}%)")
    print(f" [{bar}]")
    print("=" * 80)
    print()
    print(f" Hướng dịch: {row.get('direction', '?')}")
    print()
    print(" CÂU NGUỒN:")
    print(wrap(row["src_text_original"]))
    print()
    print(" BẢN DỊCH MÁY:")
    print(wrap(row["mt_text"]))
    if show_reference and "reference" in row and pd.notna(row["reference"]):
        print()
        print(" (tham chiếu — chỉ dùng khi thật sự bí):")
        print(wrap(row["reference"]))
    print()
    print("-" * 80)
    print(" Mức nghiêm trọng:")
    for key, (_, desc) in SEVERITY_LABELS.items():
        print(f"   [{key}] {desc}")
    print("   [s] bỏ qua    [q] lưu và thoát")
    print("-" * 80)


def ask_severity() -> str | None:
    """Trả về khoá mức nghiêm trọng, 's' để bỏ qua, hoặc None để thoát."""
    while True:
        choice = input(" Chọn > ").strip().lower()
        if choice in SEVERITY_LABELS or choice == "s":
            return choice
        if choice == "q":
            return None
        print("   Chỉ nhận 0 / 1 / 2 / 3 / s / q")


def ask_type() -> str:
    """Hỏi loại ảo giác. Enter trống mặc định là `mixed`."""
    print()
    print(" Loại ảo giác:")
    for key, name in TYPE_LABELS.items():
        print(f"   [{key}] {name:<12} — {TYPE_HELP[name]}")
    while True:
        choice = input(" Chọn > ").strip().lower()
        if choice in TYPE_LABELS:
            return TYPE_LABELS[choice]
        if choice == "":
            return "mixed"
        print("   Chỉ nhận o / d / t / f / m")


def ask_note() -> str:
    return input(" Ghi chú (Enter để bỏ qua) > ").strip()


def run_annotation(args: argparse.Namespace) -> int:
    in_path, out_path = Path(args.input), Path(args.output)
    if args.retest:
        out_path = out_path.with_name(out_path.stem + "_retest.jsonl")

    if not in_path.exists():
        print(f"Khong tim thay {in_path}")
        print("Hay chay `python scripts/build_corpus_a.py` truoc.")
        return 1

    df = pd.read_json(in_path, lines=True)

    # Xáo thứ tự để tránh hiệu ứng mỏ neo giữa các tầng lấy mẫu
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    done = load_done(out_path)
    if args.retest:
        # Chế độ đo lại: chỉ lấy N mục ĐẦU TIÊN của lượt gán gốc
        original = load_done(Path(args.output))
        first_ids = list(original)[: (args.n or 100)]
        df = df[df["cand_id"].isin(first_ids)].reset_index(drop=True)
        print(f"Che do test-retest: {len(df)} muc tu luot gan goc")
        input("Nhan Enter de bat dau (dung nho lai nhan cu!) ... ")

    pending = df[~df["cand_id"].isin(done)].reset_index(drop=True)
    if args.n:
        pending = pending.head(args.n)

    total = len(df)
    if pending.empty:
        print(f"Da gan xong toan bo {total} muc. Ket qua: {out_path}")
        return 0

    print(f"Con {len(pending)} muc chua gan (tong {total}).")
    print("Meo: gan khoang 150 muc moi buoi, nghi giua chung de giu chuan on dinh.")
    input("Nhan Enter de bat dau ... ")

    n_skipped = 0
    for _, row in pending.iterrows():
        show_item(row, len(done), total, args.show_reference)

        choice = ask_severity()
        if choice is None:
            break
        if choice == "s":
            n_skipped += 1
            continue

        severity, _ = SEVERITY_LABELS[choice]
        hall_type = ask_type() if severity > 0 else ""
        note = ask_note()

        record = {
            "cand_id": row["cand_id"],
            "direction": row.get("direction", ""),
            "src_text": row["src_text_original"],
            "mt_text": row["mt_text"],
            "severity": severity,
            "label": int(severity > 0),
            "hallucination_type": hall_type,
            "annotator_note": note,
        }
        append_record(out_path, record)
        done[row["cand_id"]] = record

    print()
    print("=" * 80)
    print(f" Da gan: {len(done)}/{total}   Bo qua trong phien nay: {n_skipped}")
    print(f" Da luu: {out_path}")
    if len(done) < total:
        print(f" Chay lai lenh nay de gan tiep {total - len(done)} muc con lai.")
    print("=" * 80)
    return 0


def run_review(args: argparse.Namespace) -> int:
    """Tóm tắt các nhãn đã gán."""
    out_path = Path(args.output)
    done = load_done(out_path)
    if not done:
        print(f"Chua co nhan nao trong {out_path}")
        return 1

    df = pd.DataFrame(done.values())
    print("=" * 70)
    print(f"DA GAN {len(df)} MUC")
    print("=" * 70)
    print()
    print("Phan bo muc nghiem trong:")
    for sev in range(4):
        n = int((df["severity"] == sev).sum())
        print(f"  {sev} {SEVERITY_LABELS[str(sev)][1]:<32} {n:>5}  "
              f"({100 * n / len(df):.1f}%)")
    print()
    print(f"Ti le ao giac (nhi phan): {100 * df['label'].mean():.1f}%")
    print()
    positives = df[df["severity"] > 0]
    if len(positives):
        print("Phan bo loai ao giac:")
        for t, n in positives["hallucination_type"].value_counts().items():
            print(f"  {t:<14} {n:>5}  ({100 * n / len(positives):.1f}%)")
    print()
    if "direction" in df.columns and df["direction"].nunique() > 1:
        print("Theo huong dich:")
        for d, g in df.groupby("direction"):
            print(f"  {d:<8} n={len(g):<5} ao giac {100 * g['label'].mean():.1f}%")
    print()
    notes = df[df["annotator_note"].astype(str).str.strip() != ""]
    print(f"So muc co ghi chu: {len(notes)}")
    return 0


def main() -> int:
    args = parse_args()
    return run_review(args) if args.review else run_annotation(args)


if __name__ == "__main__":
    raise SystemExit(main())
