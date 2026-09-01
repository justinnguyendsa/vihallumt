"""Sinh Tập A của ViHalluMT — bản dịch thật kèm ứng viên để gán nhãn tay.

Cấu trúc bám theo HalOmi
------------------------
Bộ dữ liệu chia làm hai phần, đúng như HalOmi:

``natural``    câu nguồn **nguyên vẹn**. Bản dịch là đầu ra thật của hệ dịch
               dưới nhiều cấu hình giải mã khác nhau. **Đây là phần dùng cho
               bảng kết quả chính.**
``perturbed``  câu nguồn bị làm nhiễu trước khi dịch. Giữ lại để phân tích,
               nhưng **loại khỏi kết quả chính** — P1 §2.1 loại bỏ đúng phần
               tương ứng của HalOmi với lý do "kết luận rút từ dữ liệu nhiễu
               loạn có thể không áp dụng được cho ảo giác tự nhiên".

Làm giàu mẫu dương bằng cách nào
--------------------------------
Không phải bằng nhiễu loạn, mà bằng ba cơ chế giữ nguyên tính tự nhiên:

1. **Lấy mẫu phân tầng** uniform/biased/worst — đo trên HalOmi cho thấy tầng
   ``worst`` chứa 85,7% ảo giác so với 25,6% của phân bố gốc (gấp 3,3 lần).
2. **Miền văn bản nhiễu** — OPUS-100 (phụ đề) sinh nhiều ảo giác hơn IWSLT.
3. **Đa dạng cấu hình giải mã** — sampling nhiệt độ cao và epsilon đẩy mô hình
   vào vùng đuôi phân phối. Đầu ra vẫn là đầu ra thật của mô hình.

Chạy (cần GPU cho bước dịch)::

    python scripts/build_corpus_a.py --n-source 3000 --n-annotate 700

Đầu ra:
    data/vihallumt/candidates.jsonl   toàn bộ ứng viên kèm điểm
    data/vihallumt/to_annotate.jsonl  mẫu đã chọn, chờ gán nhãn tay
    results/corpus_a_stats.csv        thống kê để đưa vào báo cáo
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from vihallumt.corpus.perturb import perturb, perturbation_plan
from vihallumt.corpus.sampling import selection_report, stratified_sample
from vihallumt.corpus.sources import build_source_pool
from vihallumt.corpus.translate import (
    DEFAULT_GENERATION_PLAN,
    SMOKE_GENERATION_PLAN,
    get_decoding,
    make_translator,
    probe_translator,
    redistribute_shares,
    validate_plan,
)
from vihallumt.detectors.base import Pair
from vihallumt.detectors.ngram import NGramRepetitionDetector

DATA = ROOT / "data" / "vihallumt"
RESULTS = ROOT / "results"
#: Ket qua dich trung gian, ghi sau moi to hop (he dich, giai ma).
#: Chay lai se tu bo qua phan da xong.
CKPT = DATA / "_checkpoints"

#: Tỉ lệ câu nguồn được đưa vào nhánh perturbed.
#: Giữ nhỏ vì phần này không dùng cho kết quả chính.
PERTURBED_SHARE = 0.20


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-source", type=int, default=3000,
                    help="So cau nguon lay tu moi huong dich")
    ap.add_argument("--n-annotate", type=int, default=700,
                    help="So cap chon ra de gan nhan tay")
    ap.add_argument("--directions", default="en-vi,vi-en",
                    help="Danh sach huong dich, ngan cach bang dau phay")
    ap.add_argument("--no-flores", action="store_true",
                    help="Bo qua FLORES (dataset gated)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None,
                    help="Chi xu ly N cau (de kiem tra nhanh duong ong)")
    ap.add_argument("--plan", default="default", choices=["default", "smoke"],
                    help="'smoke' chi dung NLLB-600M — de kiem tra duong ong")
    ap.add_argument("--fresh", action="store_true",
                    help="Xoa checkpoint va dich lai tu dau")
    return ap.parse_args()


def resolve_plan(plan_name: str):
    """Chọn kế hoạch sinh và loại bỏ hệ dịch không nạp được.

    Thử nạp tokenizer của từng hệ TRƯỚC khi dịch. Trên Kaggle, một lượt chạy
    đổ sau 20 phút GPU là mất trắng một chu kỳ làm việc; thà phát hiện ngay từ
    đầu và chạy tiếp với ít hệ dịch hơn, miễn là ghi rõ đã bỏ cái gì.
    """
    plan = SMOKE_GENERATION_PLAN if plan_name == "smoke" else DEFAULT_GENERATION_PLAN
    validate_plan(plan)

    print("[0/5] Kiem tra cac he dich ...")
    broken: set[str] = set()
    for name in sorted({s.translator for s in plan}):
        ok, msg = probe_translator(name)
        print(f"    {name:<12} {'OK' if ok else 'HONG'}  {msg if not ok else ''}")
        if not ok:
            broken.add(name)

    if broken:
        plan = redistribute_shares(plan, broken)
        print(f"    -> da bo {sorted(broken)}, chia lai ti trong cho "
              f"{sorted({s.translator for s in plan})}")
        print("    !! GHI VAO MUC LIMITATIONS CUA BAO CAO !!")
    return plan


def assign_generation_specs(n: int, seed: int, plan) -> list:
    """Phân bổ tổ hợp (hệ dịch, giải mã) cho n câu theo đúng tỉ trọng kế hoạch."""
    rng = random.Random(seed)

    schedule = []
    for spec in plan:
        schedule.extend([spec] * int(round(n * spec.share)))
    while len(schedule) < n:
        schedule.append(plan[0])
    schedule = schedule[:n]
    rng.shuffle(schedule)
    return schedule


def build_translators(specs) -> dict:
    """Khởi tạo mỗi hệ dịch đúng một lần, dùng lại cho mọi cấu hình giải mã.

    Riêng nhánh ép sai ngôn ngữ đích cần đối tượng riêng vì token BOS khác.
    """
    needed = {(s.translator, s.force_wrong_target) for s in specs}
    out = {}
    for name, off_target in sorted(needed, key=lambda t: (t[0], t[1] or "")):
        kwargs = {"force_wrong_target": off_target} if off_target else {}
        out[(name, off_target)] = make_translator(name, **kwargs)
    return out


def translate_pool(
    df: pd.DataFrame,
    src_lang: str,
    tgt_lang: str,
    batch_size: int,
    seed: int,
    plan,
    ckpt_dir: Path | None = None,
) -> pd.DataFrame:
    """Dịch từng câu bằng tổ hợp (hệ dịch, giải mã) đã phân bổ cho nó.

    Ghi checkpoint sau **mỗi** tổ hợp (hệ dịch, giải mã). Dịch là bước tốn thời
    gian nhất — 20–30 phút GPU cho một lượt đầy đủ. Nếu hỏng ở phút thứ 35 mà
    không có checkpoint thì mất trắng toàn bộ; với hạn mức GPU của Kaggle, đó
    là mất nguyên một chu kỳ làm việc. Chạy lại sẽ tự bỏ qua phần đã xong.
    """
    specs = assign_generation_specs(len(df), seed, plan)
    df = df.copy()
    df["gen_tag"] = [s.tag for s in specs]
    df["translator"] = [s.translator for s in specs]
    df["decoding"] = [s.decoding for s in specs]
    df["force_wrong_target"] = [s.force_wrong_target for s in specs]

    translators = build_translators(specs)
    df["mt_text"] = ""

    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Gom theo (hệ dịch, giải mã) để mỗi tổ hợp chỉ nạp mô hình một lần
    for (translator, decoding, off_target), group in df.groupby(
        ["translator", "decoding", "force_wrong_target"], dropna=False
    ):
        off = off_target if isinstance(off_target, str) else None
        cfg = get_decoding(decoding)
        tag = f"{translator}/{decoding}" + (f"+off{off}" if off else "")

        ckpt = None
        if ckpt_dir is not None:
            key = f"{src_lang}-{tgt_lang}__{translator}__{decoding}__{off or 'none'}"
            ckpt = ckpt_dir / f"{key}.jsonl"

        # Đã dịch xong tổ hợp này ở lần chạy trước -> dùng lại
        if ckpt is not None and ckpt.exists():
            cached = pd.read_json(ckpt, lines=True)
            if len(cached) == len(group):
                df.loc[group.index, "mt_text"] = cached["mt_text"].to_numpy()
                print(f"    [da co] bo qua {tag} — dung lai {len(cached)} cau tu checkpoint")
                continue
            print(f"    [checkpoint hong] {tag}: co {len(cached)} cau nhung can "
                  f"{len(group)} — dich lai")

        print(f"    dich {len(group):>5} cau bang {tag}", flush=True)
        model = translators[(translator, off)]
        outputs = model.translate(
            group["src_text"].tolist(), src_lang, tgt_lang,
            decoding=cfg, batch_size=batch_size,
        )
        df.loc[group.index, "mt_text"] = outputs

        if ckpt is not None:
            pd.DataFrame({"cand_key": group.index, "mt_text": outputs}).to_json(
                ckpt, orient="records", lines=True, force_ascii=False)

    return df


def score_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Chấm điểm sơ bộ bằng các detector rẻ, để phục vụ lấy mẫu phân tầng.

    Quan trọng: chấm điểm với câu nguồn **gốc** (`src_text_original`), không
    phải câu đã nhiễu loạn. Người gán nhãn cũng đánh giá bản dịch so với ý
    nghĩa gốc, nên điểm số phải nhất quán với việc đó.
    """
    from vihallumt.detectors.embed import EmbeddingDetector

    df = df.copy()
    pairs = [
        Pair(s, m, sl, tl)
        for s, m, sl, tl in zip(
            df["src_text_original"], df["mt_text"], df["src"], df["tgt"]
        )
    ]

    print("    LaBSE ...")
    df["score_labse"] = EmbeddingDetector("sentence-transformers/LaBSE").score(pairs)

    print("    dem lap n-gram ...")
    df["score_ngram"] = NGramRepetitionDetector().score(pairs)

    # Tỉ lệ độ dài: bản dịch dài bất thường là dấu hiệu ảo giác dao động
    len_src = df["src_text_original"].str.split().str.len().clip(lower=1)
    len_mt = df["mt_text"].str.split().str.len()
    df["score_length_ratio"] = (len_mt / len_src - 1.0).abs()
    return df


def build_direction(args: argparse.Namespace, src_lang: str, tgt_lang: str,
                    gen_plan) -> pd.DataFrame:
    """Dựng toàn bộ ứng viên cho một hướng dịch."""
    print(f"\n{'=' * 74}\nHUONG DICH {src_lang} -> {tgt_lang}\n{'=' * 74}")

    print("[1/5] Nap ngu lieu nguon ...")
    pool, stats = build_source_pool(
        n_per_source=args.n_source // 2,
        use_flores=not args.no_flores,
        seed=args.seed,
    )
    for st in stats:
        print(f"    {st.name:<40} {st.n_after_filter:>6} / {st.n_raw:<8} "
              f"({st.kept_pct:.1f}% giu lai)")

    if args.limit:
        pool = pool.head(args.limit)
    print(f"    tong cong: {len(pool)} cau nguon")

    # Câu nguồn theo chiều dịch
    src_col, ref_col = ("en", "vi") if src_lang == "EN" else ("vi", "en")
    pool = pool.rename(columns={src_col: "src_text_original", ref_col: "reference"})
    pool["src"], pool["tgt"] = src_lang, tgt_lang
    pool["direction"] = f"{src_lang}-{tgt_lang}"

    print("[2/5] Chia nhanh natural / perturbed ...")
    perturb_plan = perturbation_plan(len(pool), src_lang,
                                     clean_ratio=1.0 - PERTURBED_SHARE, seed=args.seed)
    rng = random.Random(args.seed)
    pool["perturbation"] = [
        "natural" if k == "clean" else "perturbed" for k in perturb_plan
    ]
    pool["perturbation_kind"] = perturb_plan
    pool["src_text"] = [
        perturb(t, src_lang, k, rng).text
        for t, k in zip(pool["src_text_original"], perturb_plan)
    ]
    print(f"    natural: {(pool.perturbation == 'natural').sum()}  |  "
          f"perturbed: {(pool.perturbation == 'perturbed').sum()}")

    print("[3/5] Dich (can GPU) ...")
    pool = translate_pool(pool, src_lang, tgt_lang, args.batch_size, args.seed,
                          gen_plan, ckpt_dir=CKPT)

    print("[4/5] Cham diem so bo ...")
    pool = score_candidates(pool)

    # Bỏ những bản dịch rỗng hoặc hỏng hoàn toàn
    before = len(pool)
    pool = pool[pool["mt_text"].str.strip().str.len() > 0].reset_index(drop=True)
    if len(pool) < before:
        print(f"    bo {before - len(pool)} ban dich rong")

    print(f"[5/5] Xong huong {src_lang}->{tgt_lang}: {len(pool)} ung vien")
    return pool


def main() -> int:
    args = parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    if args.fresh and CKPT.exists():
        import shutil
        shutil.rmtree(CKPT)
        print("Da xoa checkpoint, se dich lai tu dau.\n")

    gen_plan = resolve_plan(args.plan)

    frames = []
    for direction in args.directions.split(","):
        src_lang, tgt_lang = (p.strip().upper() for p in direction.split("-"))

        # Checkpoint muc huong dich: da lam xong ca huong nay thi dung lai luon
        done_path = CKPT / f"direction_{src_lang}-{tgt_lang}.jsonl"
        if done_path.exists():
            print(f"\n[da co] Huong {src_lang}->{tgt_lang} da hoan tat o lan chay "
                  f"truoc — dung lai tu checkpoint.")
            frames.append(pd.read_json(done_path, lines=True))
            continue

        frame = build_direction(args, src_lang, tgt_lang, gen_plan)
        CKPT.mkdir(parents=True, exist_ok=True)
        frame.to_json(done_path, orient="records", lines=True, force_ascii=False)
        frames.append(frame)

    candidates = pd.concat(frames, ignore_index=True)
    candidates.insert(0, "cand_id", [f"c{i:06d}" for i in range(len(candidates))])

    cand_path = DATA / "candidates.jsonl"
    candidates.to_json(cand_path, orient="records", lines=True, force_ascii=False)

    print(f"\n{'=' * 74}\nLAY MAU DE GAN NHAN\n{'=' * 74}")

    # Chỉ lấy mẫu từ nhánh natural — đây là phần cho kết quả chính
    natural = candidates[candidates["perturbation"] == "natural"].reset_index(drop=True)
    print(f"Kho natural: {len(natural)} ung vien")

    n_annotate = min(args.n_annotate, len(natural))
    sample = stratified_sample(
        natural,
        ["score_labse", "score_ngram", "score_length_ratio"],
        n_total=n_annotate,
        seed=args.seed,
    )

    keep = [
        "cand_id", "direction", "src", "tgt", "src_text_original", "reference",
        "mt_text", "source", "translator", "decoding", "gen_tag",
        "perturbation", "selection", "agg_score",
        "score_labse", "score_ngram", "score_length_ratio",
    ]
    sample_out = sample[[c for c in keep if c in sample.columns]].copy()
    # Ô trống để người gán nhãn điền
    sample_out["severity"] = ""
    sample_out["hallucination_type"] = ""
    sample_out["annotator_note"] = ""

    ann_path = DATA / "to_annotate.jsonl"
    sample_out.to_json(ann_path, orient="records", lines=True, force_ascii=False)

    report = selection_report(sample)
    report.to_csv(RESULTS / "corpus_a_stats.csv")

    print(report.to_string())
    print()
    print(f"Da ghi: {cand_path}   ({len(candidates)} ung vien)")
    print(f"Da ghi: {ann_path}    ({len(sample_out)} cap cho gan nhan)")
    print(f"Da ghi: {RESULTS / 'corpus_a_stats.csv'}")
    print()
    print("Buoc tiep theo: chay cong cu gan nhan tren to_annotate.jsonl")
    print()
    print(f"(Checkpoint nam o {CKPT} — chay lai se dung lai chung. "
          f"Muon dich lai tu dau thi them --fresh.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
