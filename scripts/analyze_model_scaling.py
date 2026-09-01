"""Đường cong năng lực phát hiện ảo giác theo cỡ mô hình.

Câu hỏi
-------
P1 loại GPT-3.5, Mistral Large và Llama3-8B khỏi nghiên cứu vì *"poor task
understanding"*, nhưng **không định lượng** ngưỡng đó nằm ở đâu. Script này
trả lời: khả năng phát hiện ảo giác xuất hiện từ cỡ mô hình nào?

Cách đọc
--------
`MCC` ở đây tính với ngưỡng hiệu chỉnh **trên chính tập validation**, nên là số
lạc quan (in-sample) — dùng để so sánh giữa các mô hình, không phải để công bố
như hiệu năng thật. **ROC-AUC là chỉ báo đáng tin hơn** vì không phụ thuộc
ngưỡng: AUC dưới 0,5 nghĩa là điểm số tương quan *ngược* với sự thật, tức mô
hình không hiểu tác vụ chứ không chỉ là hiệu chỉnh kém.

Chạy:  python scripts/analyze_model_scaling.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

#: Số tham số (tỉ) suy ra từ tên mô hình.
_SIZE_RE = re.compile(r"(\d+(?:[._]\d+)?)\s*[bB]\b")

#: Điểm tham chiếu từ P1 Bảng 6 (MCC trên cùng tập validation DE<->EN).
#: Đây là lý do bảng của ta so sánh được với họ: cùng tập, cùng độ đo.
PAPER_VALIDATION_MCC: dict[str, tuple[float, float]] = {
    # tên mô hình: (số tỉ tham số, MCC tốt nhất trên validation)
    "GPT4-Turbo": (float("nan"), 0.55),
    "GPT4o": (float("nan"), 0.51),
    "Command R": (35.0, 0.54),
    "Mistral 8x22b": (141.0, 0.69),
    "Claude Sonnet": (float("nan"), 0.69),
    "Claude Opus": (float("nan"), 0.73),
    "Llama3-70B": (70.0, 0.81),
}


def parse_size(model_slug: str) -> float | None:
    """Rút số tỉ tham số từ tên tệp kết quả."""
    m = _SIZE_RE.search(model_slug.replace("_", "."))
    return float(m.group(1).replace("_", ".")) if m else None


def load_runs() -> pd.DataFrame:
    """Gom mọi tệp `llm_*_validation*_summary.csv` thành một bảng."""
    rows = []
    for path in sorted(RESULTS.glob("llm_*_validation*_summary.csv")):
        slug = path.name[len("llm_"):].split("_validation")[0]
        size = parse_size(slug)
        if size is None:
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            rows.append({
                "model": slug.replace("_", "."),
                "size_b": size,
                "variant": r["variant"],
                "language": r.get("language", "en"),
                "mcc_macro": r["mcc_macro"],
                "roc_auc": r["roc_auc"],
                "f1": r["f1"],
                "pred_positive_rate": r["pred_positive_rate"],
                "sec_per_sentence": r.get("sec_per_sentence"),
            })
    if not rows:
        raise SystemExit(
            "Khong tim thay ket qua nao. Chay truoc:\n"
            "  python scripts/run_llm_detector.py --model Qwen/Qwen2.5-0.5B-Instruct "
            "--split validation --grid --no-4bit"
        )
    return pd.DataFrame(rows)


def build_curve(runs: pd.DataFrame) -> pd.DataFrame:
    """Với mỗi mô hình, lấy biến thể prompt tốt nhất — đúng quy trình P1."""
    best = runs.loc[runs.groupby("model")["mcc_macro"].idxmax()].copy()
    agg = runs.groupby("model").agg(
        n_variants=("variant", "size"),
        auc_min=("roc_auc", "min"),
        auc_max=("roc_auc", "max"),
        n_below_chance=("roc_auc", lambda s: int((s < 0.5).sum())),
        max_positive_rate=("pred_positive_rate", "max"),
    )
    out = best.merge(agg, on="model").sort_values("size_b")
    out["degenerate"] = out["max_positive_rate"] > 0.90
    return out


def prompt_language_effect(runs: pd.DataFrame) -> pd.DataFrame:
    """So sánh prompt tiếng Anh với tiếng Việt, tách MCC khỏi ROC-AUC.

    Vì sao phải tách: MCC ở đây phụ thuộc ngưỡng, mà ngưỡng lại hiệu chỉnh
    in-sample. ROC-AUC thì **không phụ thuộc ngưỡng** — nó chỉ đo chất lượng
    *xếp hạng*. Nếu một biến thể hơn về MCC nhưng không hơn về AUC thì cái
    "hơn" đó chỉ là may mắn về hiệu chỉnh, không phải hiểu tác vụ tốt hơn.

    Chỉ so các cặp **cùng prompt, cùng mô hình, khác ngôn ngữ hướng dẫn**.
    """
    runs = runs.copy()
    runs["prompt_id"] = runs["variant"].str.split("[").str[0]

    wide = runs.pivot_table(
        index=["size_b", "prompt_id"], columns="language",
        values=["mcc_macro", "roc_auc"],
    ).dropna()
    if wide.empty:
        return wide

    out = pd.DataFrame({
        "mcc_en": wide[("mcc_macro", "en")],
        "mcc_vi": wide[("mcc_macro", "vi")],
        "auc_en": wide[("roc_auc", "en")],
        "auc_vi": wide[("roc_auc", "vi")],
    })
    out["mcc_delta"] = out["mcc_vi"] - out["mcc_en"]
    out["auc_delta"] = out["auc_vi"] - out["auc_en"]
    return out.round(3)


def plot_curve(curve: pd.DataFrame, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))

    # -- MCC theo cỡ mô hình --------------------------------------------
    ax1.plot(curve["size_b"], curve["mcc_macro"], "o-", color="#4c72b0",
             linewidth=2, markersize=8, label="Qwen2.5 (đo được)")
    for _, r in curve.iterrows():
        ax1.annotate(f"{r['variant']}", (r["size_b"], r["mcc_macro"]),
                     textcoords="offset points", xytext=(6, -12), fontsize=8)

    known = [(s, m, n) for n, (s, m) in PAPER_VALIDATION_MCC.items() if s == s]
    if known:
        ax1.scatter([k[0] for k in known], [k[1] for k in known],
                    marker="s", color="#c44e52", s=60, label="P1 Bảng 6 (công bố)")
        for s, m, n in known:
            ax1.annotate(n, (s, m), textcoords="offset points",
                         xytext=(6, 4), fontsize=8, color="#c44e52")

    ax1.set_xscale("log")
    ax1.set_xlabel("Số tham số (tỉ, thang log)")
    ax1.set_ylabel("MCC (trung bình vĩ mô)")
    ax1.set_title("Năng lực phát hiện ảo giác theo cỡ mô hình")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    # -- ROC-AUC: chỉ báo đáng tin hơn vì không phụ thuộc ngưỡng ---------
    ax2.fill_between(curve["size_b"], curve["auc_min"], curve["auc_max"],
                     alpha=0.25, color="#4c72b0", label="khoảng qua các prompt")
    ax2.plot(curve["size_b"], curve["roc_auc"], "o-", color="#4c72b0",
             linewidth=2, markersize=8, label="prompt tốt nhất")
    ax2.axhline(0.5, color="#c44e52", linestyle="--", linewidth=1.5,
                label="mức đoán bừa")
    ax2.set_xscale("log")
    ax2.set_xlabel("Số tham số (tỉ, thang log)")
    ax2.set_ylabel("ROC-AUC")
    ax2.set_title("Dưới đường đỏ = tệ hơn đoán bừa")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    fig.suptitle("HalOmi validation (301 câu DE↔EN, 9,6% ảo giác)", y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)

    runs = load_runs()
    curve = build_curve(runs)
    curve.to_csv(RESULTS / "model_scaling.csv", index=False)

    print("=" * 88)
    print("NĂNG LỰC PHÁT HIỆN ẢO GIÁC THEO CỠ MÔ HÌNH — HalOmi validation")
    print("=" * 88)
    print()
    cols = ["model", "size_b", "variant", "mcc_macro", "roc_auc",
            "auc_min", "auc_max", "n_below_chance", "max_positive_rate", "degenerate"]
    print(curve[cols].to_string(index=False))
    print()

    print("-" * 88)
    print("ĐỌC KẾT QUẢ")
    print("-" * 88)
    for _, r in curve.iterrows():
        verdict = (
            "KHONG HIEU TAC VU" if r["degenerate"] or r["n_below_chance"] >= 2
            else "co nang luc thuc su"
        )
        print(f"  {r['model']:<24} {r['size_b']:>5.1f}B  MCC {r['mcc_macro']:.3f}  "
              f"AUC {r['roc_auc']:.3f}  -> {verdict}")
        if r["degenerate"]:
            print(f"{'':>26} tra loi co dinh: co prompt doan 'ao giac' cho "
                  f"{100*r['max_positive_rate']:.0f}% dau vao")
        if r["n_below_chance"]:
            print(f"{'':>26} {r['n_below_chance']}/{r['n_variants']} bien the co "
                  f"AUC < 0.5 (te hon doan bua)")
    print()

    print("-" * 88)
    print("ĐỐI CHIẾU VỚI P1 (cùng tập validation, cùng độ đo — Bảng 6)")
    print("-" * 88)
    for name, (size, mcc) in sorted(PAPER_VALIDATION_MCC.items(), key=lambda x: x[1][1]):
        size_txt = f"{size:.0f}B" if size == size else "n/a"
        print(f"  {name:<18} {size_txt:>6}   MCC {mcc:.2f}")
    print()
    best_local = curve.iloc[-1]
    print(f"  Mo hinh lon nhat ta chay duoc tren CPU: {best_local['model']} "
          f"({best_local['size_b']:.1f}B), MCC {best_local['mcc_macro']:.3f}")
    print("  -> Khoang trong toi 0.81 cua Llama3-70B chinh la thu can GPU de lap day.")
    print()

    effect = prompt_language_effect(runs)
    if not effect.empty:
        effect.to_csv(RESULTS / "prompt_language_effect.csv")
        print("-" * 88)
        print("PROMPT TIENG ANH vs TIENG VIET (chi cac cap cung prompt, cung mo hinh)")
        print("-" * 88)
        print(effect.to_string())
        print()
        n_mcc = int((effect["mcc_delta"] > 0).sum())
        n_auc = int((effect["auc_delta"] > 0).sum())
        print(f"  Tieng Viet hon ve MCC     : {n_mcc}/{len(effect)} cap")
        print(f"  Tieng Viet hon ve ROC-AUC : {n_auc}/{len(effect)} cap")
        print()
        print("  MCC phu thuoc nguong (hieu chinh in-sample); AUC thi khong.")
        print("  Hon MCC ma khong hon AUC = chi may man ve hieu chinh, khong phai")
        print("  hieu tac vu tot hon. Gia thuyet 'prompt ban ngu giup ich' chi kiem")
        print("  dinh duoc tren du lieu TIENG VIET that (ViHalluMT), khong phai o day.")
        print()

    plot_curve(curve, FIGURES / "model_scaling.png")
    print(f"Da ghi: {RESULTS / 'model_scaling.csv'}")
    if not effect.empty:
        print(f"Da ghi: {RESULTS / 'prompt_language_effect.csv'}")
    print(f"Da ghi: {FIGURES / 'model_scaling.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
