"""Khám phá HalOmi và đối chiếu với số liệu công bố trong P1.

Sinh ra:
  results/halomi_distribution.csv  — phân bố nhãn theo hướng dịch
  results/halomi_severity.csv      — phân bố mức nghiêm trọng
  results/figures/halomi_*.png     — hình minh hoạ

Chạy:  python scripts/eda_halomi.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from vihallumt.data import SEVERITY_ORDER, load_halomi, split_halomi

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

# Số liệu công bố trong P1 Bảng 2 & 3 — dùng để đối chiếu tự động.
PAPER_BINARY: dict[str, tuple[int, int, int]] = {
    # direction: (tổng, không ảo giác, có ảo giác)
    "DE-EN": (155, 140, 15), "EN-DE": (146, 132, 14),
    "EN-AR": (144, 136, 8), "AR-EN": (156, 132, 24),
    "EN-RU": (146, 141, 5), "RU-EN": (158, 146, 12),
    "EN-ES": (153, 131, 22), "ES-EN": (160, 127, 33),
    "EN-ZH": (160, 131, 29), "ZH-EN": (159, 127, 32),
    "EN-KS": (184, 111, 73), "KS-EN": (151, 89, 62),
    "EN-YO": (195, 166, 29), "YO-EN": (146, 124, 22),
    "EN-MN": (197, 78, 119), "MN-EN": (152, 43, 109),
    "ES-YO": (151, 97, 54), "YO-ES": (152, 80, 72),
}


def build_distribution_table(df: pd.DataFrame) -> pd.DataFrame:
    """Bảng phân bố nhãn theo hướng dịch, kèm cột đối chiếu với paper."""
    rows = []
    for direction, sub in df.groupby("direction", sort=False):
        n = len(sub)
        n_no = int((sub["label"] == 0).sum())
        n_hall = int((sub["label"] == 1).sum())
        exp_n, exp_no, exp_hall = PAPER_BINARY[direction]
        rows.append({
            "direction": direction,
            "split": sub["split"].iloc[0],
            "resource_level": sub["resource_level"].iloc[0],
            "direction_group": sub["direction_group"].iloc[0],
            "n": n,
            "n_no_hallucination": n_no,
            "n_hallucination": n_hall,
            "pct_hallucination": round(100 * n_hall / n, 2),
            "paper_n": exp_n,
            "paper_n_hallucination": exp_hall,
            "matches_paper": (n == exp_n) and (n_no == exp_no) and (n_hall == exp_hall),
        })
    out = pd.DataFrame(rows).sort_values(["split", "resource_level", "direction"])
    return out.reset_index(drop=True)


def build_severity_table(df: pd.DataFrame) -> pd.DataFrame:
    """Bảng phân bố 4 mức nghiêm trọng (đối chiếu P1 Bảng 1)."""
    tab = (
        df.pivot_table(index="direction", columns="class_hall", values="src_text",
                       aggfunc="count", fill_value=0)
        .reindex(columns=list(SEVERITY_ORDER), fill_value=0)
    )
    tab["total"] = tab.sum(axis=1)
    tab = tab.reset_index()
    split_map = df.drop_duplicates("direction").set_index("direction")["split"]
    tab.insert(1, "split", tab["direction"].map(split_map))
    return tab.sort_values(["split", "direction"]).reset_index(drop=True)


def plot_hallucination_rate(dist: pd.DataFrame, path: Path) -> None:
    """Tỉ lệ ảo giác theo hướng dịch — cho thấy mức mất cân bằng rất khác nhau."""
    d = dist.sort_values("pct_hallucination")
    colors = ["#c44e52" if r == "LRL" else "#4c72b0" for r in d["resource_level"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(d["direction"], d["pct_hallucination"], color=colors)
    ax.set_xlabel("Tỉ lệ câu có ảo giác (%)")
    ax.set_title("HalOmi: tỉ lệ ảo giác theo hướng dịch (chỉ bản dịch tự nhiên)")
    ax.axvline(50, color="grey", linestyle=":", linewidth=1)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#4c72b0"),
        plt.Rectangle((0, 0), 1, 1, color="#c44e52"),
    ]
    ax.legend(handles, ["Tài nguyên cao (HRL)", "Tài nguyên thấp (LRL)"], loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_score_separation(df: pd.DataFrame, path: Path) -> None:
    """Phân bố điểm của các detector tính sẵn, tách theo nhãn."""
    cols = [
        ("score_blaser2_qe", "BLASER-2.0-QE"),
        ("score_labse", "LaBSE"),
        ("score_sonar_cosine", "SONAR cosine"),
        ("score_comet_qe", "COMET-QE"),
    ]
    fig, axes = plt.subplots(1, len(cols), figsize=(16, 3.6))
    for ax, (col, name) in zip(axes, cols):
        for label, colour, tag in [(0, "#4c72b0", "Không ảo giác"), (1, "#c44e52", "Có ảo giác")]:
            vals = df.loc[df["label"] == label, col].dropna()
            ax.hist(vals, bins=40, alpha=0.6, color=colour, label=tag, density=True)
        ax.set_title(name)
        ax.set_xlabel("điểm ảo giác (cao = ảo giác)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mật độ")
    axes[0].legend()
    fig.suptitle("HalOmi: khả năng tách hai lớp của các detector có sẵn", y=1.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    df = load_halomi()
    val, test = split_halomi(df)

    print("=" * 78)
    print("HALOMI — ĐỐI CHIẾU VỚI SỐ LIỆU CÔNG BỐ TRONG P1")
    print("=" * 78)
    print(f"Tổng số cặp tự nhiên : {len(df):>6}   (paper: 2865)")
    print(f"Tập validation       : {len(val):>6}   (paper:  301, DE<->EN)")
    print(f"Tập test             : {len(test):>6}   (paper: 2564; báo cáo 2558 sau lọc)")
    print(f"Số hướng dịch        : {df['direction'].nunique():>6}   (paper:   18)")
    print()

    dist = build_distribution_table(df)
    sev = build_severity_table(df)

    dist.to_csv(RESULTS / "halomi_distribution.csv", index=False)
    sev.to_csv(RESULTS / "halomi_severity.csv", index=False)

    n_match = int(dist["matches_paper"].sum())
    print(f"Số hướng dịch khớp hoàn toàn với paper: {n_match}/{len(dist)}")
    if n_match < len(dist):
        print("\nCÁC HƯỚNG KHÔNG KHỚP:")
        print(dist[~dist["matches_paper"]].to_string(index=False))
    print()

    print("-" * 78)
    print("Phân bố nhãn theo hướng dịch")
    print("-" * 78)
    show = dist[["direction", "split", "resource_level", "n",
                 "n_no_hallucination", "n_hallucination", "pct_hallucination"]]
    print(show.to_string(index=False))
    print()

    print("-" * 78)
    print("Phân bố mức nghiêm trọng theo tập (đối chiếu P1 Bảng 1)")
    print("-" * 78)
    for split_name, expected in [("validation", [272, 5, 4, 20]), ("test", [1859, 220, 287, 198])]:
        sub = df[df["split"] == split_name]
        counts = sub["severity"].value_counts().sort_index().tolist()
        flag = "OK" if counts == expected else "LECH"
        print(f"  {split_name:<11} {counts}   paper: {expected}   [{flag}]")
    print()

    print("-" * 78)
    print("Mất cân bằng lớp theo mức tài nguyên (P1 mục Limitations)")
    print("-" * 78)
    for level in ("HRL", "LRL"):
        sub = test[test["resource_level"] == level]
        share = sub.groupby("direction")["label"].apply(lambda s: 100 * (s == 0).mean())
        print(f"  {level}: tỉ lệ 'không ảo giác' từ {share.min():.1f}% đến {share.max():.1f}% "
              f"(paper: HRL 79–94%)")
    print()

    plot_hallucination_rate(dist, FIGURES / "halomi_hallucination_rate.png")
    plot_score_separation(df, FIGURES / "halomi_score_separation.png")

    print(f"Đã ghi: {RESULTS / 'halomi_distribution.csv'}")
    print(f"Đã ghi: {RESULTS / 'halomi_severity.csv'}")
    print(f"Đã ghi: {FIGURES / 'halomi_hallucination_rate.png'}")
    print(f"Đã ghi: {FIGURES / 'halomi_score_separation.png'}")

    return 0 if n_match == len(dist) else 1


if __name__ == "__main__":
    raise SystemExit(main())
