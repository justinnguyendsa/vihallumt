"""Sinh notebook Kaggle để chạy phần cần GPU của ViHalluMT.

Viết notebook bằng script thay vì gõ tay JSON: ít lỗi cú pháp hơn, và mỗi lần
đổi nội dung chỉ cần chạy lại là notebook luôn khớp với mã nguồn.

Chạy:  python scripts/make_kaggle_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "01_build_corpus_kaggle.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().split("\n")}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().split("\n"),
    }


CELLS = [
    md("""
# ViHalluMT — Sinh ngữ liệu tiếng Việt (Kaggle / Colab)

Notebook này chạy phần **cần GPU** của đồ án: dịch câu nguồn bằng nhiều hệ dịch
và nhiều cấu hình giải mã, chấm điểm sơ bộ, rồi lấy mẫu phân tầng ra tập cần
gán nhãn tay.

**Đầu ra:**
- `candidates.jsonl` — toàn bộ ứng viên kèm điểm
- `to_annotate.jsonl` — mẫu đã chọn, tải về máy để gán nhãn
- `corpus_a_stats.csv` — thống kê cho báo cáo

**Thời gian ước tính:** 30–50 phút trên 1×T4 với 3.000 câu nguồn mỗi hướng.

> **Bật GPU trước khi chạy:** Kaggle → *Settings → Accelerator → GPU T4 x2*.
> Colab → *Runtime → Change runtime type → T4 GPU*.
"""),

    md("## 1. Cài đặt thư viện và cấu hình môi trường"),

    code("""
import subprocess, sys, os

def sh(cmd):
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=False)

# bitsandbytes la BAT BUOC: LLMTranslator (Qwen-7B) dung luong tu hoa 4-bit.
# Thieu no thi lenh dich se vo GIUA CHUNG, sau khi da ton hang chuc phut GPU.
sh(f"{sys.executable} -m pip install -q -U transformers accelerate sentencepiece")
sh(f"{sys.executable} -m pip install -q -U bitsandbytes")
sh(f"{sys.executable} -m pip install -q sentence-transformers datasets underthesea")
"""),

    code("""
import torch, transformers, importlib
print("torch       :", torch.__version__)
print("transformers:", transformers.__version__)
print("CUDA        :", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}  {p.total_memory/1e9:.1f} GB")
else:
    print("  !! Chua bat GPU — buoc dich se rat cham. Hay bat GPU roi chay lai.")

# Kiem tra cac goi bat buoc CO THAT SU import duoc, khong chi la pip bao thanh cong
print()
missing = []
for mod, why in [("bitsandbytes", "luong tu hoa 4-bit cho Qwen-7B"),
                 ("sentence_transformers", "cham diem LaBSE"),
                 ("datasets", "nap OPUS-100 / IWSLT")]:
    try:
        importlib.import_module(mod)
        print(f"  OK   {mod}")
    except Exception as e:
        missing.append(mod)
        print(f"  HONG {mod:<22} ({why}) -> {type(e).__name__}")
if missing:
    raise SystemExit(f"Thieu goi: {missing}. Chay lai o cai dat ben tren.")
"""),

    md("""
### Lấy mã nguồn

Chọn **một** trong hai cách:

**Cách A — clone từ GitHub.** `REPO_URL` đã điền sẵn kho của đồ án.

> Kho đang để **private**, nên `git clone` trên Kaggle sẽ treo vì đòi mật khẩu.
> Trước khi chạy, tạm chuyển sang public:
> ```bash
> gh repo edit justinnguyendsa/vihallumt --visibility public --accept-visibility-change-consequences
> ```
> Chạy xong thì chuyển lại private bằng lệnh tương tự với `--visibility private`.
> Không muốn đổi thì dùng **Cách B**.

**Cách B — tải lên Kaggle Dataset.** Nén thư mục `src/` và `scripts/` thành
`.zip`, tạo một Kaggle Dataset mới, rồi *Add Data* vào notebook này. Sau đó sửa
`LOCAL_SRC` cho trỏ đúng đường dẫn.
"""),

    code("""
REPO_URL  = "https://github.com/justinnguyendsa/vihallumt.git"
LOCAL_SRC = "/kaggle/input/vihallumt-src"                     # dung cho Cach B

import os, sys, shutil
from pathlib import Path

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
PROJECT = WORK / "vihallumt"

# Checkpoint phai nam NGOAI PROJECT: o duoi xoa sach PROJECT truoc khi clone,
# nen checkpoint de ben trong se bi xoa dung luc can dung nhat.
CKPT_DIR = WORK / "vihallumt_checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
os.environ["VIHALLUMT_CKPT"] = str(CKPT_DIR)

if REPO_URL and "<" not in REPO_URL:
    if PROJECT.exists():
        shutil.rmtree(PROJECT)
    os.system(f"git clone --depth 1 {REPO_URL} {PROJECT}")
elif Path(LOCAL_SRC).exists():
    PROJECT.mkdir(parents=True, exist_ok=True)
    for sub in ("src", "scripts"):
        if (Path(LOCAL_SRC) / sub).exists():
            shutil.copytree(Path(LOCAL_SRC) / sub, PROJECT / sub, dirs_exist_ok=True)
else:
    raise SystemExit(
        "Chua co ma nguon. Hay dat REPO_URL (Cach A) hoac them Kaggle Dataset (Cach B)."
    )

sys.path.insert(0, str(PROJECT / "src"))
os.chdir(PROJECT)
print("Thu muc lam viec:", Path.cwd())
print("Cac tep:", sorted(p.name for p in Path.cwd().iterdir()))
"""),

    code("""
# Kiem tra import duoc goi vihallumt
from vihallumt.corpus.translate import DEFAULT_GENERATION_PLAN, probe_translator
from vihallumt.corpus.sampling import stratified_sample
print("Import OK. Ke hoach sinh du lieu:")
for s in DEFAULT_GENERATION_PLAN:
    print(f"  {s.tag:<28} {s.share:.0%}")
"""),

    md("""
## 2. Kiểm tra các hệ dịch trước khi tốn GPU

Bước này chỉ nạp *tokenizer* (rẻ, nhanh) để phát hiện sớm mô hình không dùng
được. Thà biết ngay từ đầu còn hơn đổ sau 20 phút chạy GPU.

`envit5` và `vinai-translate` dựa trên tokenizer sentencepiece và **vỡ với
transformers ≥ 5.0**. Nếu chúng báo hỏng ở đây, script sẽ tự bỏ qua và chia lại
tỉ trọng — nhớ ghi việc này vào mục *Limitations* của báo cáo.
"""),

    code("""
for name in sorted({s.translator for s in DEFAULT_GENERATION_PLAN}):
    ok, msg = probe_translator(name)
    print(f"  {name:<12} {'OK' if ok else 'HONG  ' + msg}")
"""),

    md("""
## 3. (Tuỳ chọn) Mở khoá FLORES-200

FLORES là dataset **gated**. Có nó thì kết quả tiếng Việt so sánh trực tiếp
được với 18 hướng dịch của HalOmi, vì HalOmi cũng lấy câu nguồn từ FLORES.

1. Đồng ý điều khoản tại https://huggingface.co/datasets/openlanguagedata/flores_plus
2. Tạo token tại https://huggingface.co/settings/tokens
3. Trên Kaggle: *Add-ons → Secrets* → thêm secret tên `HF_TOKEN`

Bỏ qua ô này cũng được — đường ống vẫn chạy với OPUS-100 + IWSLT.
"""),

    code("""
USE_FLORES = False   # <-- dat True sau khi da them HF_TOKEN

if USE_FLORES:
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        print("Da nap HF_TOKEN tu Kaggle Secrets")
    except Exception as e:
        print("Khong lay duoc secret:", e)
        print("Dat tay: os.environ['HF_TOKEN'] = '...'")
"""),

    md("""
## 4. Sinh ngữ liệu

Chỉnh `N_SOURCE` và `N_ANNOTATE` cho khớp với sức gán nhãn của bạn.
Mặc định 3.000 câu nguồn mỗi hướng → 700 cặp đưa đi gán nhãn.

Muốn thử nhanh trước thì đặt `SMOKE = True` (chỉ vài phút).
"""),

    code("""
SMOKE      = False
N_SOURCE   = 3000
N_ANNOTATE = 700
DIRECTIONS = "en-vi,vi-en"
BATCH_SIZE = 32      # T4 chiu duoc 32; giam xuong neu bao het bo nho

args = [
    sys.executable, "scripts/build_corpus_a.py",
    "--n-source", str(200 if SMOKE else N_SOURCE),
    "--n-annotate", str(40 if SMOKE else N_ANNOTATE),
    "--directions", DIRECTIONS,
    "--batch-size", str(BATCH_SIZE),
    "--plan", "smoke" if SMOKE else "default",
    "--ckpt-dir", str(CKPT_DIR),
]
if not USE_FLORES:
    args.append("--no-flores")

print(" ".join(args), "\\n")

# Cho biet co the dung lai duoc gi tu lan chay truoc
existing = sorted(CKPT_DIR.glob("*.jsonl"))
if existing:
    print(f"Da co {len(existing)} checkpoint tu lan chay truoc — se dung lai:")
    for f in existing:
        print("   ", f.name)
else:
    print("Chua co checkpoint — dich tu dau.")
print()

subprocess.run(args, check=True)
"""),

    md("## 5. Kiểm tra kết quả"),

    code("""
import pandas as pd

cand = pd.read_json("data/vihallumt/candidates.jsonl", lines=True)
ann  = pd.read_json("data/vihallumt/to_annotate.jsonl", lines=True)

print(f"Ung vien       : {len(cand)}")
print(f"Cho gan nhan   : {len(ann)}")
print()
print("Theo huong dich:")
print(cand.groupby(["direction", "perturbation"]).size().to_string())
print()
print("Theo he dich:")
print(cand["gen_tag"].value_counts().to_string())
print()
print("Phan tang cua mau da chon:")
print(ann.groupby("selection")[["agg_score"]].agg(["size", "mean"]).round(3).to_string())
"""),

    code("""
# Xem thu vai ban dich diem cao nhat — day la cac ca nghi ngo ao giac
top = ann.nlargest(5, "agg_score")
for _, r in top.iterrows():
    print("-" * 78)
    print("NGUON :", r["src_text_original"][:150])
    print("DICH  :", r["mt_text"][:150])
    print(f"        ({r['gen_tag']}, tang={r['selection']}, diem={r['agg_score']:.3f})")
"""),

    code("""
# Kiem tra suc khoe du lieu truoc khi dem di gan nhan
issues = []
if ann["mt_text"].str.strip().eq("").any():
    issues.append("co ban dich rong")
if ann["cand_id"].duplicated().any():
    issues.append("co cand_id trung lap")
if ann["selection"].nunique() < 3:
    issues.append("thieu tang lay mau")
if (ann["mt_text"] == ann["src_text_original"]).mean() > 0.1:
    issues.append(">10% ban dich trung khit cau nguon (co the he dich khong dich)")

print("VAN DE PHAT HIEN:", issues if issues else "khong co")
"""),

    md("""
## 6. Tải kết quả về máy

Trên Kaggle, mọi tệp trong `/kaggle/working` đều tải về được ở tab *Output* sau
khi notebook chạy xong. Ô dưới chép kết quả ra đúng chỗ đó.

Tải hai tệp này về, đặt vào `data/vihallumt/` trong kho mã nguồn ở máy bạn, rồi
chạy công cụ gán nhãn:

```bash
python scripts/annotate.py
```
"""),

    code("""
import shutil
from pathlib import Path

dest = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
for name in ["data/vihallumt/candidates.jsonl",
             "data/vihallumt/to_annotate.jsonl",
             "results/corpus_a_stats.csv"]:
    src = Path(name)
    if src.exists():
        target = dest / src.name
        if src.resolve() != target.resolve():
            shutil.copy(src, target)
        print(f"  {target}  ({src.stat().st_size/1024:.0f} KB)")
    else:
        print(f"  !! thieu {name}")
"""),

    md("""
## 7. Bước tiếp theo

1. Tải `to_annotate.jsonl` về máy.
2. Đọc kỹ [`docs/annotation-guideline-vi.md`](../docs/annotation-guideline-vi.md).
3. Gán thử 50 cặp, chỉnh guideline theo ca khó gặp thực tế, rồi mới gán tiếp.
4. `python scripts/annotate.py` — khoảng 150 cặp mỗi buổi.
5. `python scripts/annotate.py --review` để xem thống kê nhãn đã gán.
"""),
]


def main() -> int:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")

    n_code = sum(1 for c in CELLS if c["cell_type"] == "code")
    print(f"Da ghi {OUT}")
    print(f"  {len(CELLS)} o ({n_code} o ma nguon)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
