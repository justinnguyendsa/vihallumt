# ViHalluMT — Phát hiện ảo giác trong dịch máy cho tiếng Việt

Đồ án cuối kỳ môn Xử lý ngôn ngữ tự nhiên (Thạc sĩ Khoa học Dữ liệu).
Đề tài #146: *Mô hình ngôn ngữ lớn cho bài toán phát hiện ảo giác trong dịch máy*.

Công trình tái lập Benkirane et al. (Findings of EMNLP 2024) trên benchmark
HalOmi, rồi mở rộng sang tiếng Việt bằng bộ ngữ liệu `ViHalluMT` tự xây.

| | |
|---|---|
| Đề cương chi tiết | [`docs/DE-CUONG.md`](docs/DE-CUONG.md) |
| Paper tham khảo | `papers/2024.findings-emnlp.564.pdf`, `papers/2025.naacl-long.175.pdf` |
| Trạng thái | Tuần 1 — đã tái lập baseline paper gốc; đã có 4 bộ phát hiện và bộ sinh ngữ liệu |
| Kiểm thử | 370 test, chạy hết trên CPU trong ~3 giây |

---

## Cài đặt

```bash
pip install -r requirements.txt
pip install -e .
```

Phần tái lập HalOmi **không cần GPU và không cần `torch`** — chỉ cần
`pandas`, `scikit-learn`, `scipy`, `matplotlib`.

## Tải dữ liệu

```bash
bash scripts/download_halomi.sh
```

Tải HalOmi (Dale et al., 2023) từ `dl.fbaipublicfiles.com` (19,9 MB) và giải
nén vào `data/raw/`. Script tự bỏ qua nếu dữ liệu đã có.

## Chạy phần tái lập

```bash
python scripts/eda_halomi.py
```

Đối chiếu phân bố nhãn với Bảng 1–3 của paper, ghi `results/halomi_distribution.csv`,
`results/halomi_severity.csv` và hai hình vào `results/figures/`.

```bash
python scripts/replicate_p1_baselines.py
```

Hiệu chỉnh ngưỡng trên tập validation (DE↔EN) rồi đánh giá 8 baseline phi-LLM
trên tập test, ghi `results/halomi_baselines.csv`.

## Chạy kiểm thử

```bash
python -m pytest tests/
```

370 test, trong đó 33 test kiểm chứng bộ nạp dữ liệu tái tạo **đúng từng con số**
trong Bảng 1, 2, 3 của paper gốc (cả 18 hướng dịch). Toàn bộ chạy trên CPU —
phần cần GPU được tách thành hàm thuần tuý và kiểm thử bằng mô hình giả.

---

## Kết quả tái lập

Bảng dưới là kết quả `scripts/replicate_p1_baselines.py` trên 2.564 câu của tập
test HalOmi, ngưỡng lấy từ 301 câu DE↔EN.

| Detector | MCC (macro) | HRL | LRL | ROC-AUC |
|---|---:|---:|---:|---:|
| **BLASER-2.0-QE** | **0.374** | **0.466** | 0.282 | 0.778 |
| SONAR cosine | 0.360 | 0.520 | 0.199 | 0.790 |
| ALTI+ | 0.337 | 0.562 | 0.112 | 0.700 |
| Seq log-loss | 0.336 | 0.550 | 0.123 | 0.728 |
| LaBSE cosine | 0.334 | 0.523 | 0.144 | **0.825** |
| LASER cosine | 0.284 | 0.435 | 0.134 | 0.697 |
| COMET-QE | 0.265 | 0.451 | 0.078 | 0.711 |
| XNLI | 0.219 | 0.388 | 0.050 | 0.704 |

Đối chiếu với số công bố trong paper:

| | Ta | Paper | Lệch |
|---|---:|---:|---:|
| BLASER-2.0-QE, tổng thể | 0.374 | 0.38 | 0.006 |
| BLASER-2.0-QE, riêng HRL | 0.466 | 0.46 | 0.006 |

BLASER-2.0-QE đứng đầu nhóm baseline phi-LLM, đúng như paper mô tả nó là SOTA
trước đó.

### Hai ghi chú kỹ thuật quan trọng

**1. Cách tổng hợp MCC.** Paper không nói rõ trong thân bài rằng MCC của họ là
*trung bình theo hướng dịch*; chỉ caption Figure 2 ngụ ý điều đó. Xác định được
bằng thực nghiệm:

| Cách tổng hợp | Tổng thể | Riêng HRL |
|---|---:|---:|
| MCC gộp toàn tập | 0.317 | 0.477 |
| MCC trung bình vĩ mô | **0.374** | **0.466** |
| Paper công bố | 0.38 | 0.46 |

Mọi so sánh với bảng của paper phải dùng `vihallumt.eval.macro_average`, không
dùng MCC gộp.

**2. Quy ước dấu của điểm số.** Mọi cột `score_*` trong HalOmi đã được lưu sẵn
dưới dạng *điểm ảo giác* (càng cao càng nhiều ảo giác), tức là giá trị đã đảo
dấu với các độ đo tương đồng. Đảo dấu thêm lần nữa sẽ âm thầm biến detector tốt
thành tệ hơn đoán bừa. Quy ước này được chốt bằng test.

Nhờ HalOmi đã tính sẵn cột `score_blaser2_qe`, ta **tái lập được baseline SOTA
mà không cần cài `fairseq2`/`sonar-space`** — vốn là rủi ro kỹ thuật lớn nhất
trong kế hoạch ban đầu.

---

## Cấu trúc kho mã

```
FinalProject/
├── docs/DE-CUONG.md              # đề cương chi tiết, kế hoạch 3 tuần
├── papers/                       # hai paper tham khảo
├── data/raw/                     # HalOmi (tải bằng script, không commit)
├── src/vihallumt/
│   ├── data.py                   # nạp HalOmi, nhị phân hoá nhãn, chia tập
│   ├── eval/metrics.py           # MCC, macro-average, bootstrap CI, McNemar
│   ├── prompts/binary.py         # 3 prompt + 2 CoT của P1 + biến thể tiếng Việt
│   ├── detectors/
│   │   ├── base.py               # giao diện chung, quy ước dấu
│   │   ├── llm.py                # LLM + logit-scoring
│   │   ├── embed.py              # cosine embedding đa ngữ & chuyên Việt
│   │   └── ngram.py              # đếm lặp n-gram (Raunak et al.)
│   ├── corpus/
│   │   ├── sources.py            # nạp OPUS-100 / IWSLT'15 / FLORES
│   │   ├── translate.py          # 3 hệ dịch × 5 cấu hình giải mã
│   │   ├── perturb.py            # nhiễu câu nguồn (nhánh perturbed)
│   │   ├── synthetic.py          # ảo giác tổng hợp có kiểm soát (Tập B)
│   │   ├── sampling.py           # lấy mẫu phân tầng uniform/biased/worst
│   │   └── probe_vi.py           # Tập C — cặp tối thiểu tiếng Việt
│   └── mitigation/               # post-hoc theo paper 2   (chưa làm)
├── scripts/
│   ├── download_halomi.sh
│   ├── eda_halomi.py
│   ├── replicate_p1_baselines.py
│   ├── run_llm_detector.py
│   ├── build_corpus_a.py         # sinh Tập A (cần GPU)
│   ├── annotate.py               # công cụ gán nhãn tay
│   └── make_kaggle_notebook.py
├── tests/                        # 370 test
├── notebooks/
│   └── 01_build_corpus_kaggle.ipynb
├── results/                      # bảng số + hình
└── paper/                        # báo cáo ACL             (chưa làm)
```

## Quy trình xây ngữ liệu tiếng Việt

```
                 [Kaggle / Colab, cần GPU]           [máy cá nhân, không cần GPU]
 OPUS-100 ─┐
 IWSLT'15 ─┼─> lọc & khử trùng ─> dịch bằng     ─> chấm điểm ─> lấy mẫu ─> gán nhãn tay
 FLORES*  ─┘   (sources.py)      3 hệ × 5 cách    (LaBSE,      phân tầng   (annotate.py)
                                  giải mã          n-gram)     (sampling)
                                 (translate.py)
```

`*` FLORES là dataset gated — cần đồng ý điều khoản và có `HF_TOKEN`. Bỏ qua
cũng được, đường ống vẫn chạy đủ với hai nguồn còn lại.

**Bước 1 — sinh ứng viên (cần GPU, ~30–50 phút trên T4):**

Mở [`notebooks/01_build_corpus_kaggle.ipynb`](notebooks/01_build_corpus_kaggle.ipynb)
trên Kaggle hoặc Colab. Hoặc chạy trực tiếp nếu máy có GPU:

```bash
python scripts/build_corpus_a.py --n-source 3000 --n-annotate 700
```

**Bước 2 — gán nhãn tay (không cần GPU):**

```bash
python scripts/annotate.py
```

Đọc [`docs/annotation-guideline-vi.md`](docs/annotation-guideline-vi.md) trước.
Công cụ tự lưu sau mỗi nhãn, xáo thứ tự để tránh hiệu ứng mỏ neo, và **ẩn điểm
của máy** để nhãn người không bị kéo theo dự đoán của máy.

```bash
python scripts/annotate.py --review
```

## Bộ phát hiện đã cài đặt

| Bộ | Tệp | Ghi chú |
|---|---|---|
| LLM (logit-scoring) | `detectors/llm.py` | 1 lượt truyền xuôi → điểm liên tục; nhanh hơn ~10× cách sinh chuỗi của P1 |
| LLM (sinh chuỗi) | `detectors/llm.py` | Cách gốc của P1; **bắt buộc** cho biến thể CoT |
| Embedding cosine | `detectors/embed.py` | LaBSE, mE5, mpnet + 2 mô hình chuyên tiếng Việt |
| Đếm lặp n-gram | `detectors/ngram.py` | Raunak et al.; tham số n=4, ngưỡng=2 theo P2 |

**Vì sao CoT không dùng được logit-scoring:** CoT cần mô hình viết ra phần lập
luận trước, nên token đầu tiên là chữ đầu của lập luận chứ không phải nhãn.
`LLMConfig` chặn tổ hợp này ngay lúc khởi tạo thay vì cho ra số vô nghĩa.

## Nguồn dữ liệu

| Nguồn | Cấp phép / cách lấy |
|---|---|
| HalOmi (Dale et al., 2023) | `https://dl.fbaipublicfiles.com/nllb/halomi_release_v2.zip` |
| FLORES-200 | HF `openlanguagedata/flores_plus` — **gated**, cần đồng ý điều khoản + `HF_TOKEN` |
| OPUS-100 en-vi | HF `Helsinki-NLP/opus-100`, config `en-vi` |
| IWSLT'15 en-vi | HF `mt_eng_vietnamese` |

## Trích dẫn

```bibtex
@inproceedings{benkirane-etal-2024-machine,
  title     = {Machine Translation Hallucination Detection for Low and High
               Resource Languages using Large Language Models},
  author    = {Benkirane, Kenza and Gongas, Laura and Pelles, Shahar and
               Fuchs, Naomi and Darmon, Joshua and Stenetorp, Pontus and
               Adelani, David Ifeoluwa and S{\'a}nchez, Eduardo},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2024},
  year      = {2024},
  pages     = {9647--9665}
}

@inproceedings{dale-etal-2023-halomi,
  title     = {{HalOmi}: A Manually Annotated Benchmark for Multilingual
               Hallucination and Omission Detection in Machine Translation},
  author    = {Dale, David and Voita, Elena and Lam, Janice and Hansanti, Prangthip and
               Ropers, Christophe and Kalbassi, Elahe and Gao, Cynthia and
               Barrault, Loic and Costa-juss{\`a}, Marta R.},
  booktitle = {Proceedings of EMNLP 2023},
  year      = {2023}
}

@inproceedings{tang-etal-2025-mitigating,
  title     = {Mitigating Hallucinated Translations in Large Language Models
               with Hallucination-focused Preference Optimization},
  author    = {Tang, Zilu and Chatterjee, Rajen and Garg, Sarthak},
  booktitle = {Proceedings of NAACL 2025},
  year      = {2025},
  pages     = {3410--3433}
}
```
