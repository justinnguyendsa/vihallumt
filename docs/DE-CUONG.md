# ĐỀ CƯƠNG ĐỒ ÁN CUỐI KỲ — XỬ LÝ NGÔN NGỮ TỰ NHIÊN

**Đề tài (#146):** Mô hình ngôn ngữ lớn cho bài toán phát hiện ảo giác trong dịch máy

**Paper tham khảo:**
- **[P1]** Benkirane et al. (2024). *Machine Translation Hallucination Detection for Low and High Resource Languages using LLMs*. Findings of EMNLP 2024. → **trục chính**
- **[P2]** Tang, Chatterjee, Garg (2025). *Mitigating Hallucinated Translations in LLMs with Hallucination-focused Preference Optimization*. NAACL 2025. → **phần mở rộng**

**Tên đề xuất của công trình:** `ViHalluMT` — *Benchmark và hệ thống phát hiện ảo giác cho dịch máy Việt–Anh*

---

## 0. QUYẾT ĐỊNH ĐÃ CHỐT

| Hạng mục | Quyết định | Hệ quả |
|---|---|---|
| **LLM detector** | Mã nguồn mở 4-bit trên Colab/Kaggle **+** API free tier (Gemini / Groq) | Không dùng API trả phí. Bảng kết quả có cả cột "mô hình nhỏ tự chạy" và cột "mô hình lớn qua API". Ghi rõ hạn chế tái lập của free tier trong *Limitations*. |
| **Phạm vi** | Phát hiện (P1) **+** post-hoc mitigation (P2) | **Không** làm LoRA-CPO fine-tune. Mục mitigation gói gọn 1 bảng: fallback / epsilon-sampling + LaBSE re-rank, đo MR / HR / COMET. |
| **Gán nhãn tay** | **500–600 cặp** (Tập A) + 150 cặp (Tập C) | Tương đương 3–4 hướng dịch của HalOmi. Đủ để công bố như benchmark thật. Ước tính 8–12 giờ công. |
| **Thời gian** | **2–4 tuần** → lập lịch 3 tuần, tuần 4 là đệm | Ưu tiên P0→P4. Mitigation và ablation severity làm ở tuần 3, cắt được nếu trễ. |
| **Ngôn ngữ báo cáo** | Tiếng Anh *(mặc định — đổi được)* | Khớp `acl-style-files`, dễ trích dẫn, dễ tái dùng nếu muốn nộp hội nghị. Đổi sang tiếng Việt chỉ tốn công dịch phần văn xuôi. |

---

## 1. PHÂN TÍCH ĐỀ TÀI

### 1.1. Paper 1 (EMNLP Findings 2024) làm gì

| Hạng mục | Nội dung |
|---|---|
| Bài toán | Phân loại nhị phân mức câu: cặp (source, MT output) có ảo giác hay không |
| Dữ liệu | HalOmi (Dale et al., 2023): 18 hướng dịch, 2.865 cặp. Dev = DE↔EN (301 câu), Test = 16 hướng còn lại (2.558 cặp) |
| Nhãn | 4 mức: No / Small / Partial / Full → nhị phân hoá: No vs {Small, Partial, Full} |
| Phương pháp A | **LLM zero-shot prompting**: 3 biến thể prompt × {no-CoT, CoT}, temperature=0, max_tokens=15, nhãn ràng buộc |
| Phương pháp B | **Embedding cosine**: cos(emb(src), emb(mt)), ngưỡng tối ưu hoá F1 trên tập dev |
| Baseline (SOTA cũ) | **BLASER-2.0-QE** (dựa trên SONAR) |
| Metric | **MCC** (Matthews Correlation Coefficient) — chọn vì mất cân bằng lớp nặng; ROC-AUC cho ablation severity ranking |
| Kết quả chính | Llama3-70B: MCC 0.43 tổng thể (BLASER-QE 0.38, +5 điểm). HRL: 0.63 vs 0.46 (+16 điểm). LRL: Claude Sonnet tốt nhất (+0.03). SOTA mới ở 13/16 hướng dịch |
| Kết luận | Không có "một LLM cho mọi mức tài nguyên"; embedding rất mạnh cho HRL và chữ viết phi Latin, nhưng yếu cho LRL |

### 1.2. Paper 2 (NAACL 2025) làm gì

- `HS(x,y) = 1 − BLASER(x,y)/5`, ngưỡng `T = 0.5` → gán nhãn ảo giác **không cần người**.
- Sinh dữ liệu ưu tiên (preference) *hallucination-focused*: `y_d` = bản dịch có ảo giác của chính mô hình, `y_p` = bản dịch đã được "chữa" bằng post-hoc mitigation.
- Fine-tune bằng **CPO** (biến thể DPO) với hệ số scaling theo chênh lệch chất lượng.
- Post-hoc mitigation tốt nhất: **epsilon sampling (ε=0.02) + re-rank bằng LaBSE**, mitigation rate 99,6%.
- Giảm hallucination rate 96% (5 cặp ngôn ngữ), 89% zero-shot (3 ngôn ngữ chưa thấy).
- 58–86% ảo giác là **oscillatory** (lặp n-gram).

### 1.3. Khoảng trống → đóng góp của đồ án

| Khoảng trống | Đóng góp tương ứng |
|---|---|
| HalOmi **không có tiếng Việt** (9 ngôn ngữ: AR/ZH/EN/DE/KS/MN/RU/ES/YO) | Xây **ViHalluMT** — benchmark đầu tiên cho phát hiện ảo giác MT tiếng Việt (en↔vi) |
| P1 chỉ dùng LLM đóng, đắt tiền (GPT-4o, Claude, Command R+) | Đánh giá LLM **mã nguồn mở chạy được trên Colab/Kaggle** + LLM chuyên Đông Nam Á (SeaLLMs, Vistral) |
| P1 chỉ lấy nhãn cứng từ LLM → không có điểm liên tục | **Cải tiến:** lấy điểm từ log-prob của token nhãn → có ROC-AUC, hiệu chỉnh ngưỡng, và chỉ tốn 1 forward pass |
| P1 không thử prompt bằng ngôn ngữ đích | Thêm biến thể **prompt tiếng Việt** — kiểm định giả thuyết prompt bản ngữ giúp ích cho ngôn ngữ trung tài nguyên |
| Chưa ai đo *loại* ảo giác đặc thù tiếng Việt | Bộ **probe tiếng Việt**: dấu thanh, từ láy, đại từ thân tộc, loại từ, định dạng số/ngày |
| P2 cần A100 + hàng triệu câu | Phiên bản rút gọn: post-hoc mitigation + (tuỳ chọn) LoRA-CPO trên mô hình nhỏ |

**Định vị khoa học:** Tiếng Việt là ngôn ngữ **trung tài nguyên** (mid-resource) — nằm giữa HRL và LRL của P1. Câu hỏi nghiên cứu trung tâm:

> **RQ1.** Kết luận của Benkirane et al. có tổng quát hoá sang ngôn ngữ trung tài nguyên như tiếng Việt không? (LLM > embedding? BLASER-QE có còn là baseline mạnh?)
>
> **RQ2.** LLM mã nguồn mở cỡ 7–9B có đủ khả năng thay thế LLM API thương mại cho bài toán này trên tiếng Việt không?
>
> **RQ3.** Đặc trưng ngôn ngữ học tiếng Việt (thanh điệu, từ láy, đại từ thân tộc) ảnh hưởng thế nào đến sai sót của bộ phát hiện?
>
> **RQ4.** Một mô hình nhỏ huấn luyện có giám sát trên dữ liệu tổng hợp có cạnh tranh được với LLM zero-shot không? (rẻ hơn ~100×)

---

## 2. NGỮ LIỆU: `ViHalluMT` (2,0 điểm)

Hai hướng dịch: **en→vi** và **vi→en**. Ba tập con.

### 2.1. Tập A — Ảo giác tự nhiên (natural), có gán nhãn thủ công

Đây là phần "xương sống" về mặt khoa học, mô phỏng đúng quy trình HalOmi.

**Nguồn câu** (đã kiểm chứng tải được ngày 2026-09-01):

| Nguồn | Trạng thái | Quy mô | Vai trò |
|---|---|---|---|
| OPUS-100 `en-vi` | ✅ mở | 1M / 2k / 2k | Phụ đề, web — nhiễu, **dễ gây ảo giác** |
| IWSLT'15 `en-vi` (`thainq107/iwslt2015-en-vi`) | ✅ mở | 133k | TED Talks — câu sạch, cho lớp âm khó |
| FLORES-200 (`openlanguagedata/flores_plus`) | ⚠️ **gated** | 2.009 | Cùng nguồn với HalOmi → so sánh trực tiếp được. Cần đồng ý điều khoản + `HF_TOKEN` |
| PhoMT | ⚠️ cần điền form | 3M | Không dùng — vướng thủ tục |

Không có FLORES thì đường ống vẫn chạy đủ với OPUS-100 + IWSLT; chỉ mất tính so sánh trực tiếp với HalOmi, và điều này phải ghi vào *Limitations*.

**Cấu trúc `natural` / `perturbed` — sửa so với bản đề cương đầu**

Bản đầu coi nhiễu loạn câu nguồn là cơ chế chính để kích ảo giác. Đọc lại kỹ P1 §2.1 thì thấy sai: họ **loại bỏ toàn bộ** phần perturbed của HalOmi vì *"findings from perturbed data may not be applicable to the detection of natural hallucinations"*. Nếu Tập A chủ yếu là dữ liệu nhiễu loạn thì nó tương ứng đúng với phần mà paper gốc vứt đi.

Cấu trúc đúng, bám sát HalOmi:

| Nhánh | Câu nguồn | Vai trò |
|---|---|---|
| `natural` (80%) | **nguyên vẹn** | **Bảng kết quả chính** |
| `perturbed` (20%) | bị làm nhiễu | Chỉ dùng phân tích; loại khỏi kết quả chính, đúng như P1 loại phần tương ứng của HalOmi |

**Ba cơ chế làm giàu mẫu dương** (đều giữ nguyên tính tự nhiên của bản dịch):

1. **Lấy mẫu phân tầng** `uniform`/`biased`/`worst` — đã **đo trên HalOmi**: tầng `worst` chứa **85,7%** ảo giác so với 25,6% của phân bố gốc (gấp 3,3 lần). Đây là cơ chế chính.
2. **Miền văn bản nhiễu** — OPUS-100 (phụ đề) sinh nhiều ảo giác hơn IWSLT (TED Talks).
3. **Đa dạng cấu hình giải mã** — beam5 / greedy / sampling t=1,2 / t=1,8 / epsilon ε=0,02, cộng nhánh ép sai token ngôn ngữ đích. Đầu ra vẫn là đầu ra thật của mô hình.

**Hệ dịch** — chỉ dùng hệ đã kiểm chứng chạy được:

| Hệ | Tỉ trọng | Vai trò |
|---|---|---|
| `nllb-200-distilled-600M` | 58% | Encoder-decoder nhỏ, hay ảo giác — nguồn mẫu dương chính; cùng dòng với HalOmi |
| `nllb-200-distilled-1.3B` | 20% | Bản lớn hơn — hệ đối chứng ít ảo giác |
| `Qwen2.5-7B-Instruct` | 12% | Dịch bằng LLM — tái hiện bối cảnh của P2 |
| ~~`envit5`~~, ~~`vinai-translate`~~ | 0% | **Vỡ với transformers ≥ 5.0** (tokenizer sentencepiece). Chạy được trên Colab/Kaggle (transformers 4.4x) nhưng không kiểm chứng được ở mọi môi trường → để tuỳ chọn, không đưa vào mặc định |

**Gán nhãn thủ công:** ~**500–600 cặp**, theo hướng dẫn 4 mức của HalOmi được Việt hoá (`docs/annotation-guideline-vi.md`):
- `No` (0 từ ảo giác) / `Small` (1–2 từ) / `Partial` (≥3 từ, chưa toàn bộ) / `Full` (gần như toàn bộ).
- Gán thêm **loại ảo giác**: `oscillatory` | `off-target` | `detached` (hoàn toàn không liên quan) | `fabricated-entity` (bịa tên riêng/số) | `omission`.
- Báo cáo **độ đồng thuận**: nếu nhóm ≥2 người → Cohen's kappa trên 100 cặp gán chéo; nếu 1 người → test–retest sau ≥3 ngày.

### 2.2. Tập B — Ảo giác tổng hợp có kiểm soát (synthetic)

Từ cặp song ngữ chuẩn, nhiễu loạn *phía đích* với nhãn **biết trước theo cấu tạo** (~2.500–3.000 cặp):

| Nhãn | Cách sinh |
|---|---|
| `No` | Bản dịch tham chiếu gốc + biến thể diễn giải (back-translation) → lớp âm khó |
| `Small` | Thay 1–2 từ nội dung bằng từ không liên quan / đổi số / đổi tên riêng |
| `Partial` | Thay 40–60% đoạn liên tục bằng nội dung khác |
| `Full-detached` | Thay toàn bộ bằng câu tiếng Việt ngẫu nhiên khác |
| `Full-oscillatory` | Lặp n-gram tới độ dài tương đương |
| `Off-target` | Thay bằng bản dịch sang ngôn ngữ khác (zh/th/id) |

Dùng để (a) huấn luyện mô hình có giám sát, (b) phân tích lỗi theo *loại*, (c) mở rộng quy mô. **Ghi rõ trong báo cáo** đây là tập chẩn đoán, tách biệt hoàn toàn với Tập A để tránh phóng đại kết quả.

### 2.3. Tập C — Probe đặc thù tiếng Việt (~150 cặp, thủ công)

Đây là phần thể hiện rõ nhất "khả năng xử lý tiếng Việt":

| Hiện tượng | Ví dụ phép thử |
|---|---|
| Thanh điệu / dấu | `má` ↔ `mà` ↔ `mã` ↔ `mả` ↔ `mạ`; mất dấu hoàn toàn |
| Đại từ thân tộc | `anh/chị/em/cô/chú` → he/she/uncle... (mơ hồ giới & vai vế) |
| Loại từ | `cái/con/chiếc/quyển/tấm` bị dịch sai hoặc thừa |
| Từ láy | `lung linh`, `xanh xao`, `lấp lánh` → bịa nghĩa |
| Thành ngữ | `nước đổ lá khoai`, `ăn cơm nhà vác tù và hàng tổng` |
| Số / ngày / tiền tệ | `1.000,5` vs `1,000.5`; `2/3` (ngày 2 tháng 3 vs phân số) |
| Tách từ | So sánh detector có/không dùng `underthesea` / `VnCoreNLP` word segmentation |

### 2.4. Chia tập & phát hành

- `dev` (~350 cặp): chọn prompt + hiệu chỉnh ngưỡng embedding (song ánh với vai trò DE↔EN trong P1).
- `test` (~1.200 cặp): chỉ đánh giá cuối.
- Phát hành công khai: **HuggingFace Datasets** + GitHub release (JSONL + datasheet + guideline).
- **Ngoài ra vẫn chạy toàn bộ pipeline trên HalOmi gốc** để chứng minh cài đặt lại paper trung thực (phần "cài đặt được paper" = 40%).

---

## 3. PHƯƠNG PHÁP & HỆ THỐNG (4,0 điểm)

### 3.1. Nhóm 1 — Bộ phát hiện bằng LLM (tái hiện P1)

- **3 biến thể prompt** đúng như Figure 10/11/12 của P1 + **CoT** (Figure 13/14) + **biến thể prompt tiếng Việt** (mới).
- Cấu hình: `temperature=0`, `max_new_tokens=5` (no-CoT) / `256` (CoT), tập nhãn ràng buộc.
- **Cải tiến:** thay vì chỉ đọc chuỗi sinh ra, lấy `logit` của token đầu tiên cho `"Hallucination"` vs `"No"` → điểm liên tục `p(hallu)` → tính được ROC-AUC + hiệu chỉnh ngưỡng, và **chỉ tốn 1 forward pass** (nhanh hơn ~10×, chạy được batch 32 trên T4).
**Danh sách mô hình đã chốt** (không dùng API trả phí):

| Nhóm | Mô hình | Ghi chú |
|---|---|---|
| Mở, tự chạy 4-bit trên T4 | `Qwen2.5-7B-Instruct` | Đa ngữ mạnh, không gated — **mô hình mỏ neo**, chạy đủ mọi ablation |
| | `SeaLLMs/SeaLLMs-v3-7B-Chat` | Chuyên Đông Nam Á, kỳ vọng mạnh nhất cho tiếng Việt |
| | `Viet-Mistral/Vistral-7B-Chat` | Chuyên tiếng Việt |
| | `meta-llama/Llama-3.1-8B-Instruct` | Gated — thay bằng `Qwen2.5-14B` (4-bit) nếu không xin được quyền |
| API free tier | `gemini-2.0-flash` | Cột "mô hình lớn"; có rate limit → chỉ chạy `test`, không chạy toàn bộ lưới ablation |
| | Groq `llama-3.3-70b-versatile` | Đối chứng gần nhất với `Llama3-70B` — mô hình tốt nhất của P1 |

> Hạn chế cần ghi vào *Limitations*: mô hình sau API free tier có thể thay đổi phiên bản theo thời gian → không tái lập tuyệt đối được. Vì vậy **mọi kết luận chính phải rút ra từ nhóm mã nguồn mở**, nhóm API chỉ đóng vai trò đối chiếu.

Chiến lược tiết kiệm tính toán: lưới đầy đủ (3 prompt × {no-CoT, CoT} × {EN, VI}) chỉ chạy trên `dev` (~350 cặp); `test` chỉ chạy **prompt tốt nhất của từng mô hình**, đúng quy trình của P1.

- **Ablation severity ranking** (4 lớp) với ROC-AUC, đúng như Appendix B của P1 — chỉ chạy trên mô hình mỏ neo nếu còn thời gian.

### 3.2. Nhóm 2 — Bộ phát hiện bằng embedding (tái hiện P1)

`cos(emb(src), emb(mt))`, ngưỡng tối ưu F1 trên `dev`:
- `sentence-transformers/LaBSE`
- `intfloat/multilingual-e5-large`
- `paraphrase-multilingual-mpnet-base-v2`
- **SONAR** (`facebook/SONAR` qua `sonar-space`) — nền của BLASER
- **Embedding chuyên tiếng Việt** (mới): `dangvantuan/vietnamese-embedding`, `bkai-foundation-models/vietnamese-bi-encoder`
- Thử nghiệm phụ: có/không tách từ tiếng Việt trước khi encode.

### 3.3. Nhóm 3 — Baseline (so sánh với mô hình cơ sở)

| Baseline | Vai trò |
|---|---|
| **BLASER-2.0-QE** (`facebook/blaser-2.0-qe`) | SOTA cũ trong P1 — baseline then chốt |
| **Top-4-gram repetition** (Raunak et al. 2021) | Rẻ, rất mạnh với oscillatory |
| **COMET-QE** `wmt22-cometkiwi-da` (gated) / `xcomet-lite` | QE hiện đại |
| **chrF / length-ratio** | Heuristic đơn giản |
| **Majority-class & random** | Sàn tham chiếu |

> ⚠️ **Rủi ro kỹ thuật:** `sonar-space` phụ thuộc `fairseq2`, hay xung đột phiên bản torch trên Colab. **Kế hoạch B:** ghim `torch` + cài `fairseq2` wheel tương ứng. **Kế hoạch C:** nếu vẫn hỏng, dùng LaBSE-cosine làm baseline chính và ghi rõ giới hạn này trong mục *Limitations*.

### 3.4. Nhóm 4 — Mô hình huấn luyện có giám sát

*(đóng góp riêng + đáp ứng yêu cầu nộp "mô hình đã huấn luyện")*

Cross-encoder nhị phân trên cặp `[src] </s> [mt]`:
- Backbone: `xlm-roberta-large` (đa ngữ) và `vinai/phobert-base-v2` (chuyên Việt, cho phía vi).
- Huấn luyện trên **Tập B (tổng hợp)** — không dùng nhãn người → giữ được tính khả mở của P2.
- Đánh giá trên **Tập A (tự nhiên, nhãn người)** → đo khả năng khái quát hoá synthetic → natural.
- Chi phí: ~10–15 phút trên 1×T4. Đây là mô hình sẽ đẩy lên HF Hub để nộp.

### 3.5. Nhóm 5 — Giảm thiểu ảo giác post-hoc (mở rộng theo P2) — **trong phạm vi**

Chỉ làm phần **post-hoc** của P2 (§2.2 + §5.1), **không** fine-tune CPO. Lý do: CPO cần A100 + hàng triệu câu; phần post-hoc đã đủ để trả lời "phát hiện xong thì làm gì tiếp" và cho ra một bảng kết quả trọn vẹn.

Quy trình: chạy detector tốt nhất trên đầu ra của hệ dịch → với mỗi câu bị gắn cờ, áp dụng một chiến lược chữa → đo lại:

| Chiến lược | Cài đặt | Đối chiếu với P2 |
|---|---|---|
| `Fallback` | Dịch lại bằng `NLLB-200-1.3B` / `vinai-translate-v2` | P2 đạt MR 96,5% |
| `Rerank-LaBSE` | Sinh n=16 ứng viên bằng epsilon sampling (ε=0,02) → chọn ứng viên cực đại cos-LaBSE với nguồn | **Chiến lược tốt nhất của P2**, MR 99,6% |
| `Rerank-COMET` | Như trên nhưng re-rank bằng COMET-QE | P2: LaBSE > COMET |
| `MBR-LaBSE` | Chọn ứng viên cực đại độ tương đồng trung bình với các ứng viên khác | P2: Re-rank > MBR |

Độ đo: **MR** (mitigation rate, công thức 10 của P2), **HR** trước/sau (công thức 9), và **COMET** để chứng minh chất lượng dịch chung không tụt — đây là điểm P2 nhấn mạnh và là cái bẫy dễ mắc.

> **Ngoài phạm vi (ghi vào *Future Work*):** LoRA-CPO fine-tune mô hình dịch trên preference set tự sinh.

### 3.6. Độ đo

- **Chính:** MCC. **Phụ:** F1/P/R của lớp `Hallucination`, ROC-AUC, Accuracy, AUPRC.
- Severity: ROC-AUC đa lớp (theo cách P1 hiệu chỉnh).
- Mitigation: MR, HR, COMET.
- **Khoảng tin cậy bootstrap 95%** (1.000 lần lặp) + kiểm định McNemar giữa hệ tốt nhất và baseline → tăng độ tin cậy học thuật.
- **Chi phí & độ trễ**: giây/câu, VRAM, USD/1000 câu — góc nhìn thực dụng mà P1 không có.

---

## 4. CẤU TRÚC NOTEBOOK (ánh xạ 1–1 với yêu cầu II.2)

Notebook chính `00_ViHalluMT_main.ipynb` chạy end-to-end trên Colab/Kaggle, tải mọi thứ từ HF Hub / URL công khai (không phụ thuộc đường dẫn cục bộ):

| # | Mục yêu cầu | Nội dung |
|---|---|---|
| 1 | Cài thư viện & cấu hình môi trường | `pip install`, phát hiện GPU, ghim phiên bản, cấu hình HF token (tuỳ chọn) |
| 2 | Tải / đọc dữ liệu | HalOmi (`wget` fbaipublicfiles) + ViHalluMT (HF Datasets) + FLORES |
| 3 | Khám phá & tiền xử lý | Phân bố nhãn/hướng dịch/độ dài, tỉ lệ mất cân bằng, chuẩn hoá Unicode NFC tiếng Việt, tách từ |
| 4 | Cài đặt / huấn luyện mô hình | LLM detector + embedding detector + huấn luyện cross-encoder |
| 5 | Đánh giá trên tập kiểm thử | Bảng MCC/F1/AUC cho mọi hệ, cả HalOmi lẫn ViHalluMT |
| 6 | So sánh với mô hình cơ sở | BLASER-2.0-QE, n-gram, COMET-QE, random/majority + kiểm định thống kê |
| 7 | Phân tích lỗi | Theo loại ảo giác, độ dài, hướng dịch, hiện tượng tiếng Việt (Tập C); ma trận nhầm lẫn; case study |
| 8 | Demo trên dữ liệu tiếng Việt mới | Nhập cặp (src, mt) bất kỳ → nhãn + điểm + giải thích; **Gradio UI**; ví dụ tin tức/pháp lý/y tế mới |

Notebook phụ (tái lập chi tiết, không bắt buộc chấm): `01_build_corpus.ipynb` · `02_annotation_tool.ipynb` · `03_run_detectors.ipynb` · `04_mitigation.ipynb`

Mọi bước nặng đều có **checkpoint kết quả trung gian** lưu trên HF Hub → giám khảo chạy lại nhanh mà không cần GPU.

---

## 5. CẤU TRÚC KHO MÃ NGUỒN

```
FinalProject/
├── README.md                      # hướng dẫn chạy (yêu cầu II.4)
├── requirements.txt
├── docs/
│   ├── DE-CUONG.md                # tài liệu này
│   ├── annotation-guideline-vi.md # hướng dẫn gán nhãn (Việt hoá từ HalOmi)
│   └── datasheet-vihallumt.md     # datasheet for datasets
├── data/
│   ├── vihallumt/{train,dev,test}.jsonl
│   ├── probe_vi.jsonl
│   └── raw/                       # .gitignore — có script tải
├── src/vihallumt/
│   ├── corpus/    build_natural.py, build_synthetic.py, probe_vi.py, sampling.py
│   ├── detectors/ llm.py, embed.py, blaser.py, ngram.py, crossencoder.py, base.py
│   ├── prompts/   binary_en.py, binary_vi.py, severity.py, cot.py
│   ├── mitigation/ fallback.py, rerank.py, cpo_lora.py
│   ├── eval/      metrics.py, bootstrap.py, plots.py, significance.py
│   └── cli.py
├── notebooks/     00_main.ipynb, 01..04
├── scripts/       download_halomi.sh, run_all.sh
├── results/       *.csv, figures/*.pdf
└── paper/         acl_latex.tex, custom.bib, acl.sty, figures/
```

---

## 6. BÁO CÁO ACL SHORT PAPER (4,0 điểm) — dàn ý 4–5 trang

| Mục | Nội dung | Ước lượng |
|---|---|---|
| **Abstract** | Bài toán, ViHalluMT, phát hiện chính (số liệu cụ thể) | 120 từ |
| **1. Introduction** | Ảo giác MT & rủi ro; tiếng Việt thiếu benchmark; 4 đóng góp gạch đầu dòng | 0,6 tr |
| **2. Related Work** | Phát hiện ảo giác MT (HalOmi, BLASER, ALTI+, Raunak n-gram); LLM-as-judge (G-Eval, Kocmi & Federmann); NLP tiếng Việt (PhoMT, PhoBERT, ViHallu DSC2025); giảm thiểu (P2) | 0,5 tr |
| **3. ViHalluMT Dataset** | Nguồn, hệ dịch, kích ảo giác, lấy mẫu phân tầng, quy trình gán nhãn, kappa, thống kê (bảng phân bố nhãn) | 0,9 tr |
| **4. Method** | Định nghĩa bài toán; LLM prompting + logit-scoring; embedding + ngưỡng; cross-encoder; baseline | 0,7 tr |
| **5. Experimental Setup** | Mô hình, siêu tham số, chia tập, độ đo, phần cứng, chi phí | 0,4 tr |
| **6. Results and Discussion** | **Bảng 1**: MCC/F1/AUC × hệ × hướng dịch (ViHalluMT). **Bảng 2**: tái lập trên HalOmi. **Hình 1**: MCC theo loại ảo giác. **Bảng 3**: mitigation MR/HR/COMET | 1,0 tr |
| **7. Error Analysis** | Theo loại ảo giác & hiện tượng tiếng Việt (Tập C); ma trận nhầm lẫn; ví dụ định tính; điểm mù của từng nhóm hệ | 0,6 tr |
| **8. Limitations & Ethics** | Quy mô ngữ liệu, số annotator, rủi ro nhiễm dữ liệu test, thiên lệch miền, tác động của báo động giả trong y tế/pháp lý | 0,3 tr |
| **9. Conclusion** | Tóm tắt + hướng phát triển | 0,15 tr |
| References / Appendix | không tính trang | — |

Dùng `acl-style-files` chính thức (Overleaf hoặc biên dịch local bằng `latexmk`).

---

## 7. KẾ HOẠCH THỰC HIỆN — 3 TUẦN (tuần 4 làm đệm)

Ký hiệu: 🤖 = Claude làm · 👤 = học viên làm · ⏳ = chạy nền trên GPU

### Tuần 1 — Nền móng, tái lập paper, sinh ngữ liệu

| Ngày | Việc | Ai | Đầu ra kiểm chứng được |
|---|---|---|---|
| ~~1~~ ✅ | Dựng repo, `requirements.txt`, tải & giải nén HalOmi, EDA nhãn | 🤖 | ✅ `data/raw/halomi_full.tsv`; **18/18 hướng dịch khớp Bảng 1–3 của P1** |
| ~~1~~ ✅ | Module `eval/metrics.py`: MCC, macro-average, bootstrap CI, McNemar | 🤖 | ✅ 66 test pass (33 test kiểm chứng số liệu paper) |
| ~~2~~ ✅ | Baseline phi-LLM + hiệu chỉnh ngưỡng trên dev DE↔EN | 🤖 | ✅ `results/halomi_baselines.csv` — 8 detector |
| ~~2–3~~ ✅ | ~~BLASER-2.0-QE: cài `sonar-space`+`fairseq2`~~ — **không cần**, HalOmi có sẵn điểm | 🤖 | ✅ **BLASER-QE macro-MCC 0.374 vs paper 0.38** (lệch 0.006); HRL 0.466 vs 0.46 |
| 3 | LLM detector (Qwen2.5-7B) + logit-scoring trên HalOmi | 🤖⏳ | `results/halomi_llm.csv`; MCC nằm trong khoảng của P1 |
| 4 | Sinh Tập A: dịch FLORES/OPUS/IWSLT bằng NLLB-600M + envit5 + Qwen; nhiễu nguồn & nhiễu giải mã | 🤖⏳ | ~8.000 cặp thô + điểm sơ bộ |
| 5 | Lấy mẫu phân tầng `uniform`/`biased`/`worst` → chọn 700 cặp để gán nhãn | 🤖 | `data/vihallumt/to_annotate.jsonl` |
| 5 | Sinh Tập B (tổng hợp, 6 loại nhiễu, ~2.500 cặp) + Tập C (probe tiếng Việt, 150 cặp) | 🤖 | `data/vihallumt/synthetic.jsonl`, `probe_vi.jsonl` |
| 6 | Viết `annotation-guideline-vi.md` (Việt hoá guideline HalOmi) + công cụ gán nhãn (notebook widget / Streamlit) | 🤖 | Công cụ chạy được, có phím tắt, tự lưu |
| 6 | **Gán thử 50 cặp** → chỉnh guideline theo ca khó gặp thực tế | 👤 | Guideline v2 |
| 7 | Đệm / bù việc trễ | — | — |

**Cổng kiểm tra cuối tuần 1:** đã tái lập được ít nhất 1 con số của P1 trên HalOmi, và đã có 700 cặp tiếng Việt chờ gán nhãn.

### Tuần 2 — Gán nhãn (👤) song song với thực nghiệm chính (🤖)

| Ngày | Việc | Ai | Đầu ra kiểm chứng được |
|---|---|---|---|
| 8–11 | **Gán nhãn 500–600 cặp Tập A** (~150 cặp/ngày ≈ 2,5 giờ/ngày) | 👤 | `data/vihallumt/annotated.jsonl` |
| 12 | Gán 150 cặp Tập C + gán lại 100 cặp của ngày 8 để đo kappa test–retest | 👤 | Báo cáo độ đồng thuận |
| 8–9 | Cài 3 prompt của P1 + CoT + **biến thể prompt tiếng Việt**; chạy lưới chọn prompt trên `dev` cho từng LLM mở | 🤖⏳ | `results/prompt_selection.csv` → Bảng phụ lục |
| 10 | Huấn luyện cross-encoder (XLM-R + PhoBERT) trên Tập B | 🤖⏳ | Checkpoint + `results/crossenc_dev.csv` |
| 11 | Chạy detector qua Gemini free tier + Groq llama-3.3-70b (có xử lý rate limit & cache) | 🤖⏳ | `results/api_llm.csv` |
| 12–13 | **Chạy toàn bộ hệ trên `test`** (sau khi nhãn xong) + bootstrap CI + McNemar | 🤖⏳ | **Bảng 1** — bảng kết quả chính |
| 14 | Chốt `ViHalluMT v0.1`: chia dev/test, viết datasheet, đẩy lên HF Datasets | 🤖 | Link dataset công khai |

**Cổng kiểm tra cuối tuần 2:** có Bảng 1 đầy đủ số liệu → đủ nguyên liệu để bắt đầu viết báo cáo.

### Tuần 3 — Mitigation, phân tích lỗi, notebook, báo cáo

| Ngày | Việc | Ai | Đầu ra kiểm chứng được |
|---|---|---|---|
| 15 | Post-hoc mitigation: fallback + epsilon-sampling n=16 + LaBSE/COMET re-rank | 🤖⏳ | **Bảng 3** (MR / HR / COMET) |
| 16 | Phân tích lỗi: cắt lát theo loại ảo giác, độ dài, hướng dịch, hiện tượng tiếng Việt (Tập C) | 🤖 | **Hình 1–2** + ma trận nhầm lẫn |
| 16 | Chọn 6–8 ví dụ định tính cho mục Error Analysis | 👤🤖 | Bảng case study |
| 17 | Gộp `00_main.ipynb` 8 mục + Gradio demo; chạy sạch từ runtime trống **trên Colab** | 🤖 | Notebook chạy hết không lỗi |
| 18 | Chạy lại notebook **trên Kaggle**; đẩy cross-encoder lên HF Hub | 🤖 | Link model + 2 ảnh chụp chạy thành công |
| 19–20 | Viết báo cáo ACL: LaTeX, bảng, hình, references | 🤖👤 | PDF 4–5 trang đúng `acl-style-files` |
| 21 | README, kiểm tra chéo yêu cầu II.4, nén `.zip` theo MSHV | 🤖 | File nộp hoàn chỉnh |

### Tuần 4 (đệm) — chỉ dùng nếu trễ

Thứ tự **hy sinh** khi thiếu thời gian (cắt từ trên xuống, không cắt lung tung):
1. Ablation severity ranking 4 lớp *(chỉ là phụ lục của P1)*
2. `MBR-LaBSE` và `Rerank-COMET` trong bảng mitigation *(giữ lại `Fallback` + `Rerank-LaBSE`)*
3. Cross-encoder PhoBERT *(giữ XLM-R)*
4. Nhóm LLM qua API *(kết luận chính vốn đã dựa vào nhóm mã nguồn mở)*

**Không được cắt:** tái lập trên HalOmi · 500 cặp gán nhãn tay · Bảng 1 · notebook 8 mục · báo cáo ACL.

**Tiêu chí "xong"** cho mỗi mục: có artefact kiểm chứng được (file kết quả / bảng số / notebook chạy sạch), không chỉ là "đã code xong".

---

## 8. RỦI RO & PHƯƠNG ÁN DỰ PHÒNG

| Rủi ro | Xác suất | Phương án |
|---|---|---|
| ~~`fairseq2`/`sonar-space` không cài được trên Colab → mất BLASER-QE~~ | ~~Cao~~ **ĐÃ LOẠI BỎ** | HalOmi phát hành kèm cột `score_blaser2_qe` tính sẵn (đủ 2.865 dòng, không NaN) → tái lập baseline SOTA mà không cần cài `fairseq2`. Chỉ còn cần SONAR nếu muốn tính điểm cho dữ liệu **tiếng Việt mới**; khi đó vẫn còn Kế hoạch C là LaBSE. |
| Tỉ lệ ảo giác tự nhiên quá thấp → tập dương quá nhỏ | Trung bình | Tăng nhiễu nguồn/giải mã; lấy mẫu `worst`; bổ sung bằng Tập B |
| Mô hình gated (Llama/Gemma/CometKiwi) không truy cập được | Trung bình | Thay bằng Qwen2.5 / SeaLLMs / Vistral (không gated) |
| Hết quota GPU Colab | Trung bình | Chuyển sang Kaggle (2×T4, 30h/tuần); lưu checkpoint trung gian lên HF |
| Gán nhãn 1 người → thiên lệch | Cao | Test–retest đo kappa nội tại; công bố guideline; nêu rõ ở Limitations |
| Nhiễm dữ liệu test (LLM đã thấy FLORES) | Trung bình | Bổ sung nguồn câu mới ngoài FLORES cho Tập C; nêu ở Limitations |

---

## 9. BẢN ĐỒ ĐỐI CHIẾU VỚI THANG ĐIỂM

| Yêu cầu | Điểm | Đáp ứng bằng |
|---|---|---|
| II.1 Ngữ liệu & xử lý tiếng Việt | 2,0 | ViHalluMT (A tự nhiên + gán nhãn tay, B tổng hợp, C probe tiếng Việt), guideline Việt hoá, kappa, datasheet, embedding/LLM chuyên Việt, phân tích lỗi ngôn ngữ học |
| II.2 Cài đặt, thực nghiệm, demo | 4,0 | Notebook 8 mục chạy trên Colab **và** Kaggle, tái lập HalOmi, ≥12 hệ thống, kiểm định thống kê, Gradio demo, mô hình huấn luyện trên HF |
| II.3 Báo cáo ACL | 4,0 | 4–5 trang đúng `acl-style-files`, đủ 11 mục, 3 bảng + 2 hình |
| II.4 Sản phẩm nộp | — | PDF, repo GitHub, notebook, README, link dataset HF, link model HF, nén `.zip` theo MSHV |

---

## 10. TÀI NGUYÊN ĐÃ XÁC MINH

| Tài nguyên | Trạng thái | Đường dẫn |
|---|---|---|
| HalOmi dataset | ✅ Tải được (19,9 MB) | `https://dl.fbaipublicfiles.com/nllb/halomi_release_v2.zip` |
| HalOmi code demo | ✅ | `github.com/facebookresearch/stopes/tree/main/demo/halomi` |
| BLASER-2.0-QE | ✅ Có trên HF (cần `sonar-space`) | `huggingface.co/facebook/blaser-2.0-qe` |
| FLORES-200 | ✅ Công khai | `openlanguagedata/flores_plus` (HF) |
| OPUS-100 en-vi | ✅ Công khai | `Helsinki-NLP/opus-100`, config `en-vi` |
| IWSLT'15 en-vi | ✅ Công khai | HF dataset `mt_eng_vietnamese` |
| PhoMT | ⚠️ Cần đăng ký form | `github.com/VinAIResearch/PhoMT` |
| ViHallu (DSC2025) | ℹ️ Ảo giác LLM, **không phải MT** — trích dẫn ở Related Work | `arxiv.org/abs/2601.04711` |
| ACL style files | ✅ | `github.com/acl-org/acl-style-files` |
