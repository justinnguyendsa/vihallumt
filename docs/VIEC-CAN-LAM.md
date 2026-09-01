# Việc bạn cần làm

Danh sách này chỉ liệt kê việc **bạn** phải tự làm — những thứ tôi không làm
thay được vì cần tài khoản của bạn, GPU, hoặc phán đoán của con người.

Tổng thời gian: **khoảng 13–15 giờ**, trong đó 10–12 giờ là gán nhãn.

---

## Phần 0 — Ngay bây giờ (khoảng 30 phút)

### ☑ 0.1. Tạo kho mã nguồn trên GitHub — **ĐÃ XONG**

Kho: **https://github.com/justinnguyendsa/vihallumt** (hiện đang **public** để
Kaggle clone được). Nhớ chuyển lại private sau khi chạy Kaggle xong:

```bash
gh repo edit justinnguyendsa/vihallumt --visibility private --accept-visibility-change-consequences
```

<details><summary>Cách đã làm (để tham khảo)</summary>

Đây vừa là sản phẩm bắt buộc phải nộp (yêu cầu II.4), vừa là cách notebook
Kaggle lấy được mã nguồn.

```bash
cd "D:/My learning/NLP/Projects/FinalProject"
git init -b main
git add .
git commit -m "ViHalluMT: tai lap P1 tren HalOmi + duong ong sinh ngu lieu tieng Viet"
gh repo create vihallumt --private --source=. --push
```

Để **private** lúc này cho an toàn. Khi nộp bài thì chuyển sang public, hoặc
thêm giảng viên làm collaborator:

```bash
gh repo edit --visibility public --accept-visibility-change-consequences
```

> `.gitignore` đã loại sẵn `papers/`, `data/raw/`, và các tệp `.docx`/`.xlsx`
> của môn học nên kho sẽ gọn.

</details>

### ☐ 0.2. Mở khoá FLORES-200 trên HuggingFace  ⏱ 3 phút

Nên làm, không bắt buộc. Có FLORES thì kết quả tiếng Việt so sánh trực tiếp
được với 18 hướng dịch của HalOmi, vì HalOmi cũng lấy câu nguồn từ FLORES.

1. Mở https://huggingface.co/datasets/openlanguagedata/flores_plus
2. Bấm nút đồng ý điều khoản (miễn phí, duyệt ngay)
3. Vào https://huggingface.co/settings/tokens → **New token** → quyền `read`
4. Chép token ra, giữ lại cho bước 1.3

Bỏ qua bước này cũng được — đường ống vẫn chạy đủ với OPUS-100 + IWSLT'15, chỉ
mất tính so sánh trực tiếp với HalOmi (phải ghi vào mục *Limitations*).

### ☐ 0.3. Tạo tài khoản Kaggle và xác minh số điện thoại  ⏱ 5 phút

Kaggle chỉ cho dùng GPU sau khi xác minh số điện thoại. Hạn mức 30 giờ GPU mỗi
tuần, tốt hơn Colab bản miễn phí.

https://www.kaggle.com → *Settings* → *Phone Verification*

### ☐ 0.4. Cung cấp cho tôi bốn thông tin  ⏱ 2 phút

Nhắn lại cho tôi:

| Thông tin | Dùng để làm gì |
|---|---|
| **MSHV của bạn** (và của các thành viên nhóm nếu có) | Đặt tên tệp `.zip` nộp bài |
| **Số thành viên nhóm** | Nếu ≥2 người thì chia được việc gán nhãn, và đo được Cohen's kappa liên-người thay vì test–retest — đáng tin hơn hẳn |
| **Ngôn ngữ báo cáo**: Anh hay Việt | Tôi đang mặc định tiếng Anh |
| **URL kho GitHub** vừa tạo | Để tôi ghi vào README và notebook |

---

## Phần 1 — Sinh ngữ liệu trên Kaggle (khoảng 1 giờ, phần lớn là chờ)

### ☐ 1.1. Tải notebook lên Kaggle  ⏱ 3 phút

1. https://www.kaggle.com/code → **New Notebook** → *File → Import Notebook*
2. Chọn tệp `notebooks/01_build_corpus_kaggle.ipynb`
3. Panel bên phải → *Settings* → **Accelerator: GPU T4 x2**
4. *Settings* → **Internet: On** (bắt buộc, để tải mô hình từ HuggingFace)

### ☑ 1.2. Sửa URL kho mã nguồn — **ĐÃ XONG**

`REPO_URL` trong notebook đã điền sẵn kho thật, và kho đang public nên Kaggle
clone được. Đã kiểm chứng bằng phép clone ẩn danh với config sạch: thành công,
45 tệp.

### ☐ 1.3. (Nếu làm bước 0.2) Thêm HF token  ⏱ 2 phút

*Add-ons → Secrets → Add a new secret*
- Label: `HF_TOKEN`
- Value: token bạn chép ở bước 0.2

Rồi trong notebook đặt `USE_FLORES = True`.

### ☐ 1.4. Chạy thử trước  ⏱ 5 phút

Trong ô "Sinh ngữ liệu", đặt:

```python
SMOKE = True
```

Rồi *Run All*. Chỉ vài phút. Mục đích là phát hiện lỗi cấu hình **trước khi**
tiêu 45 phút GPU. Xem ô kiểm tra sức khoẻ dữ liệu ở cuối — phải in
`VAN DE PHAT HIEN: khong co`.

### ☐ 1.5. Chạy thật  ⏱ 30–50 phút (để máy chạy, đi làm việc khác)

Đặt lại `SMOKE = False`, *Run All*. Ước lượng thời gian trên T4:

| Bước | Thời gian |
|---|---|
| Tải mô hình (NLLB-600M, NLLB-1.3B, Qwen-7B, LaBSE) | 10–15 phút |
| Dịch 6.000 câu × 2 hướng | 20–30 phút |
| Chấm điểm + lấy mẫu | 3–5 phút |

Nếu báo hết bộ nhớ GPU: giảm `BATCH_SIZE` từ 32 xuống 16 hoặc 8.

### ☐ 1.6. Tải kết quả về máy  ⏱ 2 phút

Tab **Output** của notebook → tải hai tệp:
- `to_annotate.jsonl`
- `corpus_a_stats.csv`

Đặt vào đúng chỗ trên máy:

```
D:/My learning/NLP/Projects/FinalProject/data/vihallumt/to_annotate.jsonl
D:/My learning/NLP/Projects/FinalProject/results/corpus_a_stats.csv
```

**Báo tôi biết khi xong bước này** — tôi cần xem thống kê để kiểm tra tỉ lệ ảo
giác có đủ dùng không. Nếu tầng `worst` mà tỉ lệ ảo giác quá thấp thì phải chỉnh
kế hoạch sinh dữ liệu rồi chạy lại, tốt hơn là phát hiện ngay bây giờ chứ không
phải sau khi bạn đã gán nhãn 600 cặp.

---

## Phần 2 — Gán nhãn (10–12 giờ, chia ra 5 ngày)

Đây là phần quyết định 2,0 điểm ngữ liệu tiếng Việt, và là phần dài nhất.

### ☐ 2.1. Đọc hướng dẫn gán nhãn  ⏱ 20 phút

`docs/annotation-guideline-vi.md`

Đọc kỹ **mục 2** (ảo giác ≠ thiếu sót — chỗ dễ nhầm nhất) và **mục 6** (8 ca
khó đã có quyết định thống nhất).

### ☐ 2.2. Gán thử 50 cặp rồi chỉnh hướng dẫn  ⏱ 1 giờ

```bash
cd "D:/My learning/NLP/Projects/FinalProject"
python scripts/annotate.py --n 50
```

Vừa gán vừa ghi lại những ca mà bạn **do dự quá 15 giây**. Gán xong 50 cặp thì
mở lại `docs/annotation-guideline-vi.md`, thêm các ca đó vào mục 6 **kèm quyết
định bạn đã chọn**, rồi tăng số phiên bản.

> Bước này quan trọng hơn vẻ ngoài của nó. Tiêu chuẩn không rõ ràng thì kappa
> sẽ thấp, và kappa thấp thì cả bộ dữ liệu mất giá trị. Chỉnh hướng dẫn bây giờ
> rẻ hơn nhiều so với gán lại 600 cặp sau.

### ☐ 2.3. Gán 500–600 cặp  ⏱ 8–10 giờ, chia 4–5 buổi

```bash
python scripts/annotate.py
```

- **Khoảng 150 cặp mỗi buổi**, đừng quá 200. Quá ngưỡng đó chất lượng tụt rõ
  rệt vì mỏi mắt và trôi tiêu chuẩn.
- Công cụ **tự lưu sau mỗi nhãn** — đóng giữa chừng lúc nào cũng được, chạy lại
  là gán tiếp từ chỗ dừng.
- Gõ `q` để lưu và thoát.
- Gặp ca thật sự không quyết định được thì gõ `s` để bỏ qua, còn hơn đoán bừa.

Xem tiến độ và thống kê bất cứ lúc nào:

```bash
python scripts/annotate.py --review
```

### ☐ 2.4. Đo độ đồng thuận  ⏱ 1,5 giờ

**Nếu nhóm có ≥2 người:** hai người cùng gán độc lập 100 cặp đầu tiên. Đây là
cách đo tốt hơn.

**Nếu bạn làm một mình:** đợi **ít nhất 3 ngày** sau buổi gán đầu tiên, rồi gán
lại 100 cặp đó mà không nhìn nhãn cũ:

```bash
python scripts/annotate.py --retest --n 100
```

Kappa < 0,6 nghĩa là tiêu chuẩn chưa đủ rõ — phải sửa hướng dẫn rồi gán lại,
không được đi tiếp.

### ☐ 2.5. Báo tôi biết khi gán xong

Tôi sẽ chạy toàn bộ detector trên dữ liệu của bạn và dựng bảng kết quả chính.

---

## Phần 3 — Việc tôi làm song song (bạn không cần chờ)

Trong lúc bạn gán nhãn, tôi làm:

- Chạy LLM detector trên HalOmi, quét lưới chọn prompt (tái hiện Bảng 6 của P1)
- Huấn luyện cross-encoder trên Tập B tổng hợp
- Cài đặt phần giảm thiểu ảo giác post-hoc theo paper 2
- Dựng khung báo cáo LaTeX theo định dạng ACL

---

## Tóm tắt lịch

| Ngày | Bạn làm | Tôi làm |
|---|---|---|
| Hôm nay | Phần 0 + Phần 1 (khoảng 1,5 giờ) | Chờ dữ liệu; chuẩn bị phần LLM detector |
| Ngày 1–4 | Gán nhãn, 150 cặp/ngày | Chạy detector trên HalOmi, huấn luyện cross-encoder |
| Ngày 5 | Gán nốt + đo kappa | Cài đặt mitigation |
| Ngày 6–7 | Xem lại kết quả, chọn ví dụ định tính cho phần phân tích lỗi | Dựng bảng kết quả, viết báo cáo |

---

## Nếu gặp trục trặc

| Triệu chứng | Cách xử lý |
|---|---|
| Kaggle báo hết bộ nhớ GPU | Giảm `BATCH_SIZE` xuống 16 rồi 8 |
| `git clone` trên Kaggle bị treo | Kho đang private — dùng Cách B (Kaggle Dataset) trong notebook |
| Notebook báo hỏng một hệ dịch | Bình thường, script tự bỏ và chia lại tỉ trọng. Chụp lại màn hình đó để tôi ghi vào *Limitations* |
| Tỉ lệ ảo giác trong `to_annotate` quá thấp (<15%) | Dừng, báo tôi. Phải chỉnh kế hoạch sinh dữ liệu chứ đừng gán nhãn tiếp |
| Công cụ gán nhãn hiển thị lỗi font tiếng Việt | Chạy `chcp 65001` trước, hoặc dùng Windows Terminal thay cho cmd.exe |
