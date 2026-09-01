# Hướng dẫn gán nhãn ảo giác trong dịch máy — ViHalluMT

Phiên bản 1.0 · Việt hoá và mở rộng từ hướng dẫn của HalOmi (Dale et al., 2023)

---

## 1. Bạn đang làm gì

Với mỗi cặp **(câu nguồn, bản dịch máy)**, hãy trả lời một câu hỏi duy nhất:

> Bản dịch có chứa nội dung **không hề có** trong câu nguồn hay không?

Nếu có, nội dung đó là **ảo giác** (hallucination). Bạn gán mức độ nghiêm trọng
theo thang 4 mức và ghi loại ảo giác.

**Bạn KHÔNG đánh giá chất lượng dịch nói chung.** Một bản dịch có thể vụng về,
sai ngữ pháp, dùng từ không tự nhiên mà vẫn *không* có ảo giác. Ngược lại, một
bản dịch trôi chảy hoàn hảo vẫn có thể ảo giác nặng.

---

## 2. Ranh giới quan trọng nhất: ảo giác ≠ thiếu sót

Đây là chỗ dễ nhầm nhất, và nhầm ở đây sẽ làm hỏng cả bộ dữ liệu.

| | Định nghĩa | Có tính là ảo giác? |
|---|---|---|
| **Ảo giác** (hallucination) | Bản dịch **thêm** nội dung không có trong nguồn | **CÓ** |
| **Thiếu sót** (omission) | Bản dịch **bỏ mất** nội dung có trong nguồn | **KHÔNG** |

Ví dụ:

> **Nguồn:** The red car stopped in front of the old house.
> **Dịch:** Chiếc xe dừng lại trước ngôi nhà.
> → Mất "red" và "old", nhưng **không thêm gì**. Đây là *thiếu sót*, nhãn = `0 (Không)`.

> **Nguồn:** The car stopped in front of the house.
> **Dịch:** Chiếc xe màu đỏ dừng lại trước ngôi nhà cổ ở Hà Nội.
> → Thêm "màu đỏ", "cổ", "ở Hà Nội". Đây là *ảo giác*, nhãn = `2 (Một phần)`.

HalOmi gán nhãn riêng cho thiếu sót; đồ án này **chỉ làm ảo giác**, theo đúng
phạm vi của paper P1.

---

## 3. Thang 4 mức

Đếm số **từ bị ảo giác** trong bản dịch, rồi tra bảng:

| Nhãn | Tên | Tiêu chí | Ghi vào ô `severity` |
|---|---|---|---|
| Không | No hallucination | 0 từ ảo giác | `0` |
| Nhỏ | Small hallucination | 1–2 từ ảo giác | `1` |
| Một phần | Partial hallucination | ≥3 từ ảo giác, nhưng **không phải** toàn bộ câu | `2` |
| Toàn phần | Full hallucination | Gần như toàn bộ câu là ảo giác (có thể còn sót 1–2 từ đúng) | `3` |

Bốn nhãn **loại trừ lẫn nhau**: một câu ảo giác một phần thì không đồng thời là
ảo giác toàn phần.

### Cách đếm từ ảo giác

Với mỗi từ trong bản dịch, tự hỏi ba câu (theo CoT của HalOmi):

1. Từ nguồn nào tương ứng với từ đích này?
2. Từ nguồn đó có liên hệ ngữ nghĩa với từ đích này không?
3. Có cách nào giải thích hợp lý mối liên hệ đó không?

Nếu **"không"** cho cả ba → đếm là một từ ảo giác.

---

## 4. Loại ảo giác

Ghi vào ô `hallucination_type`. Nếu có nhiều loại, ghi loại **nổi trội nhất**.

| Mã | Tên | Dấu hiệu nhận biết |
|---|---|---|
| `oscillatory` | Dao động | Lặp đi lặp lại một cụm từ. Thường thấy nhất — P2 đo được 58–86% ảo giác thuộc loại này |
| `detached` | Tách rời | Bản dịch trôi chảy, đúng ngữ pháp, nhưng nội dung hoàn toàn không liên quan tới nguồn |
| `off_target` | Sai ngôn ngữ | Bản dịch ra ngôn ngữ khác (Trung, Thái, Anh khi lẽ ra phải là Việt...) |
| `fabricated` | Bịa chi tiết | Bịa tên riêng, con số, ngày tháng, địa danh không có trong nguồn |
| `mixed` | Hỗn hợp | Nhiều loại rõ rệt, không loại nào nổi trội |

Với `severity = 0`, để trống ô này.

---

## 5. Quy trình quyết định

```
Đọc câu nguồn.  Đọc bản dịch.
        │
        ├─ Bản dịch có ra đúng ngôn ngữ đích không?
        │     KHÔNG → severity = 3, type = off_target
        │
        ├─ Có lặp cụm từ bất thường không?
        │     CÓ → severity = 3, type = oscillatory
        │
        ├─ Nội dung có liên quan gì tới câu nguồn không?
        │     KHÔNG → severity = 3, type = detached
        │
        └─ Đếm số từ được THÊM VÀO mà nguồn không có:
              0 từ    → severity = 0
              1–2 từ  → severity = 1
              ≥3 từ   → severity = 2
```

---

## 6. Ca khó — quyết định thống nhất

Những quy ước dưới đây phải **áp dụng nhất quán**. Gặp ca không nằm trong danh
sách thì ghi vào ô `annotator_note` và bổ sung vào tài liệu này.

### 6.1. Sai nghĩa có tính là ảo giác không?

**Có, nếu từ được chọn không có liên hệ ngữ nghĩa với nguồn.**

> **Nguồn:** He sold the house. → **Dịch:** Anh ấy đã *bàn* căn nhà.
> `bán` → `bàn` là hai từ khác hẳn nghĩa → 1 từ ảo giác → `severity = 1`

Nhưng chọn từ *gần nghĩa* thì **không** tính:

> **Nguồn:** He purchased a car. → **Dịch:** Anh ấy đã *mua* một chiếc xe.
> `purchase` / `mua` khác sắc thái trang trọng nhưng cùng nghĩa → `severity = 0`

### 6.2. Sai loại từ

Sai loại từ (`một con sách` thay vì `một quyển sách`) là **lỗi ngữ pháp**, không
phải ảo giác — không có nội dung mới nào được thêm vào.
→ `severity = 0`, ghi chú `sai loại từ`.

### 6.3. Sai đại từ thân tộc

Nếu bản dịch chọn sai vai vế/giới tính mà nguồn **có nói rõ**, đó là ảo giác
(bịa thông tin quan hệ):

> **Nguồn:** My older brother works at a bank. → **Dịch:** *Em gái* tôi làm việc ở ngân hàng.
> → `severity = 1`, type = `fabricated`

Nếu nguồn **không nói rõ** (chỉ có "he"/"she"), việc chọn `anh`/`em` là bắt buộc
trong tiếng Việt và không thể tránh → **không** tính là ảo giác.

### 6.4. Số và ngày tháng sai

Luôn tính là ảo giác, type = `fabricated`:

> **Nguồn:** 1,250.75 dollars → **Dịch:** 1.250,75 đô la → **đúng** (`severity = 0`)
> **Nguồn:** 1,250.75 dollars → **Dịch:** 1,250.75 đô la → **sai** (`severity = 1`)

Tiếng Việt dùng **dấu chấm cho hàng nghìn, dấu phẩy cho thập phân** — ngược với
tiếng Anh.

### 6.5. Tên riêng để nguyên

Giữ nguyên tên riêng không phải ảo giác. Đổi tên riêng thành tên khác thì có.

### 6.6. Bản dịch trống hoặc chỉ có dấu câu

→ `severity = 0` (đây là thiếu sót toàn phần, không phải ảo giác). Ghi chú
`bản dịch trống`.

### 6.7. Bản dịch lặp lại nguyên câu nguồn

Chép nguyên câu nguồn sang mà không dịch → đây là **không dịch**, không phải ảo
giác. `severity = 0`, ghi chú `không dịch`.

### 6.8. Thêm từ do đặc thù ngữ pháp tiếng Việt

Tiếng Việt bắt buộc có loại từ, tiểu từ tình thái, đại từ mà tiếng Anh không có.
Những từ này **không** tính là ảo giác:

> **Nguồn:** Book is on table. → **Dịch:** *Quyển* sách ở trên *cái* bàn.
> `quyển`, `cái` là loại từ bắt buộc → `severity = 0`

---

## 7. Chất lượng gán nhãn

### 7.1. Nhịp độ

Khoảng **150 cặp mỗi buổi (2–2,5 giờ)**. Gán quá 200 cặp một buổi thì chất
lượng tụt rõ rệt vì mỏi mắt và trôi tiêu chuẩn.

### 7.2. Đo độ đồng thuận

- **Nếu nhóm ≥2 người:** 100 cặp đầu tiên do hai người gán độc lập → tính
  Cohen's kappa. Thảo luận các ca lệch, cập nhật tài liệu này, rồi mới gán tiếp.
- **Nếu làm một mình:** gán lại 100 cặp của buổi đầu tiên sau **ít nhất 3 ngày**,
  không nhìn nhãn cũ → tính kappa test–retest.

Kappa < 0,6 nghĩa là tiêu chuẩn chưa rõ ràng: phải sửa tài liệu này rồi gán lại,
chứ không được đi tiếp.

### 7.3. Không nhìn điểm máy

File `to_annotate.jsonl` có sẵn cột `score_labse`, `score_ngram`, `selection`.
**Không được nhìn các cột này khi gán nhãn.** Nhìn vào sẽ khiến nhãn người bị
kéo theo dự đoán của máy, và bộ dữ liệu mất giá trị làm chuẩn đối chiếu. Công cụ
gán nhãn đã ẩn sẵn các cột đó.

---

## 8. Mẫu ghi nhãn

Mỗi dòng trong `to_annotate.jsonl` cần điền ba ô:

| Ô | Giá trị | Bắt buộc |
|---|---|---|
| `severity` | `0` / `1` / `2` / `3` | Có |
| `hallucination_type` | `oscillatory` / `detached` / `off_target` / `fabricated` / `mixed` | Chỉ khi `severity > 0` |
| `annotator_note` | Ghi chú tự do cho ca khó | Không |

---

## 9. Nhật ký thay đổi

| Phiên bản | Ngày | Thay đổi |
|---|---|---|
| 1.0 | 2026-09-01 | Bản đầu, Việt hoá từ HalOmi + bổ sung mục 6.2–6.8 cho tiếng Việt |

> Mỗi khi gặp ca khó chưa có trong mục 6, hãy bổ sung vào đây **kèm quyết định
> đã chọn** rồi tăng số phiên bản. Tài liệu này phải nộp kèm bộ dữ liệu — nó là
> thứ khiến nhãn của bạn tái lập được, và là một phần điểm của mục ngữ liệu.
