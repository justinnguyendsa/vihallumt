"""Nhiễu loạn câu nguồn để *kích* hệ dịch sinh ra ảo giác tự nhiên.

Tại sao cần bước này
--------------------
Hệ dịch tốt hiếm khi ảo giác trên câu nguồn sạch: tỉ lệ ảo giác tự nhiên của
`ALMA-7B-R` trong P2 chỉ 0,127%. Gán nhãn tay 600 cặp lấy ngẫu nhiên sẽ chỉ thu
được một hai mẫu dương — vô dụng. HalOmi giải quyết bằng cách nhiễu loạn câu
nguồn để đẩy hệ dịch vào vùng hoạt động bất thường, rồi mới gán nhãn thủ công
trên **đầu ra thực tế** của hệ.

Điểm mấu chốt: ta nhiễu loạn **câu nguồn**, còn nhãn thì gán trên **bản dịch
mà hệ thật sự sinh ra**. Ảo giác thu được vì thế vẫn là ảo giác *tự nhiên* của
mô hình, chỉ là ta làm nó xảy ra thường xuyên hơn. Khác hẳn với việc bịa thẳng
bản dịch hỏng (xem `synthetic.py` — dùng cho mục đích khác).

Chọn phép nhiễu loạn nào
------------------------
P2 §6.4 kiểm định chi-bình-phương và tìm ra ba đặc trưng của câu nguồn làm tăng
đáng kể (p < 0,05) khả năng ảo giác: **dấu ngoặc kép**, **URL**, và **cụm viết
hoa toàn bộ**. Các phép nhiễu ở đây tái tạo đúng ba yếu tố đó, cộng thêm lỗi
chính tả và cắt cụt theo cách làm của HalOmi.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable

# Ký tự dùng để chèn lỗi chính tả cho văn bản Latinh
_LATIN = "abcdefghijklmnopqrstuvwxyz"

# URL giả để chèn — dùng miền example.* nên không trỏ tới đâu thật
_URLS = (
    "https://example.com/news/2024/03/report",
    "http://www.example.org/index.html",
    "https://example.net/a/b?id=7391",
)


@dataclass(frozen=True)
class Perturbation:
    """Một câu nguồn đã bị nhiễu loạn, kèm nhãn ghi rõ đã làm gì."""

    text: str
    kind: str

    @property
    def is_clean(self) -> bool:
        return self.kind == "clean"


# --------------------------------------------------------------------------
# Từng phép nhiễu loạn
# --------------------------------------------------------------------------

def _corrupt_word(word: str, rng: random.Random) -> str:
    """Làm hỏng một từ, **bảo đảm** kết quả khác từ ban đầu.

    Bảo đảm này quan trọng: một phép nhiễu âm thầm không làm gì sẽ tạo ra
    những dòng dữ liệu gắn nhãn "đã nhiễu loạn" nhưng thực chất vẫn sạch, làm
    sai lệch mọi phân tích theo loại nhiễu về sau.
    """
    ops = ["delete", "replace"]  # cả hai đều chắc chắn thay đổi từ
    swap_positions = [k for k in range(len(word) - 1) if word[k] != word[k + 1]]
    if swap_positions:
        ops.append("swap")

    op = rng.choice(ops)
    if op == "swap":
        j = rng.choice(swap_positions)
        return word[:j] + word[j + 1] + word[j] + word[j + 2:]

    j = rng.randrange(len(word))
    if op == "delete":
        return word[:j] + word[j + 1:]
    # replace: bắt buộc chọn ký tự KHÁC ký tự hiện tại
    alt = rng.choice([c for c in _LATIN if c != word[j].lower()])
    return word[:j] + alt + word[j + 1:]


def misspell(text: str, rng: random.Random, rate: float = 0.15) -> str:
    """Gây lỗi chính tả ngẫu nhiên: đổi chỗ, thay, hoặc xoá ký tự.

    Áp dụng ở mức ký tự nên hoạt động với mọi hệ chữ viết, kể cả tiếng Việt có
    dấu (khi đó việc thay ký tự thường làm mất dấu — chính là loại lỗi ta muốn).
    Luôn làm hỏng ít nhất một từ.
    """
    words = text.split()
    if not words:
        return text

    # Chỉ nhắm vào từ đủ dài để làm hỏng mà vẫn còn nhận ra được là từ
    eligible = [i for i, w in enumerate(words) if len(w) >= 3]
    if not eligible:
        # Không từ nào đủ dài: nhân đôi ký tự cuối để vẫn tạo được lỗi
        return text + text[-1]

    n_hit = max(1, int(len(eligible) * rate))
    for i in rng.sample(eligible, min(n_hit, len(eligible))):
        words[i] = _corrupt_word(words[i], rng)
    return " ".join(words)


def all_caps(text: str, rng: random.Random) -> str:
    """Viết hoa toàn bộ một cụm liên tiếp — P2 chứng minh có liên hệ với ảo giác."""
    words = text.split()
    if len(words) < 2:
        return text.upper()
    span = rng.randint(2, min(5, len(words)))
    start = rng.randrange(0, len(words) - span + 1)
    for i in range(start, start + span):
        words[i] = words[i].upper()
    return " ".join(words)


def insert_quotes(text: str, rng: random.Random) -> str:
    """Bọc một cụm trong dấu ngoặc kép — đặc trưng gây ảo giác theo P2."""
    words = text.split()
    if len(words) < 3:
        return f'"{text}"'
    span = rng.randint(2, min(6, len(words)))
    start = rng.randrange(0, len(words) - span + 1)
    words[start] = '"' + words[start]
    words[start + span - 1] = words[start + span - 1] + '"'
    return " ".join(words)


def insert_url(text: str, rng: random.Random) -> str:
    """Chèn URL vào đầu hoặc cuối câu — đặc trưng gây ảo giác theo P2."""
    url = rng.choice(_URLS)
    return f"{url} {text}" if rng.random() < 0.5 else f"{text} {url}"


def truncate(text: str, rng: random.Random) -> str:
    """Cắt cụt câu, để lại 30–70% số từ.

    Câu cụt buộc hệ dịch phải "đoán" phần còn thiếu — nguồn ảo giác điển hình.
    """
    words = text.split()
    if len(words) < 4:
        return text
    keep = max(2, int(len(words) * rng.uniform(0.3, 0.7)))
    return " ".join(words[:keep])


def append_noise_tokens(text: str, rng: random.Random, n: int = 3) -> str:
    """Nối thêm các token vô nghĩa vào cuối câu."""
    junk = ["".join(rng.choices(_LATIN, k=rng.randint(3, 8))) for _ in range(n)]
    return text + " " + " ".join(junk)


def strip_diacritics(text: str) -> str:
    """Bỏ toàn bộ dấu tiếng Việt — phép nhiễu **đặc thù tiếng Việt**.

    Người Việt vẫn viết không dấu khi nhắn tin, nên đây là nhiễu thực tế chứ
    không nhân tạo. Nó cũng gây mơ hồ nghĩa nghiêm trọng: "ma" có thể là
    má / mà / mã / mả / mạ, buộc hệ dịch phải chọn bừa.
    """
    import unicodedata

    decomposed = unicodedata.normalize("NFD", text)
    without = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    # đ/Đ không phải dấu tổ hợp nên phải xử lý riêng
    return unicodedata.normalize("NFC", without).replace("đ", "d").replace("Đ", "D")


# --------------------------------------------------------------------------
# Bộ điều phối
# --------------------------------------------------------------------------

#: Các phép nhiễu dùng cho mọi ngôn ngữ.
GENERIC_PERTURBATIONS: dict[str, Callable[[str, random.Random], str]] = {
    "misspell": misspell,
    "all_caps": all_caps,
    "quotes": insert_quotes,
    "url": insert_url,
    "truncate": truncate,
    "noise_tokens": append_noise_tokens,
}

#: Phép nhiễu chỉ có nghĩa với câu nguồn tiếng Việt.
VIETNAMESE_PERTURBATIONS: dict[str, Callable[[str, random.Random], str]] = {
    "no_diacritics": lambda t, _rng: strip_diacritics(t),
}


def available_perturbations(lang: str) -> dict[str, Callable[[str, random.Random], str]]:
    """Các phép nhiễu áp dụng được cho một ngôn ngữ nguồn."""
    out = dict(GENERIC_PERTURBATIONS)
    if lang.upper() == "VI":
        out.update(VIETNAMESE_PERTURBATIONS)
    return out


def perturb(
    text: str,
    lang: str = "EN",
    kind: str | None = None,
    rng: random.Random | None = None,
) -> Perturbation:
    """Áp một phép nhiễu lên câu nguồn.

    Args:
        text: câu nguồn gốc.
        lang: viết tắt ngôn ngữ nguồn, quyết định phép nhiễu nào dùng được.
        kind: tên phép nhiễu, hoặc `"clean"` để giữ nguyên, hoặc `None` để
            chọn ngẫu nhiên.
        rng: bộ sinh số ngẫu nhiên, truyền vào để tái lập được.

    Raises:
        KeyError: nếu `kind` không áp dụng được cho ngôn ngữ này.
    """
    rng = rng or random.Random()
    options = available_perturbations(lang)

    if kind == "clean":
        return Perturbation(text, "clean")
    if kind is None:
        kind = rng.choice(sorted(options))
    if kind not in options:
        raise KeyError(f"Phep nhieu {kind!r} khong dung duoc cho ngon ngu {lang!r}. "
                       f"Cac lua chon: {sorted(options)}")

    return Perturbation(options[kind](text, rng), kind)


def perturbation_plan(
    n_sentences: int,
    lang: str = "EN",
    clean_ratio: float = 0.4,
    seed: int = 42,
) -> list[str]:
    """Sinh danh sách phép nhiễu cho `n_sentences` câu, cân bằng giữa các loại.

    Giữ lại `clean_ratio` câu không nhiễu để tập dữ liệu vẫn có bản dịch của
    câu nguồn bình thường — nếu chỉ toàn câu nguồn hỏng thì tỉ lệ ảo giác sẽ
    cao giả tạo và bộ dữ liệu mất tính đại diện.
    """
    if not 0.0 <= clean_ratio <= 1.0:
        raise ValueError("clean_ratio phai nam trong [0, 1]")

    rng = random.Random(seed)
    kinds = sorted(available_perturbations(lang))

    n_clean = int(round(n_sentences * clean_ratio))
    plan = ["clean"] * n_clean

    remaining = n_sentences - n_clean
    for i in range(remaining):
        plan.append(kinds[i % len(kinds)])

    rng.shuffle(plan)
    return plan
