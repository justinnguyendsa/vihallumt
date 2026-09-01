"""Sinh ảo giác tổng hợp có kiểm soát — Tập B của ViHalluMT.

Khác với `perturb.py` (nhiễu câu *nguồn* rồi để hệ dịch tự ảo giác), module này
làm hỏng trực tiếp câu *đích* theo luật, nên **nhãn biết trước theo cấu tạo**.

Dùng để làm gì
--------------
1. Huấn luyện cross-encoder có giám sát mà không tốn công gán nhãn tay.
2. Phân tích lỗi theo *loại* ảo giác — mỗi mẫu có nhãn loại chính xác, điều mà
   dữ liệu tự nhiên không có được nếu không gán tay tỉ mỉ.
3. Mở rộng quy mô tập dữ liệu.

KHÔNG dùng để làm gì
--------------------
**Không được dùng làm kết quả chính của báo cáo.** Ảo giác chèn bằng luật dễ
phát hiện hơn hẳn ảo giác thật, nên mọi con số trên tập này đều lạc quan quá
mức. Bảng kết quả chính phải chạy trên Tập A (dữ liệu tự nhiên, nhãn người).
Ranh giới này phải được nêu rõ trong báo cáo.

Phân loại ảo giác bám theo thang 4 mức của HalOmi (No/Small/Partial/Full) và
các loại quan sát được trong P2 §6.4 (dao động, tách rời, sai ngôn ngữ đích).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Sequence

from vihallumt.data import SEVERITY_ORDER

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

#: Bản dịch sang ngôn ngữ khác, dùng cho ảo giác "sai ngôn ngữ đích".
#: Đây là các câu tĩnh có thật, không sinh bằng máy, để tránh phụ thuộc mạng.
OFF_TARGET_POOL: dict[str, tuple[str, ...]] = {
    "zh": (
        "这是一个完全不相关的句子。",
        "会议将于明天上午九点开始。",
        "他昨天买了一本关于历史的书。",
    ),
    "th": (
        "นี่เป็นประโยคที่ไม่เกี่ยวข้องกันเลย",
        "การประชุมจะเริ่มในเวลาเก้าโมงเช้า",
    ),
    "id": (
        "Ini adalah kalimat yang sama sekali tidak berhubungan.",
        "Rapat akan dimulai besok pagi pukul sembilan.",
    ),
}


@dataclass(frozen=True)
class SyntheticExample:
    """Một cặp câu đã bị làm hỏng có kiểm soát."""

    src_text: str
    mt_text: str
    severity: int          # 0..3, trùng thang của HalOmi
    hallucination_type: str
    reference: str         # bản dịch chuẩn, giữ lại để đối chiếu

    @property
    def label(self) -> int:
        return int(self.severity > 0)

    @property
    def severity_name(self) -> str:
        return SEVERITY_ORDER[self.severity]


def tokenize(text: str) -> list[str]:
    return text.split()


# --------------------------------------------------------------------------
# Từng kiểu làm hỏng
# --------------------------------------------------------------------------

def corrupt_small(text: str, donor_words: Sequence[str], rng: random.Random) -> str:
    """Thay 1–2 từ nội dung bằng từ không liên quan -> mức `Small`.

    HalOmi định nghĩa `Small hallucination` là "bản dịch chứa 1–2 từ bị ảo
    giác". Đây là mức khó nhất với mọi detector: câu vẫn gần như đúng hoàn
    toàn nên độ tương đồng embedding hầu như không đổi.
    """
    words = tokenize(text)
    if len(words) < 3 or not donor_words:
        return text

    n = rng.randint(1, 2)
    # Ưu tiên từ dài — thường là từ nội dung, không phải hư từ
    candidates = sorted(range(len(words)), key=lambda i: -len(words[i]))[: max(3, len(words) // 2)]
    for i in rng.sample(candidates, min(n, len(candidates))):
        words[i] = rng.choice(donor_words)
    return " ".join(words)


def corrupt_partial(text: str, donor_text: str, rng: random.Random) -> str:
    """Thay một đoạn liên tục 40–60% bằng nội dung khác -> mức `Partial`."""
    words = tokenize(text)
    if len(words) < 4:
        return donor_text

    frac = rng.uniform(0.4, 0.6)
    span = max(2, int(len(words) * frac))
    start = rng.randrange(0, max(1, len(words) - span + 1))

    donor = tokenize(donor_text)
    replacement = donor[:span] if len(donor) >= span else donor
    return " ".join(words[:start] + replacement + words[start + span:])


def corrupt_detached(donor_text: str) -> str:
    """Thay toàn bộ bằng một câu không liên quan -> `Full`, loại tách rời.

    Đây là loại ảo giác trôi chảy nhưng hoàn toàn lạc đề — embedding bắt tốt,
    còn bộ đếm lặp n-gram thì mù hoàn toàn.
    """
    return donor_text


def corrupt_oscillatory(text: str, rng: random.Random, n: int = 4) -> str:
    """Lặp một cụm cho tới khi đủ dài -> `Full`, loại dao động.

    P2 §6.4 đo được 58–86% ảo giác thật thuộc loại này, nên nó phải có mặt
    trong tập tổng hợp với tỉ trọng tương xứng.

    Số lần lặp tối thiểu là 4, không phải 3. Lý do: bộ dò của P2 gắn cờ khi
    mức lặp *thừa* so với câu nguồn **vượt quá** 2. Lặp 3 lần cho mức thừa
    đúng bằng 2 — nằm ngay trên ranh giới và không bị bắt. Mẫu dao động tổng
    hợp mà chính bộ dò dao động bỏ sót thì vô dụng cho cả huấn luyện lẫn phân
    tích lỗi.
    """
    words = tokenize(text)
    if len(words) < n:
        return " ".join([text] * 4)

    start = rng.randrange(0, max(1, len(words) - n + 1))
    phrase = words[start : start + n]
    # Lặp cho tới khi dài khoảng 1,5 lần câu gốc — giống hành vi thật của mô
    # hình NMT khi nó lặp tới lúc chạm giới hạn độ dài.
    repeats = max(4, int(len(words) * 1.5) // n)
    return " ".join(words[:start] + phrase * repeats)


def corrupt_off_target(rng: random.Random, exclude_lang: str = "") -> tuple[str, str]:
    """Thay bằng câu thuộc ngôn ngữ khác -> `Full`, loại sai ngôn ngữ đích.

    Returns:
        (văn bản, mã ngôn ngữ đã dùng).
    """
    langs = [k for k in OFF_TARGET_POOL if k != exclude_lang.lower()]
    lang = rng.choice(sorted(langs))
    return rng.choice(OFF_TARGET_POOL[lang]), lang


# --------------------------------------------------------------------------
# Bộ điều phối
# --------------------------------------------------------------------------

#: Tỉ trọng mặc định của từng loại.
#: - Một nửa là lớp âm, để tập dữ liệu không lệch về phía dương một cách vô lý.
#: - Trong nhóm dương, `oscillatory` chiếm tỉ trọng cao nhất cho khớp quan sát
#:   58–86% của P2.
DEFAULT_MIX: dict[str, float] = {
    "none": 0.50,
    "small": 0.10,
    "partial": 0.12,
    "full_oscillatory": 0.14,
    "full_detached": 0.09,
    "off_target": 0.05,
}

SEVERITY_OF_TYPE: dict[str, int] = {
    "none": 0,
    "small": 1,
    "partial": 2,
    "full_oscillatory": 3,
    "full_detached": 3,
    "off_target": 3,
}


def make_example(
    src_text: str,
    reference: str,
    kind: str,
    donor_words: Sequence[str],
    donor_text: str,
    rng: random.Random,
    tgt_lang: str = "VI",
) -> SyntheticExample:
    """Tạo một mẫu tổng hợp thuộc loại `kind`."""
    if kind not in SEVERITY_OF_TYPE:
        raise KeyError(f"Loai khong hop le: {kind!r}. Cac lua chon: "
                       f"{sorted(SEVERITY_OF_TYPE)}")

    if kind == "none":
        mt = reference
    elif kind == "small":
        mt = corrupt_small(reference, donor_words, rng)
    elif kind == "partial":
        mt = corrupt_partial(reference, donor_text, rng)
    elif kind == "full_oscillatory":
        mt = corrupt_oscillatory(reference, rng)
    elif kind == "full_detached":
        mt = corrupt_detached(donor_text)
    else:  # off_target
        mt, _lang = corrupt_off_target(rng, exclude_lang=tgt_lang)

    return SyntheticExample(
        src_text=src_text,
        mt_text=mt,
        severity=SEVERITY_OF_TYPE[kind],
        hallucination_type=kind,
        reference=reference,
    )


def _pick_donor(
    references: Sequence[str],
    i: int,
    rng: random.Random,
    max_tries: int = 20,
) -> str:
    """Chọn một bản dịch chuẩn **khác nội dung** với câu thứ `i`.

    So sánh theo *nội dung* chứ không theo chỉ số. Nếu chỉ tránh trùng chỉ số
    thì trong ngữ liệu có câu lặp lại (rất phổ biến với phụ đề và dữ liệu web),
    câu cho mượn vẫn có thể giống hệt bản dịch hiện tại — sinh ra mẫu gắn nhãn
    `Full hallucination` nhưng thực chất là bản dịch hoàn hảo. Loại lỗi này
    đầu độc cả tập huấn luyện lẫn phần phân tích lỗi mà không hề báo lỗi.
    """
    n = len(references)
    target = references[i]

    for _ in range(max_tries):
        j = rng.randrange(n)
        if references[j] != target:
            return references[j]

    # Bốc ngẫu nhiên thất bại (ngữ liệu lặp nhiều) -> quét tuyến tính
    for j in range(n):
        if references[j] != target:
            return references[j]

    raise ValueError(
        "Moi ban dich chuan trong ngu lieu deu giong nhau, khong the tao "
        "ao giac 'tach roi'. Hay kiem tra lai du lieu dau vao."
    )


def build_synthetic_set(
    pairs: Sequence[tuple[str, str]],
    mix: dict[str, float] | None = None,
    tgt_lang: str = "VI",
    seed: int = 42,
) -> list[SyntheticExample]:
    """Sinh Tập B từ danh sách cặp song ngữ chuẩn.

    Args:
        pairs: danh sách (câu nguồn, bản dịch chuẩn).
        mix: tỉ trọng từng loại; mặc định `DEFAULT_MIX`.
        tgt_lang: ngôn ngữ đích, để loại nó khỏi kho câu sai-ngôn-ngữ.
        seed: hạt giống ngẫu nhiên, đảm bảo tái lập.

    Raises:
        ValueError: nếu `pairs` rỗng hoặc tỉ trọng không hợp lệ.
    """
    if not pairs:
        raise ValueError("pairs khong duoc rong")

    mix = mix or DEFAULT_MIX
    if abs(sum(mix.values()) - 1.0) > 1e-6:
        raise ValueError(f"Tong ti trong phai bang 1, dang la {sum(mix.values())}")
    unknown = set(mix) - set(SEVERITY_OF_TYPE)
    if unknown:
        raise ValueError(f"Loai khong hop le trong mix: {sorted(unknown)}")

    rng = random.Random(seed)

    # Kho từ và kho câu để lấy nội dung "không liên quan".
    # Lấy từ chính ngữ liệu nên nội dung chèn vào vẫn trôi chảy và đúng ngôn
    # ngữ — bản dịch hỏng vì thế khó phát hiện hơn, tức là bài toán khó hơn
    # và công bằng hơn cho detector.
    references = [ref for _src, ref in pairs]
    donor_words = sorted({w for ref in references for w in _TOKEN_RE.findall(ref)
                          if len(w) > 2})
    if not donor_words:
        raise ValueError("Khong trich duoc tu nao tu ban dich chuan")

    # Dựng lịch phân bổ loại, đảm bảo đúng tỉ trọng thay vì bốc ngẫu nhiên
    n = len(pairs)
    schedule: list[str] = []
    for kind, share in sorted(mix.items()):
        schedule.extend([kind] * int(round(n * share)))
    while len(schedule) < n:
        schedule.append("none")
    schedule = schedule[:n]
    rng.shuffle(schedule)

    out: list[SyntheticExample] = []
    for i, ((src, ref), kind) in enumerate(zip(pairs, schedule)):
        donor_text = _pick_donor(references, i, rng)
        out.append(
            make_example(src, ref, kind, donor_words, donor_text, rng, tgt_lang)
        )
    return out
