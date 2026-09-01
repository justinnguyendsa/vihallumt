"""Prompt phát hiện ảo giác nhị phân — chép đúng nguyên văn từ P1.

Nguồn: Benkirane et al. (2024), Appendix C.
  Prompt 1  — Figure 10, phỏng theo G-Eval
  Prompt 2  — Figure 11, G-Eval + nói rõ tên ngôn ngữ  (chính là Figure 3 ở thân bài)
  Prompt 3  — Figure 12, prompt do người thiết kế
  CoT 1     — Figure 13, theo hướng dẫn gán nhãn của HalOmi
  CoT 2     — Figure 14, hướng dẫn HalOmi + chiến lược đếm từ ảo giác

Cấu hình sinh của P1 (Appendix D.1): temperature = 0, max_output_token = 15,
zero-shot, tập nhãn ràng buộc.

Phần mở rộng của đồ án: mỗi prompt có thêm **biến thể tiếng Việt** — P1 chỉ
prompt bằng tiếng Anh. Giả thuyết cần kiểm định là với ngôn ngữ trung tài
nguyên, ra lệnh bằng chính ngôn ngữ đích có giúp mô hình hiểu tác vụ tốt hơn
không. Để cô lập biến số, biến thể `vi` **giữ nguyên nhãn tiếng Anh** và chỉ
dịch phần hướng dẫn; muốn đổi cả nhãn thì dùng `label_language="vi"`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PromptId = Literal["p1", "p2", "p3"]
CoTId = Literal["none", "cot1", "cot2"]
Language = Literal["en", "vi"]

# Tên đầy đủ của ngôn ngữ, dùng để điền vào chỗ trống src_lang / tgt_lang.
LANGUAGE_NAMES_EN: dict[str, str] = {
    "EN": "English", "DE": "German", "AR": "Arabic", "ZH": "Chinese",
    "RU": "Russian", "ES": "Spanish", "KS": "Kashmiri", "MN": "Manipuri",
    "YO": "Yoruba", "VI": "Vietnamese",
}

LANGUAGE_NAMES_VI: dict[str, str] = {
    "EN": "tiếng Anh", "DE": "tiếng Đức", "AR": "tiếng Ả Rập", "ZH": "tiếng Trung",
    "RU": "tiếng Nga", "ES": "tiếng Tây Ban Nha", "KS": "tiếng Kashmir",
    "MN": "tiếng Manipuri", "YO": "tiếng Yoruba", "VI": "tiếng Việt",
}

# Cặp nhãn (âm, dương). Thứ tự này cố định trong toàn bộ mã nguồn.
LABELS: dict[Language, tuple[str, str]] = {
    "en": ("No hallucination", "Hallucination"),
    "vi": ("Không có ảo giác", "Có ảo giác"),
}


# --------------------------------------------------------------------------
# Chuỗi Chain-of-Thought (P1 Figure 13, 14)
# --------------------------------------------------------------------------

COT1_EN = """Evaluation Steps:
1. Read the source text and the translated text carefully.
2. To decide whether the translated text contains hallucinations check if the source tokens "correspond" to erroneous target tokens. For each token answer:

- Does this source word fall into the common meaning category as this target word?
- Does this source word have a semantic connection with this target word?
- Can you try to come up with a reasonable theory on how this source word is associated with this target word?

3. If "no" to all the questions above, then hallucination"""

COT2_EN = """Evaluation Steps:
1. Read the source text and the translated text carefully.
2. Initialize a counter 'n = 0' for the number of hallucinated words.
3. To decide whether the translated text contains hallucinations check if the source tokens "correspond" to erroneous target tokens. For each token answer:

- Does this source word fall into the common meaning category as this target word?
- Does this source word have a semantic connection with this target word?
- Can you try to come up with a reasonable theory on how this source word is associated with this target word?
- If "no" to all the questions above, then hallucination

4. After analyzing each word in the translated text:

- If 'n == 0', assign the label 'No hallucination'.
- If 'n' is 1 or more, assign the label 'Hallucination'."""

COT1_VI = """Các bước đánh giá:
1. Đọc kỹ văn bản nguồn và văn bản đã dịch.
2. Để quyết định bản dịch có chứa ảo giác hay không, hãy kiểm tra xem các token nguồn có "tương ứng" với token đích sai lệch hay không. Với mỗi token, hãy trả lời:

- Từ nguồn này có thuộc cùng phạm trù nghĩa với từ đích này không?
- Từ nguồn này có mối liên hệ ngữ nghĩa nào với từ đích này không?
- Bạn có thể đưa ra một cách giải thích hợp lý về việc từ nguồn này liên quan tới từ đích này không?

3. Nếu câu trả lời là "không" cho tất cả các câu hỏi trên thì đó là ảo giác"""

COT2_VI = """Các bước đánh giá:
1. Đọc kỹ văn bản nguồn và văn bản đã dịch.
2. Khởi tạo biến đếm 'n = 0' cho số từ bị ảo giác.
3. Để quyết định bản dịch có chứa ảo giác hay không, hãy kiểm tra xem các token nguồn có "tương ứng" với token đích sai lệch hay không. Với mỗi token, hãy trả lời:

- Từ nguồn này có thuộc cùng phạm trù nghĩa với từ đích này không?
- Từ nguồn này có mối liên hệ ngữ nghĩa nào với từ đích này không?
- Bạn có thể đưa ra một cách giải thích hợp lý về việc từ nguồn này liên quan tới từ đích này không?
- Nếu câu trả lời là "không" cho tất cả các câu hỏi trên thì đó là ảo giác

4. Sau khi phân tích từng từ trong văn bản đã dịch:

- Nếu 'n == 0', gán nhãn 'No hallucination'.
- Nếu 'n' từ 1 trở lên, gán nhãn 'Hallucination'."""

COT_TEXT: dict[tuple[CoTId, Language], str] = {
    ("cot1", "en"): COT1_EN, ("cot2", "en"): COT2_EN,
    ("cot1", "vi"): COT1_VI, ("cot2", "vi"): COT2_VI,
}


# --------------------------------------------------------------------------
# Phần system của từng prompt
# --------------------------------------------------------------------------

def _system_p1_en(src_lang: str, tgt_lang: str) -> str:
    """P1 Figure 10 — phỏng theo G-Eval. Không nhắc tên ngôn ngữ."""
    return (
        "Human Evaluation of Machine Translation Systems:\n"
        "Hallucination Evaluation Criteria: Does the translated text contain "
        "information completely unrelated to the source text?\n"
        "- Hallucination: there is hallucination.\n"
        "- No Hallucination: there is no hallucination."
    )


def _system_p2_en(src_lang: str, tgt_lang: str) -> str:
    """P1 Figure 11 (= Figure 3 ở thân bài) — G-Eval + nói rõ tên ngôn ngữ."""
    return (
        "Instructions for Evaluating Machine Translation:\n"
        f"You will be given a source text in {src_lang} and a machine translated "
        f"text in {tgt_lang}.\n"
        "Your task is to identify if the machine translated text has hallucination or not.\n"
        "Please make sure you read and understand these instructions carefully.\n"
        "Please keep this document open while reviewing, and refer to it as needed.\n"
        "Evaluation Criteria:\n"
        "Hallucination: Does the translated text contain information completely "
        "unrelated to the source text?\n"
        "- Hallucination: there is hallucination.\n"
        "- No Hallucination: there is no hallucination."
    )


def _system_p3_en(src_lang: str, tgt_lang: str) -> str:
    """P1 Figure 12 — prompt do người thiết kế, có định nghĩa ảo giác tường minh."""
    return (
        "Instructions for Evaluating Machine Translation:\n"
        f"You will be given a source text in {src_lang} and a machine translated "
        f"text in {tgt_lang}.\n"
        "Your task is to identify if the machine translated text has hallucination or not.\n"
        "Please make sure you read and understand these instructions carefully.\n"
        "Please keep this document open while reviewing, and refer to it as needed.\n"
        "Definition of Hallucination: The translated text is considered a hallucination "
        "if it introduces information that is completely unrelated to the source text.\n"
        "Hallucination labels:\n"
        "- Hallucination: there is hallucination.\n"
        "- No hallucination: there is no hallucination."
    )


def _system_p1_vi(src_lang: str, tgt_lang: str) -> str:
    return (
        "Đánh giá của con người đối với hệ thống dịch máy:\n"
        "Tiêu chí đánh giá ảo giác: Văn bản đã dịch có chứa thông tin hoàn toàn "
        "không liên quan đến văn bản nguồn hay không?\n"
        "- Hallucination: có ảo giác.\n"
        "- No Hallucination: không có ảo giác."
    )


def _system_p2_vi(src_lang: str, tgt_lang: str) -> str:
    return (
        "Hướng dẫn đánh giá dịch máy:\n"
        f"Bạn sẽ được cung cấp một văn bản nguồn bằng {src_lang} và một văn bản "
        f"do máy dịch sang {tgt_lang}.\n"
        "Nhiệm vụ của bạn là xác định văn bản do máy dịch có bị ảo giác hay không.\n"
        "Hãy đọc và hiểu kỹ các hướng dẫn này.\n"
        "Hãy giữ tài liệu này mở trong khi đánh giá và tham khảo khi cần.\n"
        "Tiêu chí đánh giá:\n"
        "Ảo giác: Văn bản đã dịch có chứa thông tin hoàn toàn không liên quan đến "
        "văn bản nguồn hay không?\n"
        "- Hallucination: có ảo giác.\n"
        "- No Hallucination: không có ảo giác."
    )


def _system_p3_vi(src_lang: str, tgt_lang: str) -> str:
    return (
        "Hướng dẫn đánh giá dịch máy:\n"
        f"Bạn sẽ được cung cấp một văn bản nguồn bằng {src_lang} và một văn bản "
        f"do máy dịch sang {tgt_lang}.\n"
        "Nhiệm vụ của bạn là xác định văn bản do máy dịch có bị ảo giác hay không.\n"
        "Hãy đọc và hiểu kỹ các hướng dẫn này.\n"
        "Hãy giữ tài liệu này mở trong khi đánh giá và tham khảo khi cần.\n"
        "Định nghĩa ảo giác: Văn bản đã dịch bị coi là ảo giác nếu nó đưa vào thông "
        "tin hoàn toàn không liên quan đến văn bản nguồn.\n"
        "Các nhãn ảo giác:\n"
        "- Hallucination: có ảo giác.\n"
        "- No hallucination: không có ảo giác."
    )


_SYSTEM_BUILDERS = {
    ("p1", "en"): _system_p1_en, ("p2", "en"): _system_p2_en, ("p3", "en"): _system_p3_en,
    ("p1", "vi"): _system_p1_vi, ("p2", "vi"): _system_p2_vi, ("p3", "vi"): _system_p3_vi,
}


# --------------------------------------------------------------------------
# Phần user
# --------------------------------------------------------------------------

def _user_en(prompt_id: PromptId, src_text: str, mt_text: str, labels: tuple[str, str]) -> str:
    neg, pos = labels
    head = f"Source Text: {src_text}\nTranslated Text: {mt_text}\n"
    if prompt_id == "p3":
        # Figure 12 dùng cách hỏi khác hai prompt kia
        return (
            head
            + "Provide exactly one of the following hallucination labels as your response. "
            "Do not include any additional text or explanation:\n"
            f"- {pos}\n- {neg}\n"
        )
    return (
        head
        + "Does the translation contain hallucination?\n"
        f"Answer (label ONLY: '{pos}' OR '{neg}'):"
    )


def _user_vi(prompt_id: PromptId, src_text: str, mt_text: str, labels: tuple[str, str]) -> str:
    neg, pos = labels
    head = f"Văn bản nguồn: {src_text}\nVăn bản đã dịch: {mt_text}\n"
    if prompt_id == "p3":
        return (
            head
            + "Hãy đưa ra đúng một trong các nhãn ảo giác sau đây làm câu trả lời. "
            "Không thêm bất kỳ văn bản hay giải thích nào khác:\n"
            f"- {pos}\n- {neg}\n"
        )
    return (
        head
        + "Bản dịch có chứa ảo giác không?\n"
        f"Trả lời (CHỈ nhãn: '{pos}' HOẶC '{neg}'):"
    )


# --------------------------------------------------------------------------
# API công khai
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Prompt:
    """Một prompt đã dựng xong, sẵn sàng đưa vào chat template."""

    system: str
    user: str
    labels: tuple[str, str]  # (nhãn âm, nhãn dương)

    def as_messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]


def build_prompt(
    src_text: str,
    mt_text: str,
    src_lang: str = "EN",
    tgt_lang: str = "VI",
    prompt_id: PromptId = "p2",
    cot: CoTId = "none",
    language: Language = "en",
    label_language: Language | None = None,
) -> Prompt:
    """Dựng một prompt phát hiện ảo giác nhị phân.

    Args:
        src_text, mt_text: cặp câu cần đánh giá.
        src_lang, tgt_lang: viết tắt ngôn ngữ ("EN", "VI", ...).
        prompt_id: "p1" | "p2" | "p3" theo Figure 10/11/12 của P1.
            Mặc định "p2" vì đó là prompt được P1 in ở thân bài (Figure 3).
        cot: "none" | "cot1" | "cot2" theo Figure 13/14.
        language: ngôn ngữ của phần hướng dẫn.
        label_language: ngôn ngữ của nhãn trả lời. Mặc định luôn là "en" để
            cô lập biến số khi so sánh prompt Anh với prompt Việt.

    Raises:
        KeyError: nếu mã ngôn ngữ chưa có trong bảng tên.
    """
    if prompt_id not in ("p1", "p2", "p3"):
        raise ValueError(f"prompt_id khong hop le: {prompt_id}")
    if cot not in ("none", "cot1", "cot2"):
        raise ValueError(f"cot khong hop le: {cot}")

    names = LANGUAGE_NAMES_EN if language == "en" else LANGUAGE_NAMES_VI
    src_name, tgt_name = names[src_lang], names[tgt_lang]

    system = _SYSTEM_BUILDERS[(prompt_id, language)](src_name, tgt_name)
    if cot != "none":
        system = system + "\n\n" + COT_TEXT[(cot, language)]

    labels = LABELS[label_language or "en"]
    user_builder = _user_en if language == "en" else _user_vi
    user = user_builder(prompt_id, src_text, mt_text, labels)

    return Prompt(system=system, user=user, labels=labels)


# Toàn bộ lưới biến thể mà P1 quét trên tập validation (Bảng 6),
# cộng thêm hai biến thể tiếng Việt của đồ án.
PROMPT_GRID: tuple[dict, ...] = (
    {"prompt_id": "p1", "cot": "none", "language": "en"},
    {"prompt_id": "p1", "cot": "cot1", "language": "en"},
    {"prompt_id": "p2", "cot": "none", "language": "en"},
    {"prompt_id": "p2", "cot": "cot1", "language": "en"},
    {"prompt_id": "p3", "cot": "none", "language": "en"},
    {"prompt_id": "p3", "cot": "cot1", "language": "en"},
    {"prompt_id": "p3", "cot": "cot2", "language": "en"},
    # Mở rộng của đồ án: hướng dẫn bằng tiếng Việt
    {"prompt_id": "p2", "cot": "none", "language": "vi"},
    {"prompt_id": "p3", "cot": "none", "language": "vi"},
    {"prompt_id": "p3", "cot": "cot2", "language": "vi"},
)


def variant_name(prompt_id: PromptId, cot: CoTId, language: Language) -> str:
    """Tên ngắn gọn của một biến thể, dùng làm khoá trong bảng kết quả."""
    cot_tag = "" if cot == "none" else f"+{cot}"
    return f"{prompt_id}{cot_tag}[{language}]"
