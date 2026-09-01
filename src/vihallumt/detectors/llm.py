"""Bộ phát hiện ảo giác bằng LLM — tái hiện P1 kèm một cải tiến.

Hai chế độ chấm điểm
--------------------

``mode="logit"`` — **cải tiến của đồ án.** Chỉ chạy MỘT lượt truyền xuôi, đọc
phân phối xác suất của token đầu tiên, rồi so xác suất giữa token mở đầu nhãn
dương và nhãn âm::

    p(ảo giác) = P(pos) / (P(pos) + P(neg))

Ưu điểm so với cách của P1:
  * cho **điểm liên tục** -> tính được ROC-AUC và hiệu chỉnh được ngưỡng,
    trong khi P1 chỉ có nhãn cứng nên không so sánh được theo AUC;
  * **nhanh hơn khoảng một bậc**, vì không phải sinh chuỗi;
  * **không bao giờ trả lời sai định dạng** — P1 phải loại bỏ những lần mô hình
    trả lời lan man, còn ở đây điều đó không thể xảy ra.

``mode="generate"`` — cách gốc của P1: sinh tối đa `max_new_tokens` token rồi
bóc nhãn từ văn bản. **Bắt buộc dùng chế độ này cho các biến thể CoT**, vì CoT
cần mô hình lập luận ra thành lời trước khi kết luận; token đầu tiên khi đó là
chữ đầu của phần lập luận chứ không phải nhãn.

Cấu hình theo P1 Appendix D.1: temperature = 0 (giải mã tham lam),
max_output_token = 15, zero-shot, tập nhãn ràng buộc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from vihallumt.detectors.base import Detector, Pair
from vihallumt.prompts.binary import CoTId, Language, Prompt, PromptId, build_prompt


def _accelerate_available() -> bool:
    """`accelerate` co san khong — quyet dinh co dung duoc `device_map` hay khong."""
    try:
        import accelerate  # noqa: F401
        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------
# Hàm thuần tuý — kiểm thử được mà không cần mô hình
# --------------------------------------------------------------------------

def label_first_token_ids(tokenizer: Any, label: str) -> set[int]:
    """Tập id token có thể mở đầu cho một nhãn.

    Cần xét nhiều biến thể bề mặt vì bộ tách token đối xử khác nhau với chữ
    hoa/thường và với dấu cách đứng trước. Ví dụ Llama tách " Hallucination"
    khác với "Hallucination".
    """
    variants = {
        label,
        label.lower(),
        label.upper(),
        label[:1].upper() + label[1:],
        " " + label,
        " " + label.lower(),
    }
    ids: set[int] = set()
    for v in variants:
        toks = tokenizer.encode(v, add_special_tokens=False)
        if len(toks) > 0:
            ids.add(int(toks[0]))
    return ids


def resolve_label_token_ids(
    tokenizer: Any, labels: tuple[str, str]
) -> tuple[set[int], set[int]]:
    """Trả về (id nhãn âm, id nhãn dương), đã loại bỏ phần giao nhau.

    Raises:
        ValueError: nếu hai nhãn không phân biệt được ở token đầu tiên. Khi đó
            logit-scoring không dùng được và phải chuyển sang `mode="generate"`.
    """
    neg_label, pos_label = labels
    neg_ids = label_first_token_ids(tokenizer, neg_label)
    pos_ids = label_first_token_ids(tokenizer, pos_label)

    overlap = neg_ids & pos_ids
    neg_ids -= overlap
    pos_ids -= overlap

    if not neg_ids or not pos_ids:
        raise ValueError(
            f"Khong phan biet duoc hai nhan {labels!r} o token dau tien "
            f"(giao nhau: {sorted(overlap)}). Hay dung mode='generate'."
        )
    return neg_ids, pos_ids


def hallucination_prob_from_logits(
    logits: np.ndarray,
    neg_ids: Iterable[int],
    pos_ids: Iterable[int],
) -> np.ndarray:
    """Đổi logit của token kế tiếp thành xác suất ảo giác.

    Chuẩn hoá **chỉ trên hai nhãn** chứ không trên toàn bộ từ vựng, nên xác
    suất luôn nằm trong [0, 1] và không bị loãng bởi các token không liên quan.

    Args:
        logits: mảng (batch, vocab) hoặc (vocab,).
        neg_ids, pos_ids: tập id token mở đầu mỗi nhãn.

    Returns:
        Mảng (batch,) chứa p(ảo giác).
    """
    logits = np.atleast_2d(np.asarray(logits, dtype=np.float64))
    neg_ids, pos_ids = list(neg_ids), list(pos_ids)
    if not neg_ids or not pos_ids:
        raise ValueError("neg_ids va pos_ids khong duoc rong")

    # Trừ đi max theo hàng cho ổn định số học trước khi lấy exp
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)

    p_neg = exp[:, neg_ids].sum(axis=1)
    p_pos = exp[:, pos_ids].sum(axis=1)
    total = p_neg + p_pos

    # Nếu cả hai nhãn đều gần như không thể xảy ra thì trả về 0.5 (không biết)
    out = np.where(total > 0, p_pos / np.where(total > 0, total, 1.0), 0.5)
    return out


def parse_generated_label(text: str, labels: tuple[str, str]) -> int | None:
    """Bóc nhãn 0/1 từ văn bản mô hình sinh ra. Trả về None nếu không nhận ra.

    Xử lý được trường hợp nhãn âm chứa nhãn dương như một chuỗi con
    ("No hallucination" chứa "hallucination", "Không có ảo giác" chứa
    "có ảo giác") bằng cách so vị trí xuất hiện đầu tiên.
    """
    neg_label, pos_label = labels
    t = text.strip().lower()
    neg, pos = neg_label.lower(), pos_label.lower()

    i_neg, i_pos = t.find(neg), t.find(pos)

    if i_neg < 0 and i_pos < 0:
        return None
    if i_neg < 0:
        return 1
    if i_pos < 0:
        return 0
    if i_neg != i_pos:
        return 0 if i_neg < i_pos else 1
    # Cùng vị trí bắt đầu -> nhãn dài hơn là nhãn khớp thật sự
    return 0 if len(neg) >= len(pos) else 1


# --------------------------------------------------------------------------
# Bộ phát hiện
# --------------------------------------------------------------------------

@dataclass
class LLMConfig:
    """Cấu hình sinh, mặc định theo P1 Appendix D.1."""

    prompt_id: PromptId = "p2"
    cot: CoTId = "none"
    language: Language = "en"
    label_language: Language | None = None
    mode: str = "logit"           # "logit" | "generate"
    max_new_tokens: int = 15      # P1: MAX_OUTPUT_TOKEN = 15
    batch_size: int = 8
    load_in_4bit: bool = True

    def __post_init__(self) -> None:
        if self.mode not in ("logit", "generate"):
            raise ValueError(f"mode khong hop le: {self.mode}")
        if self.mode == "logit" and self.cot != "none":
            raise ValueError(
                "Khong the dung logit-scoring voi CoT: token dau tien khi do la "
                "chu dau cua phan lap luan chu khong phai nhan. Dung mode='generate'."
            )


class LLMDetector(Detector):
    """Phát hiện ảo giác bằng LLM nhân quả chạy cục bộ (HuggingFace).

    Ví dụ::

        det = LLMDetector("Qwen/Qwen2.5-7B-Instruct")
        scores = det.score_frame(df)     # p(ảo giác) trong [0, 1]

    Việc nạp mô hình diễn ra lười (lazy) ở lần chấm điểm đầu tiên, nên khởi tạo
    đối tượng không cần GPU — thuận tiện cho kiểm thử.
    """

    produces_continuous_score = True

    def __init__(
        self,
        model_id: str,
        config: LLMConfig | None = None,
        device: str | None = None,
        model: Any = None,
        tokenizer: Any = None,
    ) -> None:
        self.model_id = model_id
        self.config = config or LLMConfig()
        self.device = device
        self._model = model
        self._tokenizer = tokenizer

        from vihallumt.prompts.binary import variant_name

        self.name = (
            f"{model_id.split('/')[-1]} "
            f"[{variant_name(self.config.prompt_id, self.config.cot, self.config.language)}"
            f", {self.config.mode}]"
        )
        self.produces_continuous_score = self.config.mode == "logit"

    # -- nạp mô hình -------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        # Mô hình nhân quả cần đệm bên trái để vị trí cuối cùng của mọi câu
        # trong lô đều là token thật, không phải token đệm.
        self._tokenizer.padding_side = "left"

        if self._model is None:
            kwargs: dict[str, Any] = {"dtype": "auto"}
            has_cuda = torch.cuda.is_available()

            # `device_map` cần `accelerate`; trên máy CPU không có accelerate thì
            # nạp thẳng rồi tự chuyển thiết bị. Nhờ vậy smoke-test chạy được ở
            # mọi nơi mà đường chạy GPU trên Kaggle vẫn tối ưu.
            if _accelerate_available():
                kwargs["device_map"] = self.device or ("auto" if has_cuda else "cpu")

            if self.config.load_in_4bit and has_cuda:
                from vihallumt.corpus.translate import _bitsandbytes_available

                if _bitsandbytes_available():
                    from transformers import BitsAndBytesConfig

                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                else:
                    # Thiếu bitsandbytes -> chạy fp16 thay vì vỡ. Tốn bộ nhớ
                    # gấp ~4 lần nhưng vẫn chạy được; dừng giữa chừng sau khi
                    # đã chấm điểm hàng nghìn câu thì tệ hơn nhiều.
                    print(f"    [canh bao] chua cai bitsandbytes -> {self.model_id} "
                          f"chay fp16 thay vi 4-bit. Cai: pip install bitsandbytes")
                    kwargs["dtype"] = torch.float16

            self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)

            if "device_map" not in kwargs:
                self._model = self._model.to(self.device or ("cuda" if has_cuda else "cpu"))

        self._model.eval()

    # -- dựng prompt -------------------------------------------------------

    def build(self, pair: Pair) -> Prompt:
        c = self.config
        return build_prompt(
            src_text=pair.src_text,
            mt_text=pair.mt_text,
            src_lang=pair.src_lang,
            tgt_lang=pair.tgt_lang,
            prompt_id=c.prompt_id,
            cot=c.cot,
            language=c.language,
            label_language=c.label_language,
        )

    def _render(self, prompts: Sequence[Prompt]) -> list[str]:
        """Áp chat template, để ngỏ lượt của trợ lý cho mô hình điền tiếp."""
        return [
            self._tokenizer.apply_chat_template(
                p.as_messages(), tokenize=False, add_generation_prompt=True
            )
            for p in prompts
        ]

    # -- chấm điểm ---------------------------------------------------------

    def score(self, pairs: list[Pair]) -> np.ndarray:
        if not pairs:
            return np.array([], dtype=float)
        self._ensure_loaded()
        fn = self._score_logit if self.config.mode == "logit" else self._score_generate
        out: list[np.ndarray] = []
        bs = self.config.batch_size
        for i in range(0, len(pairs), bs):
            out.append(fn(pairs[i : i + bs]))
        return np.concatenate(out)

    def _score_logit(self, batch: list[Pair]) -> np.ndarray:
        import torch

        prompts = [self.build(p) for p in batch]
        neg_ids, pos_ids = resolve_label_token_ids(self._tokenizer, prompts[0].labels)

        enc = self._tokenizer(
            self._render(prompts), return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self._model.device)

        with torch.no_grad():
            logits = self._model(**enc).logits[:, -1, :]

        return hallucination_prob_from_logits(
            logits.float().cpu().numpy(), neg_ids, pos_ids
        )

    def _score_generate(self, batch: list[Pair]) -> np.ndarray:
        import torch

        prompts = [self.build(p) for p in batch]
        enc = self._tokenizer(
            self._render(prompts), return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self._model.device)

        with torch.no_grad():
            out = self._model.generate(
                **enc,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,           # temperature = 0 theo P1
                pad_token_id=self._tokenizer.pad_token_id,
            )

        generated = out[:, enc["input_ids"].shape[1] :]
        texts = self._tokenizer.batch_decode(generated, skip_special_tokens=True)

        labels = prompts[0].labels
        # Không nhận ra nhãn -> 0.5, tức là "không kết luận được".
        # P1 loại bỏ những trường hợp này; ta giữ lại và báo cáo tỉ lệ.
        return np.array(
            [0.5 if (v := parse_generated_label(t, labels)) is None else float(v)
             for t in texts],
            dtype=float,
        )
