"""Bộ bọc hệ dịch máy để sinh Tập A của ViHalluMT.

Ta cần bản dịch **thật** do mô hình sinh ra, kể cả bản hỏng — đó mới là ảo giác
tự nhiên. Vì thế module này cho phép điều khiển cả hệ dịch lẫn cấu hình giải mã.

Vì sao cần nhiều hệ dịch
------------------------
Mỗi hệ ảo giác theo cách khác nhau. NLLB-600M (encoder-decoder nhỏ) hay dao
động và dịch sai ngôn ngữ; envit5 chuyên Việt–Anh nên mạnh hơn, dùng làm đối
chứng "ít ảo giác". Trộn nhiều hệ giúp bộ dữ liệu không bị đóng khuôn theo một
kiểu lỗi duy nhất — đây chính là điều HalOmi bị phê là thiếu (họ chỉ dùng NLLB).

Vì sao cần nhiều cấu hình giải mã
---------------------------------
P2 §5.1 cho thấy chiến lược giải mã tác động rất mạnh tới ảo giác. `greedy` và
`beam` cho bản dịch tốt; `sampling` nhiệt độ cao và `epsilon` đẩy mô hình vào
vùng đuôi phân phối, nơi ảo giác xuất hiện thường xuyên hơn nhiều. Ta dùng
chính công cụ mà P2 dùng để *chữa* ảo giác, nhưng theo chiều ngược lại, để
*gây* ra chúng.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

#: Mã ngôn ngữ FLORES-200 mà NLLB dùng.
NLLB_LANG: dict[str, str] = {
    "EN": "eng_Latn",
    "VI": "vie_Latn",
    "ZH": "zho_Hans",
    "TH": "tha_Thai",
    "ID": "ind_Latn",
}


@dataclass(frozen=True)
class DecodingConfig:
    """Cấu hình giải mã. Tên gọi được ghi vào dữ liệu để phân tích về sau."""

    name: str
    num_beams: int = 1
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    epsilon_cutoff: float = 0.0
    max_new_tokens: int = 128

    def to_kwargs(self) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "num_beams": self.num_beams,
            "do_sample": self.do_sample,
            "max_new_tokens": self.max_new_tokens,
        }
        if self.do_sample:
            kw["temperature"] = self.temperature
            kw["top_p"] = self.top_p
            if self.epsilon_cutoff > 0:
                kw["epsilon_cutoff"] = self.epsilon_cutoff
        return kw


#: Các cấu hình dùng khi sinh Tập A, xếp từ "lành" tới "dễ gây ảo giác".
DECODING_PRESETS: tuple[DecodingConfig, ...] = (
    DecodingConfig("beam5", num_beams=5),
    DecodingConfig("greedy", num_beams=1),
    DecodingConfig("sample_t1.2", do_sample=True, temperature=1.2, top_p=0.95),
    DecodingConfig("sample_t1.8", do_sample=True, temperature=1.8, top_p=0.98),
    DecodingConfig("epsilon", do_sample=True, temperature=1.0, epsilon_cutoff=0.02),
)


class Translator:
    """Lớp cơ sở cho hệ dịch."""

    name: str = "translator"

    def translate(
        self,
        texts: list[str],
        src_lang: str = "EN",
        tgt_lang: str = "VI",
        decoding: DecodingConfig | None = None,
        batch_size: int = 16,
    ) -> list[str]:
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}(name={self.name!r})"


class NLLBTranslator(Translator):
    """NLLB-200 — cùng dòng mô hình HalOmi dùng để tạo benchmark gốc.

    Args:
        model_id: mặc định bản chưng cất 600M, chạy vừa GPU T4.
        force_wrong_target: ép token ngôn ngữ đích sai để **cố tình** tạo ảo
            giác lệch ngôn ngữ. Nhận mã ngôn ngữ như ``"ZH"``, ``"TH"``.
    """

    def __init__(
        self,
        model_id: str = "facebook/nllb-200-distilled-600M",
        force_wrong_target: str | None = None,
        model: Any = None,
        tokenizer: Any = None,
    ) -> None:
        self.model_id = model_id
        self.force_wrong_target = force_wrong_target
        self.name = model_id.split("/")[-1]
        if force_wrong_target:
            self.name += f"[off-target:{force_wrong_target}]"
        self._model = model
        self._tokenizer = tokenizer

    def _ensure_loaded(self, src_lang: str) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        code = NLLB_LANG[src_lang.upper()]
        # Tokenizer của NLLB gắn mã ngôn ngữ nguồn vào lúc khởi tạo, nên phải
        # nạp lại khi đổi chiều dịch.
        if self._tokenizer is None or getattr(self._tokenizer, "src_lang", None) != code:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, src_lang=code)

        if self._model is None:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
            self._model = self._model.to("cuda" if torch.cuda.is_available() else "cpu")
            self._model.eval()

    def translate(
        self,
        texts: list[str],
        src_lang: str = "EN",
        tgt_lang: str = "VI",
        decoding: DecodingConfig | None = None,
        batch_size: int = 16,
    ) -> list[str]:
        if not texts:
            return []
        import torch

        self._ensure_loaded(src_lang)
        decoding = decoding or DECODING_PRESETS[0]

        effective_target = self.force_wrong_target or tgt_lang
        bos = self._tokenizer.convert_tokens_to_ids(NLLB_LANG[effective_target.upper()])

        out: list[str] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self._tokenizer(batch, return_tensors="pt", padding=True,
                                  truncation=True, max_length=256).to(self._model.device)
            with torch.no_grad():
                gen = self._model.generate(
                    **enc, forced_bos_token_id=bos, **decoding.to_kwargs()
                )
            out.extend(self._tokenizer.batch_decode(gen, skip_special_tokens=True))
        return out


class EnViT5Translator(Translator):
    """VietAI/envit5-translation — hệ chuyên Việt–Anh, dùng làm đối chứng mạnh.

    Mô hình T5 này đòi hỏi tiền tố ngôn ngữ trong đầu vào (``"en: ..."``) và
    trả về đầu ra cũng kèm tiền tố (``"vi: ..."``), phải cắt bỏ.
    """

    def __init__(self, model_id: str = "VietAI/envit5-translation",
                 model: Any = None, tokenizer: Any = None) -> None:
        self.model_id = model_id
        self.name = model_id.split("/")[-1]
        self._model = model
        self._tokenizer = tokenizer

    def _ensure_loaded(self) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self._model is None:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
            self._model = self._model.to("cuda" if torch.cuda.is_available() else "cpu")
            self._model.eval()

    @staticmethod
    def add_prefix(text: str, src_lang: str) -> str:
        return f"{src_lang.lower()}: {text}"

    @staticmethod
    def strip_prefix(text: str) -> str:
        """Bỏ tiền tố ``vi:`` / ``en:`` mà mô hình thêm vào đầu ra."""
        stripped = text.strip()
        for prefix in ("vi:", "en:", "vi :", "en :"):
            if stripped.lower().startswith(prefix):
                return stripped[len(prefix):].strip()
        return stripped

    def translate(
        self,
        texts: list[str],
        src_lang: str = "EN",
        tgt_lang: str = "VI",
        decoding: DecodingConfig | None = None,
        batch_size: int = 16,
    ) -> list[str]:
        if not texts:
            return []
        import torch

        self._ensure_loaded()
        decoding = decoding or DECODING_PRESETS[0]
        prefixed = [self.add_prefix(t, src_lang) for t in texts]

        out: list[str] = []
        for i in range(0, len(prefixed), batch_size):
            batch = prefixed[i : i + batch_size]
            enc = self._tokenizer(batch, return_tensors="pt", padding=True,
                                  truncation=True, max_length=256).to(self._model.device)
            with torch.no_grad():
                gen = self._model.generate(**enc, **decoding.to_kwargs())
            decoded = self._tokenizer.batch_decode(gen, skip_special_tokens=True)
            out.extend(self.strip_prefix(t) for t in decoded)
        return out


class LLMTranslator(Translator):
    """Dịch bằng LLM sinh văn bản — đúng bối cảnh của P2.

    P2 nghiên cứu ảo giác của **hệ dịch nền LLM** (ALMA-7B-R), khác hẳn ảo giác
    của encoder-decoder truyền thống. Có hệ này trong ngữ liệu thì mới trả lời
    được câu hỏi liệu detector có bắt được cả hai kiểu hay không.
    """

    def __init__(self, model_id: str = "Qwen/Qwen2.5-7B-Instruct",
                 load_in_4bit: bool = True, model: Any = None,
                 tokenizer: Any = None) -> None:
        self.model_id = model_id
        self.load_in_4bit = load_in_4bit
        self.name = model_id.split("/")[-1]
        self._model = model
        self._tokenizer = tokenizer

    @staticmethod
    def build_prompt(text: str, src_name: str, tgt_name: str) -> list[dict[str, str]]:
        return [
            {"role": "system",
             "content": f"You are a professional translator. Translate the user's "
                        f"{src_name} text into {tgt_name}. Output only the translation, "
                        f"with no explanation or extra text."},
            {"role": "user", "content": text},
        ]

    def _ensure_loaded(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

        if self._model is None:
            kwargs: dict[str, Any] = {"dtype": "auto"}
            if self.load_in_4bit and torch.cuda.is_available():
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                )
            self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
            if "quantization_config" not in kwargs:
                self._model = self._model.to(
                    "cuda" if torch.cuda.is_available() else "cpu")
            self._model.eval()

    def translate(
        self,
        texts: list[str],
        src_lang: str = "EN",
        tgt_lang: str = "VI",
        decoding: DecodingConfig | None = None,
        batch_size: int = 8,
    ) -> list[str]:
        if not texts:
            return []
        import torch

        self._ensure_loaded()
        decoding = decoding or DECODING_PRESETS[0]
        src_name = LANGUAGE_FULL_NAME[src_lang.upper()]
        tgt_name = LANGUAGE_FULL_NAME[tgt_lang.upper()]

        rendered = [
            self._tokenizer.apply_chat_template(
                self.build_prompt(t, src_name, tgt_name),
                tokenize=False, add_generation_prompt=True,
            )
            for t in texts
        ]

        out: list[str] = []
        for i in range(0, len(rendered), batch_size):
            enc = self._tokenizer(rendered[i : i + batch_size], return_tensors="pt",
                                  padding=True, add_special_tokens=False
                                  ).to(self._model.device)
            with torch.no_grad():
                gen = self._model.generate(
                    **enc, pad_token_id=self._tokenizer.pad_token_id,
                    **decoding.to_kwargs(),
                )
            new = gen[:, enc["input_ids"].shape[1]:]
            out.extend(t.strip() for t in
                       self._tokenizer.batch_decode(new, skip_special_tokens=True))
        return out


LANGUAGE_FULL_NAME: dict[str, str] = {
    "EN": "English", "VI": "Vietnamese", "ZH": "Chinese",
    "TH": "Thai", "ID": "Indonesian",
}


#: Các hệ dịch dùng được, kèm tình trạng đã kiểm chứng.
#:
#: CẢNH BÁO VỀ PHIÊN BẢN: `envit5` và `vinai` dựa trên tokenizer sentencepiece
#: và **vỡ với transformers >= 5.0** (lỗi `'dict' object cannot be converted to
#: Sequence`, và lỗi nhận nhầm tệp tiktoken). Chúng chạy tốt trên transformers
#: 4.4x — tức là trên Colab/Kaggle hiện nay. Vì không kiểm chứng được ở mọi môi
#: trường nên chúng KHÔNG nằm trong kế hoạch mặc định; muốn dùng thì bật tay và
#: tự kiểm tra trước.
TRANSLATOR_REGISTRY: dict[str, dict[str, Any]] = {
    "nllb600m": {
        "factory": lambda **kw: NLLBTranslator("facebook/nllb-200-distilled-600M", **kw),
        "verified": True,
        "note": "Encoder-decoder nho, hay ao giac — nguon mau duong chinh",
    },
    "nllb1.3b": {
        "factory": lambda **kw: NLLBTranslator("facebook/nllb-200-distilled-1.3B", **kw),
        "verified": True,
        "note": "Ban lon hon, it ao giac hon — he doi chung manh",
    },
    "qwen7b": {
        "factory": lambda **kw: LLMTranslator("Qwen/Qwen2.5-7B-Instruct"),
        "verified": True,
        "note": "Dich bang LLM — tai hien boi canh cua P2",
    },
    "envit5": {
        "factory": lambda **kw: EnViT5Translator(),
        "verified": False,
        "note": "Chuyen Viet-Anh. CAN transformers < 5.0",
    },
}


def make_translator(name: str, **kwargs: Any) -> Translator:
    """Khởi tạo hệ dịch theo tên trong registry."""
    if name not in TRANSLATOR_REGISTRY:
        raise KeyError(f"He dich khong ton tai: {name!r}. Cac lua chon: "
                       f"{sorted(TRANSLATOR_REGISTRY)}")
    return TRANSLATOR_REGISTRY[name]["factory"](**kwargs)


def probe_translator(name: str) -> tuple[bool, str]:
    """Thử nạp *tokenizer* của một hệ dịch để phát hiện sớm hệ không dùng được.

    Chỉ nạp tokenizer chứ không nạp trọng số: rẻ, nhanh, mà vẫn bắt được đúng
    loại lỗi hay gặp nhất (xung đột phiên bản tokenizer). Mục đích là **thất
    bại sớm**, trước khi tiêu hàng chục phút GPU rồi mới đổ.

    Returns:
        (dùng được, thông điệp giải thích).
    """
    from transformers import AutoTokenizer

    model_ids = {
        "nllb600m": "facebook/nllb-200-distilled-600M",
        "nllb1.3b": "facebook/nllb-200-distilled-1.3B",
        "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
        "envit5": "VietAI/envit5-translation",
    }
    if name not in model_ids:
        return False, f"khong co trong registry"

    try:
        AutoTokenizer.from_pretrained(model_ids[name])
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:90]}"


#: Kế hoạch sinh Tập A: cặp (hệ dịch, cấu hình giải mã) và tỉ trọng câu.
#: Phần lớn dùng giải mã lành để bộ dữ liệu vẫn phản ánh hành vi bình thường
#: của hệ dịch; phần nhỏ dùng giải mã hung hãn để bảo đảm có đủ mẫu dương.
@dataclass(frozen=True)
class GenerationSpec:
    """Một tổ hợp (hệ dịch, giải mã, ép sai ngôn ngữ) kèm tỉ trọng."""

    translator: str
    decoding: str
    share: float
    force_wrong_target: str | None = None

    @property
    def tag(self) -> str:
        off = f"+off{self.force_wrong_target}" if self.force_wrong_target else ""
        return f"{self.translator}/{self.decoding}{off}"


#: Kế hoạch mặc định — chỉ dùng hệ dịch đã kiểm chứng trên transformers 5.x.
DEFAULT_GENERATION_PLAN: tuple[GenerationSpec, ...] = (
    GenerationSpec("nllb600m", "beam5", 0.22),
    GenerationSpec("nllb600m", "greedy", 0.14),
    GenerationSpec("nllb600m", "sample_t1.2", 0.12),
    GenerationSpec("nllb600m", "sample_t1.8", 0.10),
    GenerationSpec("nllb600m", "epsilon", 0.06),
    GenerationSpec("nllb600m", "greedy", 0.04, force_wrong_target="ZH"),
    GenerationSpec("nllb1.3b", "beam5", 0.14),
    GenerationSpec("nllb1.3b", "sample_t1.8", 0.06),
    GenerationSpec("qwen7b", "beam5", 0.08),
    GenerationSpec("qwen7b", "sample_t1.2", 0.04),
)

#: Kế hoạch tối giản để kiểm tra đường ống — một mô hình nhỏ, giải mã nhanh.
#:
#: **Chỉ dùng giải mã tất định.** Sampling sinh tới `max_new_tokens` mà không
#: dừng sớm được, nên trên CPU một câu mất hàng chục giây — kế hoạch smoke mà
#: không chạy xong nổi trên CPU thì vô dụng. Nhánh sampling vẫn được kiểm thử
#: riêng ở tầng đơn vị (`DecodingConfig.to_kwargs`), và chạy thật trên GPU.
SMOKE_GENERATION_PLAN: tuple[GenerationSpec, ...] = (
    GenerationSpec("nllb600m", "greedy", 0.6),
    GenerationSpec("nllb600m", "beam5", 0.2),
    GenerationSpec("nllb600m", "greedy", 0.2, force_wrong_target="ZH"),
)


def validate_plan(plan: Iterable[GenerationSpec]) -> None:
    """Kiểm tra tỉ trọng cộng lại bằng 1 và mọi tên tham chiếu đều hợp lệ."""
    plan = list(plan)
    total = sum(s.share for s in plan)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Tong ti trong phai bang 1, dang la {total}")

    known_decodings = {d.name for d in DECODING_PRESETS}
    for spec in plan:
        if spec.decoding not in known_decodings:
            raise ValueError(f"Cau hinh giai ma khong ton tai: {spec.decoding!r}")
        if spec.translator not in TRANSLATOR_REGISTRY:
            raise ValueError(f"He dich khong ton tai: {spec.translator!r}")
        if spec.force_wrong_target and spec.force_wrong_target.upper() not in NLLB_LANG:
            raise ValueError(f"Ngon ngu khong ho tro: {spec.force_wrong_target!r}")


def redistribute_shares(
    plan: Iterable[GenerationSpec],
    drop: set[str],
) -> tuple[GenerationSpec, ...]:
    """Bỏ các hệ dịch hỏng và chia lại tỉ trọng cho các hệ còn lại.

    Dùng khi một mô hình không nạp được giữa chừng. Trên Kaggle, một lượt chạy
    chết sau 20 phút GPU là mất trắng một chu kỳ làm việc, nên thà chạy tiếp
    với ít hệ dịch hơn còn hơn dừng hẳn — miễn là ghi rõ đã bỏ cái gì.
    """
    kept = [s for s in plan if s.translator not in drop]
    if not kept:
        raise RuntimeError("Moi he dich deu that bai — khong the tiep tuc.")

    total = sum(s.share for s in kept)
    return tuple(
        GenerationSpec(s.translator, s.decoding, s.share / total, s.force_wrong_target)
        for s in kept
    )


def get_decoding(name: str) -> DecodingConfig:
    """Tra cấu hình giải mã theo tên."""
    for d in DECODING_PRESETS:
        if d.name == name:
            return d
    raise KeyError(f"Khong co cau hinh giai ma {name!r}. Cac lua chon: "
                   f"{[d.name for d in DECODING_PRESETS]}")
