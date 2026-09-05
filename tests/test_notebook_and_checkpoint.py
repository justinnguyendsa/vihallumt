"""Kiểm thử notebook Kaggle và cơ chế checkpoint.

Hai lớp lỗi mà bộ test cũ không bắt được, và cả hai đều đã thật sự làm hỏng
một lượt chạy GPU trên Kaggle:

1. **Notebook thiếu gói phụ thuộc.** `requirements.txt` có `bitsandbytes`
   nhưng ô cài đặt trong notebook thì không. Lỗi chỉ nổ ra sau ~30 phút GPU,
   ngay lúc nạp mô hình LLM đầu tiên.
2. **Không có checkpoint.** Script chỉ ghi kết quả ở cuối, nên một lỗi ở phút
   thứ 35 làm mất trắng toàn bộ công đã làm.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "01_build_corpus_kaggle.ipynb"


@pytest.fixture(scope="module")
def notebook() -> dict:
    if not NOTEBOOK.exists():
        pytest.skip("Chua sinh notebook — chay scripts/make_kaggle_notebook.py")
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def notebook_source(notebook) -> str:
    return "\n".join("".join(c["source"]) for c in notebook["cells"])


# ==========================================================================
# Notebook hợp lệ
# ==========================================================================

def test_notebook_is_valid_json_and_nbformat(notebook):
    assert notebook["nbformat"] == 4
    assert notebook["cells"]


def test_every_code_cell_parses(notebook):
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("\n".join(cell["source"]), filename=f"cell-{i}")


def test_notebook_requests_gpu(notebook):
    assert notebook["metadata"].get("accelerator") == "GPU"


# ==========================================================================
# Gói phụ thuộc — lỗi đã làm hỏng lượt chạy Kaggle
# ==========================================================================

#: Gói mà mã nguồn thật sự cần lúc chạy, kèm lý do.
REQUIRED_PACKAGES = {
    "bitsandbytes": "luong tu hoa 4-bit cho LLMTranslator (Qwen-7B)",
    "transformers": "moi mo hinh",
    "accelerate": "device_map khi nap mo hinh",
    "sentence-transformers": "cham diem LaBSE",
    "datasets": "nap OPUS-100 / IWSLT",
    "sentencepiece": "tokenizer cua NLLB",
}


@pytest.mark.parametrize("package,reason", sorted(REQUIRED_PACKAGES.items()))
def test_notebook_installs_every_required_package(notebook_source, package, reason):
    """Mọi gói mã nguồn cần phải có trong ô cài đặt của notebook.

    Đây chính là test đáng lẽ phải bắt được lỗi `bitsandbytes`: gói có trong
    `requirements.txt` nhưng notebook không cài, nên chỉ vỡ trên Kaggle.
    """
    assert package in notebook_source, (
        f"Notebook khong cai {package!r} (can cho: {reason}). "
        f"Sua trong scripts/make_kaggle_notebook.py roi chay lai script do."
    )


def test_required_packages_are_also_in_requirements_txt():
    """Hai nơi khai báo phụ thuộc phải khớp nhau."""
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    for package in REQUIRED_PACKAGES:
        assert package in req, f"{package} thieu trong requirements.txt"


def test_notebook_verifies_imports_before_the_expensive_run(notebook_source):
    """Notebook phải kiểm tra import ĐƯỢC, không chỉ tin vào việc pip báo xong.

    pip có thể báo thành công mà gói vẫn không import được (sai kiến trúc,
    xung đột phiên bản). Phải thử import thật, và thử TRƯỚC bước tốn GPU.
    """
    assert "importlib.import_module" in notebook_source
    install_at = notebook_source.index("pip install")
    check_at = notebook_source.index("importlib.import_module")
    run_at = notebook_source.index("build_corpus_a.py")
    assert install_at < check_at < run_at, "Kiem tra import phai nam giua cai dat va luot chay"


def test_notebook_uses_the_real_repo_url(notebook_source):
    assert "justinnguyendsa/vihallumt" in notebook_source
    assert "<tai-khoan>" not in notebook_source


# ==========================================================================
# Checkpoint
# ==========================================================================

@pytest.fixture(scope="module")
def build_module():
    """Nạp `scripts/build_corpus_a.py` như một module."""
    import importlib.util
    import sys

    sys.argv = ["build_corpus_a.py"]
    spec = importlib.util.spec_from_file_location(
        "build_corpus_a", ROOT / "scripts" / "build_corpus_a.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_checkpoint_directory_is_defined(build_module):
    assert hasattr(build_module, "CKPT")
    assert build_module.CKPT.name == "_checkpoints"


def test_translate_pool_accepts_checkpoint_dir(build_module):
    import inspect

    sig = inspect.signature(build_module.translate_pool)
    assert "ckpt_dir" in sig.parameters


def test_fresh_flag_exists(build_module):
    """Phải có cách xoá checkpoint, nếu không thì kẹt với kết quả cũ."""
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf), pytest.raises(SystemExit):
        import sys
        sys.argv = ["x", "--help"]
        build_module.parse_args()
    assert "--fresh" in buf.getvalue()


class _FakeTranslator:
    """Hệ dịch giả, đếm số lần thật sự được gọi."""

    def __init__(self) -> None:
        self.calls = 0
        self.unloaded = 0

    def translate(self, texts, src_lang="EN", tgt_lang="VI", decoding=None,
                  batch_size=16, force_wrong_target=None):
        self.calls += 1
        return [f"dich: {t}" for t in texts]

    def unload(self):
        self.unloaded += 1


def test_checkpoint_prevents_retranslation(build_module, tmp_path, monkeypatch):
    """Chạy lần hai phải dùng lại checkpoint, KHÔNG gọi lại hệ dịch.

    Đây là điều quyết định: một lượt chạy hỏng ở phút 35 rồi chạy lại không
    được phép dịch lại từ đầu.
    """
    from vihallumt.corpus.translate import GenerationSpec

    fake = _FakeTranslator()
    monkeypatch.setattr(build_module, "build_translators",
                        lambda specs: {"nllb600m": fake})

    df = pd.DataFrame({
        "src_text": [f"source sentence number {i}" for i in range(10)],
        "src": ["EN"] * 10, "tgt": ["VI"] * 10,
    })
    plan = (GenerationSpec("nllb600m", "greedy", 1.0),)

    first = build_module.translate_pool(df, "EN", "VI", 4, 42, plan, ckpt_dir=tmp_path)
    assert fake.calls == 1
    assert (first["mt_text"] != "").all()

    second = build_module.translate_pool(df, "EN", "VI", 4, 42, plan, ckpt_dir=tmp_path)
    assert fake.calls == 1, "da goi lai he dich du da co checkpoint"
    assert second["mt_text"].tolist() == first["mt_text"].tolist()


def test_checkpoint_files_are_written(build_module, tmp_path, monkeypatch):
    from vihallumt.corpus.translate import GenerationSpec

    monkeypatch.setattr(build_module, "build_translators",
                        lambda specs: {"nllb600m": _FakeTranslator()})
    df = pd.DataFrame({"src_text": [f"s{i}" for i in range(6)],
                       "src": ["EN"] * 6, "tgt": ["VI"] * 6})
    build_module.translate_pool(df, "EN", "VI", 4, 42,
                                (GenerationSpec("nllb600m", "greedy", 1.0),),
                                ckpt_dir=tmp_path)
    files = list(tmp_path.glob("*.jsonl"))
    assert files, "khong ghi checkpoint nao"
    assert "EN-VI" in files[0].name


def test_corrupt_checkpoint_triggers_retranslation(build_module, tmp_path, monkeypatch):
    """Checkpoint sai kích thước phải bị bỏ và dịch lại, không được dùng bừa."""
    from vihallumt.corpus.translate import GenerationSpec

    fake = _FakeTranslator()
    monkeypatch.setattr(build_module, "build_translators",
                        lambda specs: {"nllb600m": fake})
    df = pd.DataFrame({"src_text": [f"s{i}" for i in range(8)],
                       "src": ["EN"] * 8, "tgt": ["VI"] * 8})
    plan = (GenerationSpec("nllb600m", "greedy", 1.0),)

    build_module.translate_pool(df, "EN", "VI", 4, 42, plan, ckpt_dir=tmp_path)
    assert fake.calls == 1

    # Cắt bớt checkpoint để giả lập lần ghi bị dở dang
    ckpt = next(tmp_path.glob("*.jsonl"))
    rows = ckpt.read_text(encoding="utf-8").strip().split("\n")
    ckpt.write_text("\n".join(rows[:3]), encoding="utf-8")

    out = build_module.translate_pool(df, "EN", "VI", 4, 42, plan, ckpt_dir=tmp_path)
    assert fake.calls == 2, "checkpoint hong ma van dung bua"
    assert (out["mt_text"] != "").all()


def test_no_checkpoint_dir_still_works(build_module, monkeypatch):
    """Không truyền ckpt_dir thì vẫn chạy bình thường."""
    from vihallumt.corpus.translate import GenerationSpec

    monkeypatch.setattr(build_module, "build_translators",
                        lambda specs: {"nllb600m": _FakeTranslator()})
    df = pd.DataFrame({"src_text": ["a b c", "d e f"],
                       "src": ["EN"] * 2, "tgt": ["VI"] * 2})
    out = build_module.translate_pool(df, "EN", "VI", 4, 42,
                                      (GenerationSpec("nllb600m", "greedy", 1.0),))
    assert (out["mt_text"] != "").all()


# ==========================================================================
# Suy giảm mềm khi thiếu bitsandbytes
# ==========================================================================

def test_bitsandbytes_probe_exists_and_returns_bool():
    from vihallumt.corpus.translate import _bitsandbytes_available

    assert isinstance(_bitsandbytes_available(), bool)


def test_importing_bnb_config_is_not_proof_of_availability():
    """`BitsAndBytesConfig` import được ngay cả khi chưa cài bitsandbytes.

    Chính vì thế mà lỗi chỉ nổ ra tận lúc `from_pretrained`, sau khi đã tiêu
    hàng chục phút GPU. Test này chốt lại lý do phải kiểm tra bằng cách khác.
    """
    from transformers import BitsAndBytesConfig  # noqa: F401

    from vihallumt.corpus.translate import _bitsandbytes_available

    # Import thành công không nói lên điều gì về việc bitsandbytes có sẵn hay không
    assert BitsAndBytesConfig is not None
    assert isinstance(_bitsandbytes_available(), bool)


# ==========================================================================
# Quản lý bộ nhớ GPU — lỗi đã gây tràn bộ nhớ trên Kaggle
# ==========================================================================

class _MemoryTrackingTranslator:
    """Hệ dịch giả, mô phỏng việc chiếm và giải phóng bộ nhớ GPU."""

    def __init__(self, name: str, gb: float) -> None:
        self.name = name
        self.gb = gb
        self.loaded = False
        self.load_count = 0
        self.unload_count = 0
        self.targets_seen: list[str | None] = []

    def translate(self, texts, src_lang="EN", tgt_lang="VI", decoding=None,
                  batch_size=16, force_wrong_target=None):
        if not self.loaded:
            self.loaded = True
            self.load_count += 1
        self.targets_seen.append(force_wrong_target)
        return [f"{self.name}: {t}" for t in texts]

    def unload(self):
        self.loaded = False
        self.unload_count += 1


def test_model_is_unloaded_after_its_groups_finish(build_module, tmp_path, monkeypatch):
    """Mô hình phải được giải phóng ngay khi dùng xong.

    Nếu không, bộ nhớ chiếm dụng là TỔNG mọi mô hình chứ không phải mô hình
    lớn nhất — đúng cách làm tràn 15 GB GPU của Kaggle dù từng mô hình một
    đều thừa sức vừa.
    """
    from vihallumt.corpus.translate import GenerationSpec

    models = {
        "nllb600m": _MemoryTrackingTranslator("nllb600m", 1.2),
        "nllb1.3b": _MemoryTrackingTranslator("nllb1.3b", 2.6),
        "qwen7b": _MemoryTrackingTranslator("qwen7b", 4.5),
    }
    monkeypatch.setattr(build_module, "build_translators", lambda specs: models)

    df = pd.DataFrame({"src_text": [f"s{i}" for i in range(30)],
                       "src": ["EN"] * 30, "tgt": ["VI"] * 30})
    plan = (
        GenerationSpec("nllb600m", "greedy", 0.4),
        GenerationSpec("nllb1.3b", "beam5", 0.3),
        GenerationSpec("qwen7b", "beam5", 0.3),
    )
    build_module.translate_pool(df, "EN", "VI", 8, 42, plan, ckpt_dir=tmp_path)

    for name, m in models.items():
        assert m.unload_count == 1, f"{name} khong duoc giai phong"
        assert not m.loaded, f"{name} van con trong bo nho sau khi xong"


def test_only_one_model_is_loaded_at_a_time(build_module, tmp_path, monkeypatch):
    """Đỉnh bộ nhớ phải là mô hình lớn nhất, không phải tổng."""
    from vihallumt.corpus.translate import GenerationSpec

    models = {n: _MemoryTrackingTranslator(n, gb)
              for n, gb in [("nllb600m", 1.2), ("nllb1.3b", 2.6), ("qwen7b", 4.5)]}
    peak = {"gb": 0.0}

    original_translate = _MemoryTrackingTranslator.translate

    def tracking_translate(self, *a, **kw):
        out = original_translate(self, *a, **kw)
        peak["gb"] = max(peak["gb"], sum(m.gb for m in models.values() if m.loaded))
        return out

    monkeypatch.setattr(_MemoryTrackingTranslator, "translate", tracking_translate)
    monkeypatch.setattr(build_module, "build_translators", lambda specs: models)

    df = pd.DataFrame({"src_text": [f"s{i}" for i in range(30)],
                       "src": ["EN"] * 30, "tgt": ["VI"] * 30})
    plan = (GenerationSpec("nllb600m", "greedy", 0.4),
            GenerationSpec("nllb1.3b", "beam5", 0.3),
            GenerationSpec("qwen7b", "beam5", 0.3))
    build_module.translate_pool(df, "EN", "VI", 8, 42, plan, ckpt_dir=tmp_path)

    assert peak["gb"] <= 4.5, (
        f"dinh bo nho {peak['gb']} GB — dang giu nhieu mo hinh cung luc "
        f"(tong cua ca ba la 8.3 GB)"
    )


def test_off_target_shares_weights_with_normal_branch(build_module, tmp_path, monkeypatch):
    """Nhánh ép sai ngôn ngữ đích KHÔNG được nạp bản sao trọng số riêng.

    Ép sai ngôn ngữ chỉ đổi token BOS lúc sinh; trọng số y hệt. Tách thành hai
    đối tượng khiến GPU giữ hai bản sao của cùng một mô hình 600M.
    """
    from vihallumt.corpus.translate import GenerationSpec

    model = _MemoryTrackingTranslator("nllb600m", 1.2)
    monkeypatch.setattr(build_module, "build_translators",
                        lambda specs: {"nllb600m": model})

    df = pd.DataFrame({"src_text": [f"s{i}" for i in range(20)],
                       "src": ["EN"] * 20, "tgt": ["VI"] * 20})
    plan = (GenerationSpec("nllb600m", "greedy", 0.5),
            GenerationSpec("nllb600m", "greedy", 0.5, force_wrong_target="ZH"))
    build_module.translate_pool(df, "EN", "VI", 8, 42, plan, ckpt_dir=tmp_path)

    assert model.load_count == 1, "nap trong so nhieu hon mot lan"
    # Ca hai nhanh deu chay, va nhanh off-target nhan dung tham so
    assert None in model.targets_seen
    assert "ZH" in model.targets_seen


def test_build_translators_keys_by_model_name_only(build_module):
    """Khoá theo tên mô hình, không tách theo force_wrong_target."""
    from vihallumt.corpus.translate import GenerationSpec

    specs = [GenerationSpec("nllb600m", "greedy", 0.5),
             GenerationSpec("nllb600m", "greedy", 0.5, force_wrong_target="ZH")]
    out = build_module.build_translators(specs)
    assert set(out) == {"nllb600m"}, f"tao {len(out)} doi tuong thay vi 1: {set(out)}"


def test_translators_load_in_half_precision_on_gpu():
    """fp32 la mac dinh cua from_pretrained va ngon gap doi bo nho can thiet."""
    import torch

    from vihallumt.corpus.translate import preferred_dtype

    expected = torch.float16 if torch.cuda.is_available() else torch.float32
    assert preferred_dtype() is expected


def test_every_translator_implements_unload():
    from vihallumt.corpus.translate import (
        EnViT5Translator,
        LLMTranslator,
        NLLBTranslator,
    )

    for cls in (NLLBTranslator, EnViT5Translator, LLMTranslator):
        assert hasattr(cls, "unload")


def test_every_translator_accepts_force_wrong_target():
    """Chữ ký phải đồng nhất để `translate_pool` gọi được cho mọi hệ dịch."""
    import inspect

    from vihallumt.corpus.translate import (
        EnViT5Translator,
        LLMTranslator,
        NLLBTranslator,
    )

    for cls in (NLLBTranslator, EnViT5Translator, LLMTranslator):
        params = inspect.signature(cls.translate).parameters
        assert "force_wrong_target" in params, f"{cls.__name__} thieu tham so"
