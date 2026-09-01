"""Kiểm thử phần nạp nguồn và bộ bọc hệ dịch (không cần mạng, không cần GPU)."""

from __future__ import annotations

import pytest

from vihallumt.corpus.sources import (
    MAX_WORDS,
    MIN_WORDS,
    clean_pairs,
    is_usable,
    normalise,
)
from vihallumt.corpus.translate import (
    DECODING_PRESETS,
    DEFAULT_GENERATION_PLAN,
    NLLB_LANG,
    DecodingConfig,
    EnViT5Translator,
    GenerationSpec,
    get_decoding,
    validate_plan,
)


# ==========================================================================
# sources.normalise
# ==========================================================================

def test_normalise_collapses_whitespace():
    assert normalise("  hai    từ  \n cách nhau ") == "hai từ cách nhau"


def test_normalise_unifies_vietnamese_unicode_forms():
    """Tiếng Việt có hai cách mã hoá cùng một chữ: dựng sẵn và tổ hợp.

    Không chuẩn hoá thì hai chuỗi trông y hệt trên màn hình lại không bằng
    nhau, làm hỏng cả khử trùng lặp lẫn tách token.
    """
    precomposed = "tiếng Việt"            # U+1EBF
    decomposed = "tiếng Việt"  # e + ^ + ´
    assert precomposed != decomposed
    assert normalise(precomposed) == normalise(decomposed)


def test_normalise_handles_non_string_input():
    assert normalise(123) == "123"


# ==========================================================================
# sources.is_usable
# ==========================================================================

EN_OK = "The meeting will start at nine in the morning tomorrow."
VI_OK = "Cuộc họp sẽ bắt đầu lúc chín giờ sáng ngày mai."


def test_accepts_a_reasonable_pair():
    assert is_usable(EN_OK, VI_OK)


def test_rejects_empty_side():
    assert not is_usable("", VI_OK)
    assert not is_usable(EN_OK, "")


def test_rejects_too_short():
    assert not is_usable("Hello there.", "Xin chào bạn.")


def test_rejects_too_long():
    long_en = " ".join(["word"] * (MAX_WORDS + 5))
    long_vi = " ".join(["từ"] * (MAX_WORDS + 5))
    assert not is_usable(long_en, long_vi)


def test_accepts_at_length_boundaries():
    en = " ".join(["word"] * MIN_WORDS)
    vi = " ".join(["từ"] * MIN_WORDS)
    assert is_usable(en, vi)


def test_rejects_extreme_length_mismatch():
    """Lệch độ dài lớn thường là lỗi căn chỉnh, không phải bản dịch."""
    long_vi = " ".join(["từ"] * 40)
    assert not is_usable(EN_OK, long_vi)


def test_rejects_identical_sides():
    """Hai phía giống hệt nhau: gần như chắc chắn là lỗi căn chỉnh."""
    text = "This is a sentence that was never translated."
    assert not is_usable(text, text)


# ==========================================================================
# sources.clean_pairs
# ==========================================================================

def test_clean_pairs_filters_and_reports():
    pairs = [
        (EN_OK, VI_OK),
        ("", ""),                    # rỗng
        ("Hi.", "Chào."),            # quá ngắn
        (EN_OK, VI_OK),              # trùng
        ("She works at a big hospital downtown.", "Cô ấy làm việc ở một bệnh viện lớn."),
    ]
    df, stats = clean_pairs(pairs, "test")
    assert stats.n_raw == 5
    assert len(df) == 2
    assert stats.n_after_filter == 2
    assert 0 < stats.kept_pct < 100
    assert (df["source"] == "test").all()


def test_clean_pairs_deduplicates_by_english_side():
    """Cùng câu nguồn xuất hiện nhiều lần sẽ làm lệch lấy mẫu và gây rò rỉ
    giữa tập dev và test."""
    pairs = [
        (EN_OK, VI_OK),
        (EN_OK, "Một bản dịch khác cho cùng câu nguồn đó."),
    ]
    df, _ = clean_pairs(pairs, "test")
    assert len(df) == 1


def test_clean_pairs_on_empty_input():
    df, stats = clean_pairs([], "test")
    assert len(df) == 0
    assert stats.n_raw == 0
    assert stats.kept_pct == 0.0


# ==========================================================================
# translate.DecodingConfig
# ==========================================================================

def test_greedy_config_omits_sampling_parameters():
    kw = DecodingConfig("greedy").to_kwargs()
    assert kw["do_sample"] is False
    assert "temperature" not in kw
    assert "epsilon_cutoff" not in kw


def test_sampling_config_includes_temperature():
    kw = DecodingConfig("s", do_sample=True, temperature=1.5).to_kwargs()
    assert kw["temperature"] == 1.5


def test_epsilon_config_matches_paper2_value():
    """P2 §5.1: epsilon sampling với ε = 0.02 là chiến lược tốt nhất của họ."""
    eps = get_decoding("epsilon")
    assert eps.epsilon_cutoff == 0.02
    assert "epsilon_cutoff" in eps.to_kwargs()


def test_epsilon_ignored_when_not_sampling():
    kw = DecodingConfig("x", do_sample=False, epsilon_cutoff=0.02).to_kwargs()
    assert "epsilon_cutoff" not in kw


def test_all_presets_have_unique_names():
    names = [d.name for d in DECODING_PRESETS]
    assert len(names) == len(set(names))


def test_preset_lookup():
    assert get_decoding("beam5").num_beams == 5
    with pytest.raises(KeyError):
        get_decoding("khong_ton_tai")


# ==========================================================================
# translate.GenerationSpec / plan
# ==========================================================================

def test_default_plan_is_valid():
    validate_plan(DEFAULT_GENERATION_PLAN)


def test_default_plan_shares_sum_to_one():
    assert sum(s.share for s in DEFAULT_GENERATION_PLAN) == pytest.approx(1.0)


def test_default_plan_favours_benign_decoding():
    """Đa số câu phải dùng giải mã lành, để bộ dữ liệu vẫn phản ánh hành vi
    bình thường của hệ dịch chứ không chỉ toàn ca cực đoan."""
    benign = sum(s.share for s in DEFAULT_GENERATION_PLAN
                 if s.decoding in ("beam5", "greedy") and not s.force_wrong_target)
    assert benign > 0.5


def test_default_plan_uses_more_than_one_translator():
    assert len({s.translator for s in DEFAULT_GENERATION_PLAN}) >= 2


def test_default_plan_includes_off_target_generation():
    assert any(s.force_wrong_target for s in DEFAULT_GENERATION_PLAN)


def test_plan_validation_rejects_bad_shares():
    with pytest.raises(ValueError, match="Tong ti trong"):
        validate_plan([GenerationSpec("nllb600m", "greedy", 0.5)])


def test_plan_validation_rejects_unknown_decoding():
    with pytest.raises(ValueError, match="Cau hinh giai ma khong ton tai"):
        validate_plan([GenerationSpec("nllb600m", "khong_ton_tai", 1.0)])


def test_plan_validation_rejects_unknown_language():
    with pytest.raises(ValueError, match="Ngon ngu khong ho tro"):
        validate_plan([GenerationSpec("nllb600m", "greedy", 1.0, force_wrong_target="XX")])


def test_spec_tag_is_descriptive():
    assert GenerationSpec("nllb600m", "greedy", 0.1).tag == "nllb600m/greedy"
    assert GenerationSpec("nllb600m", "greedy", 0.1, "ZH").tag == "nllb600m/greedy+offZH"


def test_all_plan_tags_unique_per_translator_decoding_combo():
    tags = [s.tag for s in DEFAULT_GENERATION_PLAN]
    assert len(tags) == len(set(tags))


# ==========================================================================
# translate — xử lý tiền tố của envit5
# ==========================================================================

def test_envit5_adds_source_language_prefix():
    assert EnViT5Translator.add_prefix("Hello world.", "EN") == "en: Hello world."
    assert EnViT5Translator.add_prefix("Xin chào.", "VI") == "vi: Xin chào."


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("vi: Xin chào bạn.", "Xin chào bạn."),
        ("en: Hello there.", "Hello there."),
        ("  vi:   Xin chào.  ", "Xin chào."),
        ("vi : Xin chào.", "Xin chào."),
        ("Xin chào bạn.", "Xin chào bạn."),   # không có tiền tố
    ],
)
def test_envit5_strips_output_prefix(raw, expected):
    assert EnViT5Translator.strip_prefix(raw) == expected


def test_envit5_does_not_strip_content_that_merely_starts_with_letters():
    """Không được cắt nhầm nội dung thật chỉ vì trùng chữ cái đầu."""
    assert EnViT5Translator.strip_prefix("view of the city") == "view of the city"


# ==========================================================================
# translate — mã ngôn ngữ NLLB
# ==========================================================================

def test_nllb_language_codes_are_flores_style():
    assert NLLB_LANG["VI"] == "vie_Latn"
    assert NLLB_LANG["EN"] == "eng_Latn"


def test_off_target_languages_available():
    for code in ("ZH", "TH", "ID"):
        assert code in NLLB_LANG


def test_empty_input_returns_empty_without_loading_model():
    """Không được nạp mô hình khi đầu vào rỗng — nếu nạp thì test này sẽ treo."""
    assert EnViT5Translator().translate([]) == []
