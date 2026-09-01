"""Kiểm thử module prompt — đảm bảo chép đúng nguyên văn P1."""

from __future__ import annotations

import pytest

from vihallumt.prompts.binary import (
    COT1_EN,
    COT2_EN,
    LABELS,
    PROMPT_GRID,
    build_prompt,
    variant_name,
)

SRC = "The cat sat on the mat."
MT = "Con mèo ngồi trên tấm thảm."


# --------------------------------------------------------------------------
# Dựng prompt cơ bản
# --------------------------------------------------------------------------

@pytest.mark.parametrize("prompt_id", ["p1", "p2", "p3"])
@pytest.mark.parametrize("language", ["en", "vi"])
def test_every_prompt_variant_builds(prompt_id, language):
    p = build_prompt(SRC, MT, "EN", "VI", prompt_id=prompt_id, language=language)
    assert p.system.strip()
    assert p.user.strip()
    assert SRC in p.user
    assert MT in p.user


def test_prompt2_names_both_languages():
    """Figure 11 của P1 là biến thể 'có nói rõ tên ngôn ngữ'."""
    p = build_prompt(SRC, MT, "EN", "VI", prompt_id="p2")
    assert "English" in p.system
    assert "Vietnamese" in p.system


def test_prompt1_does_not_name_languages():
    """Figure 10 lấy nguyên từ G-Eval, không nhắc tên ngôn ngữ."""
    p = build_prompt(SRC, MT, "EN", "VI", prompt_id="p1")
    assert "English" not in p.system
    assert "Vietnamese" not in p.system


def test_prompt3_defines_hallucination_explicitly():
    """Figure 12 là prompt người thiết kế, có định nghĩa tường minh."""
    p = build_prompt(SRC, MT, prompt_id="p3")
    assert "Definition of Hallucination" in p.system


def test_prompt3_uses_list_style_answer_format():
    p1 = build_prompt(SRC, MT, prompt_id="p1")
    p3 = build_prompt(SRC, MT, prompt_id="p3")
    assert "Answer (label ONLY" in p1.user
    assert "Provide exactly one of the following" in p3.user


def test_vietnamese_prompt_uses_vietnamese_language_names():
    p = build_prompt(SRC, MT, "EN", "VI", prompt_id="p2", language="vi")
    assert "tiếng Anh" in p.system
    assert "tiếng Việt" in p.system


# --------------------------------------------------------------------------
# Chain of Thought
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cot,expected", [("cot1", COT1_EN), ("cot2", COT2_EN)])
def test_cot_is_appended_to_system(cot, expected):
    p = build_prompt(SRC, MT, prompt_id="p3", cot=cot)
    assert expected in p.system


def test_no_cot_by_default():
    p = build_prompt(SRC, MT, prompt_id="p3")
    assert "Evaluation Steps" not in p.system


def test_cot2_includes_counting_strategy():
    """Figure 14 khác Figure 13 ở chỗ có biến đếm 'n'."""
    p = build_prompt(SRC, MT, prompt_id="p3", cot="cot2")
    assert "n = 0" in p.system
    assert "n == 0" in p.system


def test_vietnamese_cot_is_in_vietnamese():
    p = build_prompt(SRC, MT, prompt_id="p3", cot="cot2", language="vi")
    assert "Các bước đánh giá" in p.system
    # Nhãn vẫn giữ tiếng Anh để cô lập biến số
    assert "'No hallucination'" in p.system


# --------------------------------------------------------------------------
# Ngôn ngữ của nhãn
# --------------------------------------------------------------------------

def test_labels_stay_english_even_with_vietnamese_instructions():
    """Cô lập biến số: chỉ đổi ngôn ngữ hướng dẫn, giữ nguyên nhãn."""
    p = build_prompt(SRC, MT, prompt_id="p2", language="vi")
    assert p.labels == LABELS["en"]
    assert "Hallucination" in p.user


def test_label_language_can_be_switched_to_vietnamese():
    p = build_prompt(SRC, MT, prompt_id="p2", language="vi", label_language="vi")
    assert p.labels == LABELS["vi"]
    assert "Có ảo giác" in p.user


def test_label_tuple_order_is_negative_then_positive():
    for lang in ("en", "vi"):
        neg, pos = LABELS[lang]
        assert "No" in neg or "Không" in neg
        assert neg != pos


# --------------------------------------------------------------------------
# Cấu trúc thông điệp
# --------------------------------------------------------------------------

def test_as_messages_has_system_then_user():
    msgs = build_prompt(SRC, MT).as_messages()
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"]
    assert msgs[1]["content"]


# --------------------------------------------------------------------------
# Lưới biến thể
# --------------------------------------------------------------------------

def test_prompt_grid_covers_p1_table6():
    """P1 Bảng 6 quét 7 tổ hợp: p1/p2 × {no-CoT, CoT1}, p3 × {no-CoT, CoT1, CoT2}."""
    english = [v for v in PROMPT_GRID if v["language"] == "en"]
    assert len(english) == 7
    assert {"prompt_id": "p1", "cot": "none", "language": "en"} in english
    assert {"prompt_id": "p3", "cot": "cot2", "language": "en"} in english
    # P1 KHÔNG thử CoT2 với prompt 1 và 2
    assert {"prompt_id": "p1", "cot": "cot2", "language": "en"} not in english
    assert {"prompt_id": "p2", "cot": "cot2", "language": "en"} not in english


def test_prompt_grid_adds_vietnamese_variants():
    vietnamese = [v for v in PROMPT_GRID if v["language"] == "vi"]
    assert len(vietnamese) == 3


def test_every_grid_entry_builds():
    for variant in PROMPT_GRID:
        p = build_prompt(SRC, MT, "EN", "VI", **variant)
        assert p.system and p.user


def test_variant_names_are_unique():
    names = [variant_name(v["prompt_id"], v["cot"], v["language"]) for v in PROMPT_GRID]
    assert len(names) == len(set(names))


def test_variant_name_format():
    assert variant_name("p2", "none", "en") == "p2[en]"
    assert variant_name("p3", "cot2", "vi") == "p3+cot2[vi]"


# --------------------------------------------------------------------------
# Đầu vào không hợp lệ
# --------------------------------------------------------------------------

def test_invalid_prompt_id_raises():
    with pytest.raises(ValueError):
        build_prompt(SRC, MT, prompt_id="p9")


def test_invalid_cot_raises():
    with pytest.raises(ValueError):
        build_prompt(SRC, MT, cot="cot3")


def test_unknown_language_code_raises():
    with pytest.raises(KeyError):
        build_prompt(SRC, MT, src_lang="XX", prompt_id="p2")


def test_all_halomi_languages_have_names():
    """Mọi ngôn ngữ trong HalOmi phải dựng được prompt p2 (cần tên đầy đủ)."""
    for code in ("EN", "DE", "AR", "ZH", "RU", "ES", "KS", "MN", "YO", "VI"):
        for lang in ("en", "vi"):
            build_prompt(SRC, MT, src_lang=code, tgt_lang="EN",
                         prompt_id="p2", language=lang)
