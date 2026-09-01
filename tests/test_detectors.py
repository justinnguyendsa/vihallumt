"""Kiểm thử bộ phát hiện ảo giác.

Toàn bộ chạy được trên CPU không cần GPU: phần logic của LLM detector được
tách thành hàm thuần tuý, và phần cần mô hình thì dùng tokenizer giả.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vihallumt.detectors.base import Pair, pairs_from_frame
from vihallumt.detectors.llm import (
    LLMConfig,
    LLMDetector,
    hallucination_prob_from_logits,
    label_first_token_ids,
    parse_generated_label,
    resolve_label_token_ids,
)
from vihallumt.detectors.ngram import (
    NGramRepetitionDetector,
    repetition_excess,
    tokenize,
    top_ngram_count,
)


class FakeTokenizer:
    """Tokenizer mức từ, đủ để kiểm thử logic chọn token nhãn."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = []
        for word in text.split():
            if word not in self._vocab:
                self._vocab[word] = len(self._vocab) + 10
            ids.append(self._vocab[word])
        return ids


# --------------------------------------------------------------------------
# hallucination_prob_from_logits
# --------------------------------------------------------------------------

def test_prob_is_half_when_labels_tie():
    logits = np.array([[1.0, 1.0, 0.0]])
    assert hallucination_prob_from_logits(logits, [0], [1])[0] == pytest.approx(0.5)


def test_prob_approaches_one_when_positive_dominates():
    logits = np.array([[0.0, 20.0, 0.0]])
    assert hallucination_prob_from_logits(logits, [0], [1])[0] > 0.999


def test_prob_approaches_zero_when_negative_dominates():
    logits = np.array([[20.0, 0.0, 0.0]])
    assert hallucination_prob_from_logits(logits, [0], [1])[0] < 0.001


def test_normalisation_ignores_unrelated_vocabulary():
    """Chuẩn hoá chỉ trên hai nhãn, không trên toàn bộ từ vựng.

    Nếu chuẩn hoá trên toàn từ vựng thì một token vô can có logit rất cao sẽ
    ép xác suất hai nhãn xuống gần 0 và làm hỏng thang điểm.
    """
    logits = np.array([[0.0, 0.0, 1000.0]])  # token 2 áp đảo nhưng không phải nhãn
    assert hallucination_prob_from_logits(logits, [0], [1])[0] == pytest.approx(0.5)


def test_returns_half_when_both_labels_impossible():
    logits = np.array([[-1e9, -1e9, 0.0]])
    assert hallucination_prob_from_logits(logits, [0], [1])[0] == pytest.approx(0.5)


def test_accepts_one_dimensional_logits():
    out = hallucination_prob_from_logits(np.array([0.0, 5.0, 0.0]), [0], [1])
    assert out.shape == (1,)
    assert out[0] > 0.99


def test_handles_batches():
    logits = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [1.0, 1.0, 0.0]])
    out = hallucination_prob_from_logits(logits, [0], [1])
    assert out.shape == (3,)
    assert out[0] < 0.01 and out[1] > 0.99 and out[2] == pytest.approx(0.5)


def test_multiple_ids_per_label_are_summed():
    """Nhiều biến thể bề mặt của cùng một nhãn phải được cộng xác suất."""
    logits = np.array([[0.0, np.log(0.5), np.log(0.5)]])
    # nhãn âm = {0}, nhãn dương = {1, 2} -> p_pos = 0.5+0.5 = 1.0, p_neg = 1.0
    assert hallucination_prob_from_logits(logits, [0], [1, 2])[0] == pytest.approx(0.5)


def test_empty_id_set_raises():
    with pytest.raises(ValueError):
        hallucination_prob_from_logits(np.array([[1.0, 2.0]]), [], [1])


# --------------------------------------------------------------------------
# Chọn token nhãn
# --------------------------------------------------------------------------

def test_label_first_token_ids_covers_case_variants():
    tok = FakeTokenizer()
    ids = label_first_token_ids(tok, "Hallucination")
    assert len(ids) >= 2  # ít nhất "Hallucination" và "hallucination"


def test_resolve_label_token_ids_separates_english_labels():
    tok = FakeTokenizer()
    neg, pos = resolve_label_token_ids(tok, ("No hallucination", "Hallucination"))
    assert neg and pos
    assert not (neg & pos)


def test_resolve_label_token_ids_separates_vietnamese_labels():
    tok = FakeTokenizer()
    neg, pos = resolve_label_token_ids(tok, ("Không có ảo giác", "Có ảo giác"))
    assert neg and pos
    assert not (neg & pos)


def test_resolve_label_token_ids_raises_for_indistinguishable_labels():
    """Hai nhãn cùng token đầu -> logit-scoring vô nghĩa, phải báo lỗi rõ ràng."""
    tok = FakeTokenizer()
    with pytest.raises(ValueError, match="Khong phan biet duoc"):
        resolve_label_token_ids(tok, ("Yes definitely", "Yes"))


# --------------------------------------------------------------------------
# parse_generated_label
# --------------------------------------------------------------------------

EN = ("No hallucination", "Hallucination")
VI = ("Không có ảo giác", "Có ảo giác")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hallucination", 1),
        ("No hallucination", 0),
        ("no hallucination", 0),
        ("HALLUCINATION", 1),
        ("  Hallucination  ", 1),
        ("The answer is: No hallucination", 0),
        ("There is hallucination in this translation.", 1),
        ("I think there is no hallucination here.", 0),
        ("完全に無関係", None),
        ("", None),
    ],
)
def test_parse_english_label(text, expected):
    assert parse_generated_label(text, EN) is expected or \
           parse_generated_label(text, EN) == expected


def test_negative_label_wins_when_it_contains_positive():
    """'No hallucination' chứa 'hallucination' — không được nhầm thành dương."""
    assert parse_generated_label("No hallucination", EN) == 0


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Có ảo giác", 1),
        ("Không có ảo giác", 0),
        ("Bản dịch này không có ảo giác.", 0),
        ("Kết luận: có ảo giác", 1),
    ],
)
def test_parse_vietnamese_label(text, expected):
    assert parse_generated_label(text, VI) == expected


def test_unrecognised_output_returns_none():
    assert parse_generated_label("maybe?", EN) is None


# --------------------------------------------------------------------------
# LLMConfig
# --------------------------------------------------------------------------

def test_logit_mode_with_cot_is_rejected():
    """CoT cần sinh chuỗi lập luận; token đầu tiên không phải nhãn."""
    with pytest.raises(ValueError, match="logit-scoring voi CoT"):
        LLMConfig(mode="logit", cot="cot1")


def test_generate_mode_with_cot_is_allowed():
    cfg = LLMConfig(mode="generate", cot="cot2")
    assert cfg.cot == "cot2"


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        LLMConfig(mode="beam")


def test_default_config_matches_paper():
    """P1 Appendix D.1: max_output_token = 15, zero-shot."""
    cfg = LLMConfig()
    assert cfg.max_new_tokens == 15
    assert cfg.cot == "none"


# --------------------------------------------------------------------------
# LLMDetector (không nạp mô hình)
# --------------------------------------------------------------------------

def test_detector_name_encodes_variant():
    det = LLMDetector("Qwen/Qwen2.5-7B-Instruct",
                      LLMConfig(prompt_id="p3", cot="cot2", language="vi", mode="generate"))
    assert "Qwen2.5-7B-Instruct" in det.name
    assert "p3+cot2[vi]" in det.name
    assert "generate" in det.name


def test_detector_marks_generate_mode_as_non_continuous():
    """Chế độ generate chỉ cho nhãn cứng -> ROC-AUC không có ý nghĩa."""
    assert LLMDetector("x", LLMConfig(mode="logit")).produces_continuous_score
    assert not LLMDetector("x", LLMConfig(mode="generate")).produces_continuous_score


def test_detector_builds_prompt_from_pair():
    det = LLMDetector("x", LLMConfig(prompt_id="p2"))
    p = det.build(Pair("hello", "xin chào", "EN", "VI"))
    assert "hello" in p.user and "xin chào" in p.user
    assert "English" in p.system and "Vietnamese" in p.system


def test_empty_input_returns_empty_array():
    det = LLMDetector("x")
    assert det.score([]).shape == (0,)


# --------------------------------------------------------------------------
# Bộ đếm lặp n-gram
# --------------------------------------------------------------------------

def test_tokenize_handles_vietnamese_diacritics():
    assert tokenize("Con mèo ngồi trên thảm.") == ["con", "mèo", "ngồi", "trên", "thảm"]


def test_top_ngram_count_on_repeated_text():
    text = "the cat sat on the cat sat on the cat sat on"
    assert top_ngram_count(text, n=4) == 3


def test_top_ngram_count_on_normal_text():
    assert top_ngram_count("the cat sat on the mat quietly today", n=4) == 1


def test_top_ngram_count_for_short_text_is_zero():
    assert top_ngram_count("hai từ", n=4) == 0


def test_repetition_excess_subtracts_source_repetition():
    """Câu nguồn vốn đã lặp thì không được phạt oan bản dịch."""
    src = "a b c d a b c d a b c d"
    mt = "a b c d a b c d a b c d"
    assert repetition_excess(src, mt, n=4) == 0


def test_repetition_excess_flags_oscillatory_translation():
    src = "The meeting will start at nine in the morning."
    mt = "Cuộc họp bắt đầu lúc chín giờ sáng " + "lúc chín giờ sáng " * 5
    assert repetition_excess(src, mt, n=4) >= 3


def test_ngram_detector_uses_paper2_threshold():
    det = NGramRepetitionDetector()
    assert det.n == 4
    assert det.threshold == 2


def test_ngram_detector_predicts_binary():
    src = "The meeting will start at nine in the morning."
    normal = Pair(src, "Cuộc họp sẽ bắt đầu lúc chín giờ sáng.")
    oscillatory = Pair(src, "Cuộc họp bắt đầu " + "lúc chín giờ sáng " * 6)
    det = NGramRepetitionDetector()
    assert list(det.predict([normal, oscillatory])) == [0, 1]


def test_ngram_detector_scores_are_ordered():
    src = "one two three four five six"
    pairs = [
        Pair(src, "một hai ba bốn năm sáu"),
        Pair(src, "một hai ba bốn " * 4),
    ]
    scores = NGramRepetitionDetector().score(pairs)
    assert scores[1] > scores[0]


# --------------------------------------------------------------------------
# base.pairs_from_frame
# --------------------------------------------------------------------------

def test_pairs_from_frame_reads_language_columns():
    df = pd.DataFrame({
        "src_text": ["a", "b"],
        "mt_text": ["x", "y"],
        "src": ["EN", "VI"],
        "tgt": ["VI", "EN"],
    })
    pairs = pairs_from_frame(df)
    assert len(pairs) == 2
    assert pairs[0] == Pair("a", "x", "EN", "VI")
    assert pairs[1].src_lang == "VI"


def test_pairs_from_frame_defaults_languages():
    df = pd.DataFrame({"src_text": ["a"], "mt_text": ["x"]})
    assert pairs_from_frame(df)[0] == Pair("a", "x", "EN", "VI")


def test_pairs_from_frame_missing_column_raises():
    with pytest.raises(KeyError):
        pairs_from_frame(pd.DataFrame({"src_text": ["a"]}))
