"""Kiểm thử phần xây dựng ngữ liệu ViHalluMT."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from vihallumt.corpus.perturb import (
    GENERIC_PERTURBATIONS,
    available_perturbations,
    perturb,
    perturbation_plan,
    strip_diacritics,
)
from vihallumt.corpus.synthetic import (
    DEFAULT_MIX,
    OFF_TARGET_POOL,
    SEVERITY_OF_TYPE,
    build_synthetic_set,
    corrupt_oscillatory,
    corrupt_small,
    make_example,
)
from vihallumt.detectors.base import Pair
from vihallumt.detectors.ngram import NGramRepetitionDetector, repetition_excess

EN = "The meeting will start at nine in the morning tomorrow."
VI = "Cuộc họp sẽ bắt đầu lúc chín giờ sáng ngày mai."


# ==========================================================================
# perturb.py
# ==========================================================================

@pytest.mark.parametrize("kind", sorted(GENERIC_PERTURBATIONS))
def test_every_perturbation_changes_the_text(kind):
    rng = random.Random(0)
    out = perturb(EN, "EN", kind, rng)
    assert out.kind == kind
    assert out.text != EN
    assert out.text.strip()


def test_clean_leaves_text_untouched():
    out = perturb(EN, "EN", "clean", random.Random(0))
    assert out.text == EN
    assert out.is_clean


def test_perturbation_is_reproducible_given_seed():
    a = perturb(EN, "EN", "misspell", random.Random(7)).text
    b = perturb(EN, "EN", "misspell", random.Random(7)).text
    assert a == b


def test_different_seeds_give_different_results():
    outs = {perturb(EN, "EN", "misspell", random.Random(s)).text for s in range(10)}
    assert len(outs) > 1


def test_unknown_perturbation_raises():
    with pytest.raises(KeyError):
        perturb(EN, "EN", "khong_ton_tai", random.Random(0))


# -- nhiễu loạn đặc thù tiếng Việt -----------------------------------------

def test_diacritic_stripping_removes_all_vietnamese_marks():
    assert strip_diacritics("Cuộc họp sẽ bắt đầu") == "Cuoc hop se bat dau"


def test_diacritic_stripping_handles_d_with_stroke():
    """đ/Đ không phải dấu tổ hợp Unicode nên phải xử lý riêng."""
    assert strip_diacritics("đường Đông") == "duong Dong"


def test_diacritic_stripping_creates_genuine_ambiguity():
    """Năm chữ khác nghĩa hoàn toàn quy về cùng một dạng không dấu."""
    forms = ["má", "mà", "mã", "mả", "mạ"]
    assert {strip_diacritics(f) for f in forms} == {"ma"}


def test_diacritic_stripping_is_noop_for_ascii():
    assert strip_diacritics("hello world") == "hello world"


def test_vietnamese_perturbation_only_offered_for_vietnamese():
    assert "no_diacritics" in available_perturbations("VI")
    assert "no_diacritics" not in available_perturbations("EN")


def test_vietnamese_perturbation_rejected_for_english_source():
    with pytest.raises(KeyError):
        perturb(EN, "EN", "no_diacritics", random.Random(0))


def test_vietnamese_perturbation_applies_to_vietnamese_source():
    out = perturb(VI, "VI", "no_diacritics", random.Random(0))
    assert out.text == "Cuoc hop se bat dau luc chin gio sang ngay mai."


# -- kế hoạch nhiễu loạn ---------------------------------------------------

def test_plan_has_requested_length():
    assert len(perturbation_plan(100, "EN")) == 100


def test_plan_respects_clean_ratio():
    plan = perturbation_plan(100, "EN", clean_ratio=0.4)
    assert Counter(plan)["clean"] == 40


def test_plan_with_zero_clean_ratio_has_no_clean():
    plan = perturbation_plan(50, "EN", clean_ratio=0.0)
    assert "clean" not in plan


def test_plan_covers_all_perturbation_kinds():
    plan = perturbation_plan(200, "EN", clean_ratio=0.4)
    assert set(plan) - {"clean"} == set(GENERIC_PERTURBATIONS)


def test_plan_for_vietnamese_includes_diacritic_removal():
    plan = perturbation_plan(200, "VI", clean_ratio=0.4)
    assert "no_diacritics" in plan


def test_plan_is_reproducible():
    assert perturbation_plan(50, "EN", seed=1) == perturbation_plan(50, "EN", seed=1)


def test_plan_rejects_invalid_ratio():
    with pytest.raises(ValueError):
        perturbation_plan(10, "EN", clean_ratio=1.5)


# ==========================================================================
# synthetic.py
# ==========================================================================

PAIRS = [
    ("The meeting starts at nine.", "Cuộc họp bắt đầu lúc chín giờ."),
    ("He bought a history book.", "Anh ấy đã mua một cuốn sách lịch sử."),
    ("The weather is cold today.", "Hôm nay trời lạnh."),
    ("She works at a hospital.", "Cô ấy làm việc tại một bệnh viện."),
    ("They travelled to Da Nang.", "Họ đã đi du lịch Đà Nẵng."),
    ("The report was published.", "Bản báo cáo đã được công bố."),
]


def test_severity_mapping_matches_halomi_scale():
    assert SEVERITY_OF_TYPE["none"] == 0
    assert SEVERITY_OF_TYPE["small"] == 1
    assert SEVERITY_OF_TYPE["partial"] == 2
    for kind in ("full_oscillatory", "full_detached", "off_target"):
        assert SEVERITY_OF_TYPE[kind] == 3


def test_label_derives_from_severity():
    rng = random.Random(0)
    clean = make_example("a", "bản dịch", "none", ["x"], "khác", rng)
    dirty = make_example("a", "bản dịch", "full_detached", ["x"], "khác", rng)
    assert clean.label == 0 and clean.severity == 0
    assert dirty.label == 1 and dirty.severity == 3


def test_none_type_returns_reference_unchanged():
    ex = make_example("src", VI, "none", ["x"], "khác", random.Random(0))
    assert ex.mt_text == VI
    assert ex.mt_text == ex.reference


def test_severity_name_matches_halomi_vocabulary():
    ex = make_example("src", VI, "small", ["xyz"], "khác", random.Random(0))
    assert ex.severity_name == "2_Small_hallucination"


def test_unknown_type_raises():
    with pytest.raises(KeyError):
        make_example("src", VI, "khong_ton_tai", ["x"], "khác", random.Random(0))


# -- từng kiểu làm hỏng ----------------------------------------------------

def test_small_corruption_changes_few_words():
    rng = random.Random(3)
    out = corrupt_small(VI, ["quốc_tế", "ngân_hàng", "vũ_trụ"], rng)
    original, corrupted = VI.split(), out.split()
    assert len(original) == len(corrupted)
    n_changed = sum(a != b for a, b in zip(original, corrupted))
    assert 1 <= n_changed <= 2


def test_oscillatory_corruption_is_caught_by_ngram_detector():
    """Kiểm tra chéo giữa hai module: ảo giác dao động tổng hợp phải bị bắt.

    Nếu test này hỏng thì hoặc bộ sinh không thật sự tạo lặp, hoặc bộ đếm
    n-gram không thật sự đếm được — cả hai đều nghiêm trọng.
    """
    src = "The meeting will start at nine in the morning tomorrow."
    ref = "Cuộc họp sẽ bắt đầu lúc chín giờ sáng ngày mai."
    out = corrupt_oscillatory(ref, random.Random(1))
    assert repetition_excess(src, out) > 2
    assert NGramRepetitionDetector().predict([Pair(src, out)])[0] == 1


def test_off_target_pool_contains_non_vietnamese_text():
    for lang, sentences in OFF_TARGET_POOL.items():
        assert sentences
        for s in sentences:
            assert s.strip()


def test_off_target_excludes_target_language():
    rng = random.Random(0)
    ex = make_example("src", VI, "off_target", ["x"], "khác", rng, tgt_lang="id")
    assert ex.mt_text not in OFF_TARGET_POOL["id"]


# -- dựng cả tập -----------------------------------------------------------

def test_build_synthetic_set_size():
    out = build_synthetic_set(PAIRS)
    assert len(out) == len(PAIRS)


def test_build_synthetic_set_is_reproducible():
    a = build_synthetic_set(PAIRS, seed=5)
    b = build_synthetic_set(PAIRS, seed=5)
    assert [x.mt_text for x in a] == [x.mt_text for x in b]
    assert [x.hallucination_type for x in a] == [x.hallucination_type for x in b]


def test_build_synthetic_set_respects_mix_proportions():
    pairs = PAIRS * 100  # 600 cặp
    out = build_synthetic_set(pairs, seed=1)
    counts = Counter(x.hallucination_type for x in out)
    for kind, share in DEFAULT_MIX.items():
        expected = len(pairs) * share
        assert abs(counts[kind] - expected) <= 2, f"{kind}: {counts[kind]} vs {expected}"


def test_default_mix_is_half_negative():
    """Tập tổng hợp phải cân bằng, không lệch hẳn về phía dương."""
    assert DEFAULT_MIX["none"] == pytest.approx(0.50)


def test_default_mix_weights_oscillatory_highest_among_full():
    """P2 đo được 58–86% ảo giác thật là loại dao động."""
    assert DEFAULT_MIX["full_oscillatory"] > DEFAULT_MIX["full_detached"]
    assert DEFAULT_MIX["full_oscillatory"] > DEFAULT_MIX["off_target"]


def test_mix_must_sum_to_one():
    with pytest.raises(ValueError, match="Tong ti trong"):
        build_synthetic_set(PAIRS, mix={"none": 0.3, "small": 0.3})


def test_mix_rejects_unknown_type():
    with pytest.raises(ValueError, match="Loai khong hop le"):
        build_synthetic_set(PAIRS, mix={"none": 0.5, "khong_ton_tai": 0.5})


def test_empty_pairs_raises():
    with pytest.raises(ValueError):
        build_synthetic_set([])


def test_labels_are_consistent_with_types():
    for ex in build_synthetic_set(PAIRS * 20, seed=2):
        assert ex.label == int(ex.severity > 0)
        assert ex.severity == SEVERITY_OF_TYPE[ex.hallucination_type]


def test_negative_examples_keep_reference_text():
    for ex in build_synthetic_set(PAIRS * 20, seed=3):
        if ex.hallucination_type == "none":
            assert ex.mt_text == ex.reference


def test_positive_examples_differ_from_reference():
    changed = [ex for ex in build_synthetic_set(PAIRS * 20, seed=4)
               if ex.hallucination_type != "none"]
    assert changed
    # Cho phép vài ca biên (câu quá ngắn không hỏng được), nhưng đa số phải đổi
    n_changed = sum(ex.mt_text != ex.reference for ex in changed)
    assert n_changed / len(changed) > 0.9


# -- hồi quy cho hai lỗi đã sửa --------------------------------------------

@pytest.mark.parametrize("seed", range(30))
def test_misspell_never_silently_does_nothing(seed):
    """Lỗi từng gặp: từ được chọn quá ngắn hoặc ký tự thay trùng ký tự cũ.

    Phép nhiễu âm thầm không làm gì sẽ tạo ra dòng gắn nhãn 'đã nhiễu loạn'
    nhưng thực chất sạch, làm hỏng mọi phân tích theo loại nhiễu.
    """
    assert perturb(EN, "EN", "misspell", random.Random(seed)).text != EN


@pytest.mark.parametrize("seed", range(30))
def test_misspell_works_on_vietnamese_too(seed):
    assert perturb(VI, "VI", "misspell", random.Random(seed)).text != VI


def test_misspell_handles_text_with_only_short_words():
    text = "a b c"
    assert perturb(text, "EN", "misspell", random.Random(0)).text != text


@pytest.mark.parametrize("seed", range(30))
def test_oscillatory_always_exceeds_paper2_detection_threshold(seed):
    """Lỗi từng gặp: chỉ lặp 3 lần -> mức thừa đúng bằng 2, không vượt ngưỡng."""
    src = "The meeting will start at nine in the morning tomorrow."
    ref = "Cuộc họp sẽ bắt đầu lúc chín giờ sáng ngày mai."
    out = corrupt_oscillatory(ref, random.Random(seed))
    assert repetition_excess(src, out) > 2
    assert NGramRepetitionDetector().predict([Pair(src, out)])[0] == 1


def test_oscillatory_handles_very_short_reference():
    out = corrupt_oscillatory("Chào bạn", random.Random(0))
    assert out.count("Chào") >= 4


def test_all_synthetic_oscillatory_examples_are_detectable():
    """Mọi mẫu dao động trong tập sinh ra phải bị bộ dò n-gram bắt được."""
    det = NGramRepetitionDetector()
    examples = [e for e in build_synthetic_set(PAIRS * 20, seed=11)
                if e.hallucination_type == "full_oscillatory"]
    assert examples
    preds = det.predict([Pair(e.src_text, e.mt_text) for e in examples])
    assert preds.mean() == 1.0, f"bo sot {int((preds == 0).sum())}/{len(preds)} mau"


def test_detached_donor_is_never_the_same_as_the_reference():
    """Lỗi từng gặp: ngữ liệu có câu trùng -> mẫu 'tách rời' lại là bản dịch đúng.

    Chọn câu cho mượn theo chỉ số (j != i) là chưa đủ; phải so theo nội dung.
    Dùng `PAIRS * 5` để cố tình tạo ra rất nhiều bản dịch trùng nhau.
    """
    for ex in build_synthetic_set(PAIRS * 5, seed=9):
        if ex.hallucination_type == "full_detached":
            assert ex.mt_text != ex.reference, (
                f"Mau gan nhan Full hallucination nhung lai la ban dich dung: "
                f"{ex.mt_text!r}"
            )


@pytest.mark.parametrize("seed", range(15))
def test_no_positive_example_equals_its_reference(seed):
    """Không mẫu dương nào được trùng khít bản dịch chuẩn của chính nó."""
    for ex in build_synthetic_set(PAIRS * 5, seed=seed):
        if ex.severity > 0:
            assert ex.mt_text != ex.reference


def test_corpus_of_identical_references_raises_clearly():
    pairs = [("src %d" % i, "cùng một câu") for i in range(10)]
    with pytest.raises(ValueError, match="deu giong nhau"):
        build_synthetic_set(pairs, seed=0)
