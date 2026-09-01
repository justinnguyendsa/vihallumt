"""Tập C — bộ probe đặc thù tiếng Việt cho ViHalluMT.

Thiết kế: **cặp tối thiểu** (minimal pair)
------------------------------------------
Mỗi mục gồm cùng một câu nguồn tiếng Anh, một bản dịch **đúng** và một bản
dịch **sai** khác nhau ở đúng một hiện tượng ngôn ngữ học. Nhờ đó, chênh lệch
điểm mà detector gán cho hai bản cô lập được đúng biến số đang xét, thay vì bị
trộn lẫn với độ dài câu, chủ đề hay độ trôi chảy.

Cách đọc kết quả::

    Δ = score(bản sai) − score(bản đúng)

  Δ lớn  -> detector nhạy với hiện tượng này.
  Δ ≈ 0  -> **điểm mù**: detector không phân biệt được, dù nghĩa đã đổi hẳn.

Sáu hiện tượng được chọn vì chúng là chỗ mà công cụ NLP xây cho tiếng Anh hay
hỏng nhất khi đem sang tiếng Việt:

1. `tone`        — thanh điệu: đổi dấu là đổi hẳn từ (ma/má/mà/mã/mả/mạ).
2. `kinship`     — đại từ thân tộc mã hoá cả tuổi tác lẫn quan hệ, tiếng Anh
                   chỉ có "I/you" nên dịch máy hay chọn bừa.
3. `classifier`  — loại từ (cái/con/chiếc/quyển) bắt buộc và phụ thuộc danh từ.
4. `reduplication` — từ láy: nghĩa không suy ra được từ các thành tố.
5. `idiom`       — thành ngữ: dịch từng chữ ra nghĩa vô lý.
6. `number_format` — tiếng Việt dùng dấu chấm cho hàng nghìn và dấu phẩy cho
                   thập phân, ngược với tiếng Anh; ngày ghi dd/mm.

Mức nghiêm trọng gán theo thang HalOmi: đổi 1–2 từ -> `Small` (1). Đây là mức
khó nhất với mọi detector vì câu gần như không đổi về mặt bề mặt.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import pandas as pd


@dataclass(frozen=True)
class ProbeItem:
    """Một cặp tối thiểu: cùng nguồn, một bản đúng và một bản sai."""

    src_text: str
    mt_correct: str
    mt_corrupted: str
    phenomenon: str
    severity: int
    note: str

    def to_rows(self, pair_id: int) -> list[dict]:
        """Trải thành hai dòng dữ liệu (bản đúng nhãn 0, bản sai nhãn 1)."""
        common = {
            "pair_id": pair_id,
            "src_text": self.src_text,
            "src": "EN",
            "tgt": "VI",
            "direction": "EN-VI",
            "phenomenon": self.phenomenon,
            "note": self.note,
            "subset": "probe_vi",
        }
        return [
            {**common, "mt_text": self.mt_correct, "label": 0, "severity": 0,
             "variant": "correct"},
            {**common, "mt_text": self.mt_corrupted, "label": 1,
             "severity": self.severity, "variant": "corrupted"},
        ]


# --------------------------------------------------------------------------
# 1. Thanh điệu — đổi dấu là đổi từ
# --------------------------------------------------------------------------

TONE_ITEMS: tuple[ProbeItem, ...] = (
    ProbeItem(
        "My mother is cooking in the kitchen.",
        "Má tôi đang nấu ăn trong bếp.",
        "Mả tôi đang nấu ăn trong bếp.",
        "tone", 1,
        "má (mẹ) -> mả (ngôi mộ): đổi một dấu, câu thành vô nghĩa",
    ),
    ProbeItem(
        "He sold the house last year.",
        "Anh ấy đã bán căn nhà năm ngoái.",
        "Anh ấy đã bàn căn nhà năm ngoái.",
        "tone", 1,
        "bán (sell) -> bàn (bàn bạc/cái bàn)",
    ),
    ProbeItem(
        "She is my friend.",
        "Cô ấy là bạn của tôi.",
        "Cô ấy là bán của tôi.",
        "tone", 1,
        "bạn (friend) -> bán (sell)",
    ),
    ProbeItem(
        "The leaf fell from the tree.",
        "Chiếc lá rơi từ trên cây xuống.",
        "Chiếc lạ rơi từ trên cây xuống.",
        "tone", 1,
        "lá (leaf) -> lạ (strange)",
    ),
    ProbeItem(
        "There is grass in the garden.",
        "Trong vườn có cỏ.",
        "Trong vườn có cò.",
        "tone", 1,
        "cỏ (grass) -> cò (con cò)",
    ),
    ProbeItem(
        "My grandmother is eighty years old.",
        "Bà tôi năm nay tám mươi tuổi.",
        "Bả tôi năm nay tám mươi tuổi.",
        "tone", 1,
        "bà (grandmother) -> bả (bả chuột / cách gọi suồng sã)",
    ),
    ProbeItem(
        "The horse is running fast.",
        "Con ngựa đang chạy nhanh.",
        "Con ngừa đang chạy nhanh.",
        "tone", 1,
        "ngựa (horse) -> ngừa (phòng ngừa): từ không tồn tại ở vị trí này",
    ),
    ProbeItem(
        "He is three years old.",
        "Cậu bé lên ba tuổi.",
        "Cậu bé lên bà tuổi.",
        "tone", 1,
        "ba (three) -> bà (grandmother)",
    ),
)


# --------------------------------------------------------------------------
# 2. Đại từ thân tộc — tiếng Anh không mã hoá tuổi/quan hệ
# --------------------------------------------------------------------------

KINSHIP_ITEMS: tuple[ProbeItem, ...] = (
    ProbeItem(
        "My older brother works at a bank.",
        "Anh trai tôi làm việc ở ngân hàng.",
        "Em gái tôi làm việc ở ngân hàng.",
        "kinship", 1,
        "anh trai (older brother) -> em gái (younger sister): sai cả giới lẫn vai vế",
    ),
    ProbeItem(
        "My younger sister is studying medicine.",
        "Em gái tôi đang học ngành y.",
        "Chị gái tôi đang học ngành y.",
        "kinship", 1,
        "em gái (younger sister) -> chị gái (older sister)",
    ),
    ProbeItem(
        "His uncle lives in Hue.",
        "Chú của anh ấy sống ở Huế.",
        "Cháu của anh ấy sống ở Huế.",
        "kinship", 1,
        "chú (uncle) -> cháu (nephew/niece): đảo ngược quan hệ thế hệ",
    ),
    ProbeItem(
        "My grandfather told me a story.",
        "Ông tôi kể cho tôi nghe một câu chuyện.",
        "Bà tôi kể cho tôi nghe một câu chuyện.",
        "kinship", 1,
        "ông (grandfather) -> bà (grandmother)",
    ),
    ProbeItem(
        "She is my older sister.",
        "Chị ấy là chị gái của tôi.",
        "Chị ấy là mẹ của tôi.",
        "kinship", 1,
        "chị gái (older sister) -> mẹ (mother)",
    ),
    ProbeItem(
        "The teacher praised her student.",
        "Cô giáo khen học trò của mình.",
        "Cô giáo khen mẹ của mình.",
        "kinship", 1,
        "học trò (student) -> mẹ (mother)",
    ),
    ProbeItem(
        "My aunt is a doctor.",
        "Cô tôi là bác sĩ.",
        "Chú tôi là bác sĩ.",
        "kinship", 1,
        "cô (aunt, nữ) -> chú (uncle, nam)",
    ),
)


# --------------------------------------------------------------------------
# 3. Loại từ — bắt buộc và phụ thuộc danh từ
# --------------------------------------------------------------------------

CLASSIFIER_ITEMS: tuple[ProbeItem, ...] = (
    ProbeItem(
        "I bought a new book yesterday.",
        "Hôm qua tôi mua một quyển sách mới.",
        "Hôm qua tôi mua một con sách mới.",
        "classifier", 1,
        "quyển (sách) -> con (động vật): sai loại từ",
    ),
    ProbeItem(
        "There is a cat on the roof.",
        "Có một con mèo trên mái nhà.",
        "Có một quyển mèo trên mái nhà.",
        "classifier", 1,
        "con (động vật) -> quyển (sách vở)",
    ),
    ProbeItem(
        "He rode a motorbike to work.",
        "Anh ấy đi làm bằng một chiếc xe máy.",
        "Anh ấy đi làm bằng một ngôi xe máy.",
        "classifier", 1,
        "chiếc (phương tiện) -> ngôi (nhà cửa)",
    ),
    ProbeItem(
        "She hung a picture on the wall.",
        "Cô ấy treo một bức tranh lên tường.",
        "Cô ấy treo một con tranh lên tường.",
        "classifier", 1,
        "bức (tranh ảnh) -> con (động vật)",
    ),
    ProbeItem(
        "They built a big house.",
        "Họ xây một ngôi nhà lớn.",
        "Họ xây một chiếc nhà lớn.",
        "classifier", 1,
        "ngôi (nhà) -> chiếc (phương tiện, vật lẻ)",
    ),
    ProbeItem(
        "He put the rug on the floor.",
        "Anh ấy trải tấm thảm xuống sàn.",
        "Anh ấy trải quyển thảm xuống sàn.",
        "classifier", 1,
        "tấm (vật phẳng) -> quyển (sách vở)",
    ),
)


# --------------------------------------------------------------------------
# 4. Từ láy — nghĩa không suy ra được từ thành tố
# --------------------------------------------------------------------------

REDUPLICATION_ITEMS: tuple[ProbeItem, ...] = (
    ProbeItem(
        "The lights were shimmering on the lake.",
        "Ánh đèn lung linh trên mặt hồ.",
        "Ánh đèn lung lay trên mặt hồ.",
        "reduplication", 1,
        "lung linh (lấp lánh) -> lung lay (không vững): hai từ láy khác nghĩa hẳn",
    ),
    ProbeItem(
        "Her face looked pale after the illness.",
        "Gương mặt cô ấy xanh xao sau trận ốm.",
        "Gương mặt cô ấy xanh um sau trận ốm.",
        "reduplication", 1,
        "xanh xao (nhợt nhạt, nói về người) -> xanh um (xanh tốt, nói về cây)",
    ),
    ProbeItem(
        "The stars were sparkling in the night sky.",
        "Những vì sao lấp lánh trên bầu trời đêm.",
        "Những vì sao lấp ló trên bầu trời đêm.",
        "reduplication", 1,
        "lấp lánh (toả sáng) -> lấp ló (thấp thoáng ẩn hiện)",
    ),
    ProbeItem(
        "He left the house in a hurry.",
        "Anh ấy vội vàng rời khỏi nhà.",
        "Anh ấy vội vã rời khỏi nhà.",
        "reduplication", 0,
        "vội vàng / vội vã gần như đồng nghĩa — ĐÂY LÀ CẶP ĐỐI CHỨNG ÂM",
    ),
    ProbeItem(
        "The festival was brilliant and colourful.",
        "Lễ hội diễn ra rực rỡ và nhiều màu sắc.",
        "Lễ hội diễn ra rả rích và nhiều màu sắc.",
        "reduplication", 1,
        "rực rỡ (chói lọi) -> rả rích (mưa dai dẳng)",
    ),
    ProbeItem(
        "She felt very sad after the news.",
        "Cô ấy buồn bã sau khi nghe tin.",
        "Cô ấy buồn cười sau khi nghe tin.",
        "reduplication", 1,
        "buồn bã (sad) -> buồn cười (funny): đảo ngược sắc thái",
    ),
)


# --------------------------------------------------------------------------
# 5. Thành ngữ — dịch từng chữ ra nghĩa vô lý
# --------------------------------------------------------------------------

IDIOM_ITEMS: tuple[ProbeItem, ...] = (
    ProbeItem(
        "All his advice was wasted on them.",
        "Mọi lời khuyên của anh ấy đều như nước đổ lá khoai.",
        "Mọi lời khuyên của anh ấy đều như nước đổ lá chuối.",
        "idiom", 1,
        "'nước đổ lá khoai' là thành ngữ cố định; đổi 'khoai' thành 'chuối' làm mất tính thành ngữ",
    ),
    ProbeItem(
        "Perseverance will pay off in the end.",
        "Có công mài sắt có ngày nên kim.",
        "Có công mài sắt có ngày nên dao.",
        "idiom", 1,
        "'nên kim' là phần cố định của thành ngữ",
    ),
    ProbeItem(
        "You reap what you sow.",
        "Gieo gió gặt bão.",
        "Gieo gió gặt mưa.",
        "idiom", 1,
        "'gặt bão' cố định; 'gặt mưa' mất hoàn toàn ý nghĩa nhân quả nặng nề",
    ),
    ProbeItem(
        "Travel broadens the mind.",
        "Đi một ngày đàng học một sàng khôn.",
        "Đi một ngày đường học một sàng khôn.",
        "idiom", 1,
        "'ngày đàng' (cổ) -> 'ngày đường': sai dạng cố định của thành ngữ",
    ),
    ProbeItem(
        "One person alone cannot do it.",
        "Một cây làm chẳng nên non.",
        "Một cây làm chẳng nên rừng.",
        "idiom", 1,
        "'nên non' cố định (non = núi); 'nên rừng' phá vỡ vần và điển cố",
    ),
)


# --------------------------------------------------------------------------
# 6. Định dạng số / ngày — quy ước ngược với tiếng Anh
# --------------------------------------------------------------------------

NUMBER_ITEMS: tuple[ProbeItem, ...] = (
    ProbeItem(
        "The device costs 1,250.75 dollars.",
        "Thiết bị này có giá 1.250,75 đô la.",
        "Thiết bị này có giá 1,250.75 đô la.",
        "number_format", 1,
        "tiếng Việt: dấu chấm cho hàng nghìn, dấu phẩy cho thập phân — giữ nguyên "
        "định dạng Anh làm số bị đọc sai hàng nghìn lần",
    ),
    ProbeItem(
        "The population reached 2,500,000 people.",
        "Dân số đạt 2.500.000 người.",
        "Dân số đạt 2,5 người.",
        "number_format", 2,
        "hiểu nhầm dấu phẩy thành thập phân: 2.500.000 -> 2,5",
    ),
    ProbeItem(
        "The meeting is on 03/02/2024.",
        "Cuộc họp diễn ra ngày 2 tháng 3 năm 2024.",
        "Cuộc họp diễn ra ngày 3 tháng 2 năm 2024.",
        "number_format", 1,
        "mm/dd của Anh-Mỹ so với dd/mm của Việt Nam: đảo ngày và tháng",
    ),
    ProbeItem(
        "It weighs 0.5 kilograms.",
        "Nó nặng 0,5 ki-lô-gam.",
        "Nó nặng 5 ki-lô-gam.",
        "number_format", 2,
        "mất dấu thập phân: sai mười lần",
    ),
    ProbeItem(
        "The temperature dropped to -3.2 degrees.",
        "Nhiệt độ giảm xuống -3,2 độ.",
        "Nhiệt độ giảm xuống 3,2 độ.",
        "number_format", 1,
        "mất dấu âm: đảo ngược ý nghĩa nhiệt độ",
    ),
)


ALL_PROBE_ITEMS: tuple[ProbeItem, ...] = (
    TONE_ITEMS
    + KINSHIP_ITEMS
    + CLASSIFIER_ITEMS
    + REDUPLICATION_ITEMS
    + IDIOM_ITEMS
    + NUMBER_ITEMS
)

PHENOMENA: tuple[str, ...] = (
    "tone", "kinship", "classifier", "reduplication", "idiom", "number_format",
)


def build_probe_frame(items: tuple[ProbeItem, ...] = ALL_PROBE_ITEMS) -> pd.DataFrame:
    """Trải các cặp tối thiểu thành DataFrame hai dòng mỗi cặp."""
    rows = []
    for i, item in enumerate(items):
        rows.extend(item.to_rows(pair_id=i))
    return pd.DataFrame(rows)


def probe_delta(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Tính Δ = điểm(bản sai) − điểm(bản đúng) cho từng cặp tối thiểu.

    Đây là bảng dùng cho mục Phân tích lỗi: Δ nhỏ nghĩa là detector không nhận
    ra sự khác biệt, tức là **điểm mù** với hiện tượng ngôn ngữ đó.
    """
    wide = df.pivot_table(
        index=["pair_id", "phenomenon"], columns="variant", values=score_col
    ).reset_index()
    missing = {"correct", "corrupted"} - set(wide.columns)
    if missing:
        raise ValueError(f"Thieu bien the: {sorted(missing)}")
    wide["delta"] = wide["corrupted"] - wide["correct"]
    return wide


def probe_summary(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Δ trung bình và tỉ lệ xếp hạng đúng, gộp theo hiện tượng."""
    deltas = probe_delta(df, score_col)
    out = deltas.groupby("phenomenon").agg(
        n_pairs=("delta", "size"),
        mean_delta=("delta", "mean"),
        # Tỉ lệ cặp mà detector chấm bản sai cao điểm hơn bản đúng
        pct_ranked_correctly=("delta", lambda s: 100.0 * (s > 0).mean()),
    )
    return out.round(4).sort_values("mean_delta")
