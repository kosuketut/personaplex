"""datagen.units: TTS 入力・アライメント・テキストストリームが同じ単位列から作られることの確認。"""
from japersonaplex.datagen.units import merge_punct, tts_text, utterance_units


def words(units):
    return [u["word"] for u in units]


def test_entity_keeps_context_reading():
    # 「様」は文全体で解析しないと「よう」と読まれる
    units = utterance_units("川崎様ですね。", {"川崎": "かわさき"})
    assert words(units) == ["川崎", "様", "です", "ね", "。"]
    assert units[0]["entity"] and units[0]["hira"] == "かわさき"
    assert units[1]["hira"] == "さま"


def test_multi_unit_entity_is_merged_and_reading_overridden():
    units = utterance_units("はい、前田動物病院、辻でございます。", {"前田動物病院": "まえだどうぶつびょういん", "辻": "つじ"})
    assert "前田動物病院" in words(units)
    assert tts_text(units) == "はい、まえだどうぶつびょういん、つじでございます。"
    # テキストストリーム側は漢字の表記のまま（プロンプトとバイト単位で一致させるため）
    assert "".join(words(units)) == "はい、前田動物病院、辻でございます。"


def test_reading_override_changes_only_tts_input():
    units = utterance_units("担当の東海林です。", {"東海林": "とうかいりん"})
    assert tts_text(units) == "担当のとうかいりんです。"
    assert "".join(words(units)) == "担当の東海林です。"


def test_entity_boundary_inside_a_unit():
    # 「学習」の途中で固有名詞が終わる場合も、表記は元の文に戻る
    units = utterance_units("富士見学習塾です", {"富士見学": "ふじみがく"})
    assert units[0] == {"word": "富士見学", "hira": "ふじみがく", "read": "ふじみがく", "punct": False, "entity": True}
    assert "".join(words(units)) == "富士見学習塾です"


def test_numbers_keep_arabic_digits_in_text_and_kanji_in_tts():
    units = utterance_units("料金は3500円です。")
    assert "".join(words(units)) == "料金は3500円です。"  # テキストストリームはプロンプトと同じ算用数字
    assert tts_text(units) == "料金は三千五百円です。"
    num = next(u for u in units if u["word"] == "3500")
    assert num["spoken"] == "三千五百" and num["hira"] == "さんぜんごひゃく" and not num["entity"]


def test_numbers_inside_a_word_and_compound_numbers():
    # 「3日間」は助数詞ごと 1 単位、「6万5千」は万・千を含めて 1 つの数、「1分」の読みは文脈どおり
    units = utterance_units("3日間で6万5千円、1分ほど、10時から19時まで。")
    assert "".join(words(units)) == "3日間で6万5千円、1分ほど、10時から19時まで。"
    assert [u["word"] for u in units if "spoken" in u] == ["3日間", "6万5千", "1", "10", "19"]
    assert tts_text(units) == "三日間で六万五千円、一分ほど、十時から十九時まで。"
    assert tts_text(units, kana_for_kanji=True).startswith("みっかかんでろくまんごせんえん、いっぷんほど")


def test_numbers_left_as_fullwidth_digits_by_the_frontend():
    # 月や「つ」の前の数字は pyopenjtalk が全角の算用数字のまま残す。表記は半角に戻し、TTS には全角のまま渡す
    units = utterance_units("10月18日に3つ、2024年です。")
    assert "".join(words(units)) == "10月18日に3つ、2024年です。"
    assert tts_text(units) == "１０月十八日に３つ、二千二十四年です。"


def test_merge_punct_keeps_spoken_for_numbers():
    words = [{"word": "1050", "spoken": "千五十", "punct": False}, {"word": "円", "punct": False},
             {"word": "。", "punct": True}, {"word": "19", "spoken": "十九", "punct": False}, {"word": "、", "punct": True}]
    merged = merge_punct(words)
    assert [w["word"] for w in merged] == ["1050", "円。", "19、"]
    assert merged[2]["spoken"] == "十九、" and "spoken" not in merged[1]


def test_numbers_and_entities_together():
    units = utterance_units("川崎様、2名様ですね。", {"川崎": "かわさき"})
    assert "".join(words(units)) == "川崎様、2名様ですね。"
    assert tts_text(units) == "かわさき様、二名様ですね。"
    assert units[0]["entity"]


def test_merge_punct_attaches_to_previous_word():
    units = utterance_units("はい、そうです。")
    timed = [dict(u, start=float(i), end=float(i) + 0.5) for i, u in enumerate(units)]
    merged = merge_punct(timed)
    assert [w["word"] for w in merged] == ["はい、", "そう", "です。"]
    assert merged[0]["start"] == 0.0 and merged[2]["start"] == 3.0


def test_wildcard_edit_distance_ignores_entity_spans():
    from japersonaplex.datagen.align import _canon, substring_distance, wildcard_edit_distance
    # 名前の部分（*）は whisper が別の漢字で書いても数えない
    assert wildcard_edit_distance("ハイ*デゴザイマス", "ハイタニグチハスタカーテンデゴザイマス") == 0
    assert wildcard_edit_distance("ハイ*デゴザイマス", "ハイデゴザイマス") == 0
    assert wildcard_edit_distance("ハイ*デゴザイマス", "ハイデゴザマス") == 1
    assert wildcard_edit_distance("アイウ", "アイウ") == 0 and wildcard_edit_distance("アイウ", "") == 3
    assert substring_distance("カンベ", "カベサマデ") == 1
    assert substring_distance("スミダ", "ハイウケツケノスミダデス") == 0
    assert _canon("トウカイリン") == _canon("トーカイリン")


def test_kana_for_kanji_fallback_reads_every_kanji_word():
    units = utterance_units("はい、承っております。佐藤様", {"佐藤": "さとう"})
    assert tts_text(units) == "はい、承っております。さとう様"
    assert tts_text(units, kana_for_kanji=True) == "はい、うけたまわっております。さとうさま"
