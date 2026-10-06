"""台本の 1 発話を、TTS・アライメント・テキストストリームで共通に使う単語の単位に分ける。

単位は pyopenjtalk.run_frontend の単語。固有名詞は 1 単位にまとめ、読みを辞書の値で上書きする。
算用数字は pyopenjtalk が漢数字に正規化するが、その範囲にかかる単位は 1 単位にまとめ、表記を台本の算用数字に
戻す（漢数字の表記は "spoken" に残す）。この同じ単位列から次の 3 つを作るので、三者の表記と読みがずれない。
- TTS の入力: 固有名詞は読みのかな、数字は漢数字（spoken）、それ以外は正規化した表記
- アライメント: 各単位の表記（数字は spoken）と読み
- テキストストリーム: 表記（固有名詞と数字はプロンプトとバイト単位で同じ表記）。句読点は前の単語にくっつける
数字を算用数字で書くのは、元のモデル（llm-jp-moshi）がテキストストリームに算用数字を出し、プロンプトも算用数字で
書くため（漢数字だと「2850円」を「二千八百五十円」に言い換える必要があり、pilot40 v1 では値段の答えが 0/8 だった）。
"""
from __future__ import annotations

import re
import unicodedata

import pyopenjtalk

# 算用数字の範囲。「6万5千」「2.5」のように、数字の間の万・千などと小数点も含める
NUMBER = re.compile(r"[0-9０-９]+(?:[.．][0-9０-９]+)?(?:[万億千百][0-9０-９]*)*")


def kata2hira(s: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


def _frontend_units(text: str) -> list[dict]:
    out = []
    for f in pyopenjtalk.run_frontend(text):
        if not f["string"].strip():  # 全角空白など
            continue
        pron = f["pron"].replace("’", "").replace("'", "")
        punct = f["pos"] == "記号" and f["mora_size"] == 0
        out.append({"word": f["string"], "hira": "" if punct else kata2hira(pron),
                    "read": "" if punct else kata2hira(f["read"]), "punct": punct, "entity": False})
    return out


def utterance_units(text: str, readings: dict[str, str] | None = None) -> list[dict]:
    """text を単位に分ける。readings（表記 -> ひらがなの読み）にある表記は 1 単位にし、読みを上書きする。

    文全体を pyopenjtalk にかけてから（「川崎様」の「様」は文脈がないと「よう」と読まれる）、固有名詞の
    文字範囲にかかる単位を 1 つにまとめる。固有名詞の境界が単位の途中にある場合は、はみ出した部分だけを
    pyopenjtalk にかけ直す。長い表記を先に照合する（「中山」より「中山商店」）。最後に算用数字の範囲を
    1 単位にまとめて表記を算用数字に戻す（_restore_numbers）。
    """
    return _restore_numbers(text, _merge_entities(_frontend_units(text), readings))


def _restore_numbers(text: str, units: list[dict]) -> list[dict]:
    """算用数字の範囲にかかる単位をまとめ、表記を元の文の算用数字に戻す。

    元の文と、単位の表記をつないだ正規化後の文字列を 1 文字ずつ対応させる（数字の範囲は、その範囲だけを
    pyopenjtalk にかけた漢数字と対応させる）。対応が取れない文は単位を変えずに返す（数字は漢数字のまま）。
    読みは文全体で解析したもの（「1分」の「いっぷん」）をつなぐ。
    """
    spans = [(m.start(), m.end(), "".join(u["word"] for u in _frontend_units(m.group()))) for m in NUMBER.finditer(text)]
    if not spans:
        return units
    joined = "".join(u["word"] for u in units)
    src: list[tuple[int, int]] = []  # joined の各文字が来た元の文の範囲
    is_number: list[bool] = []
    i = j = 0
    for b, e, kanji in spans + [(len(text), len(text), "")]:
        while i < b:
            if not text[i].strip():  # 空白は frontend が落とす
                i += 1
                continue
            if j >= len(joined) or unicodedata.normalize("NFKC", joined[j]) != unicodedata.normalize("NFKC", text[i]):
                return units
            src.append((i, i + 1))
            is_number.append(False)
            i, j = i + 1, j + 1
        if kanji:
            # 「5月」「3つ」の数字は漢数字にならず全角の算用数字のまま残る（文脈によって変わる）
            n = len(kanji) if joined[j:j + len(kanji)] == kanji else (
                e - b if unicodedata.normalize("NFKC", joined[j:j + e - b]) == unicodedata.normalize("NFKC", text[b:e])
                else 0)
            if not n:
                return units
            src += [(b, e)] * n
            is_number += [True] * n
            i, j = e, j + n
    if j != len(joined):
        return units
    # 元の文の範囲が重なる単位（同じ数字の範囲にかかるもの）を 1 つのまとまりにする
    groups: list[list] = []  # [元の begin, 元の end, 数字を含むか, 単位のリスト]
    pos = 0
    for u in units:
        b, e = pos, pos + len(u["word"])
        pos = e
        ob, oe = src[b][0], src[e - 1][1]
        if groups and ob < groups[-1][1]:
            g = groups[-1]
            g[1], g[2] = max(g[1], oe), g[2] or any(is_number[b:e])
            g[3].append(u)
        else:
            groups.append([ob, oe, any(is_number[b:e]), [u]])
    out = []
    for ob, oe, has_number, us in groups:
        if not has_number or any(u["entity"] or u["punct"] for u in us):
            out += us
            continue
        out.append({"word": text[ob:oe], "hira": "".join(u["hira"] for u in us), "read": "".join(u["read"] for u in us),
                    "punct": False, "entity": False, "spoken": "".join(u["word"] for u in us)})
    return out


def _merge_entities(units: list[dict], readings: dict[str, str] | None) -> list[dict]:
    if not readings:
        return units
    joined = "".join(u["word"] for u in units)
    spans: list[tuple[int, int, str, str]] = []  # 正規化後の文字列上の [begin, end)
    for surface, reading in sorted(readings.items(), key=lambda kv: -len(kv[0])):
        start = 0
        while surface and (pos := joined.find(surface, start)) >= 0:
            end = pos + len(surface)
            if all(end <= b or pos >= e for b, e, _, _ in spans):
                spans.append((pos, end, surface, reading))
            start = end
    if not spans:
        return units
    spans.sort()
    out: list[dict] = []
    offset, si = 0, 0
    for u in units:
        b, e = offset, offset + len(u["word"])
        offset = e
        while si < len(spans) and spans[si][1] <= b:
            si += 1
        if si >= len(spans) or e <= spans[si][0]:
            out.append(u)
            continue
        sb, se, surface, reading = spans[si]
        if b < sb:  # 単位の前半が固有名詞の外
            out += _frontend_units(u["word"][:sb - b])
        if not out or out[-1].get("_span") != sb:
            out.append({"word": surface, "hira": kata2hira(reading), "read": kata2hira(reading), "punct": False,
                        "entity": True, "_span": sb})
        if e > se:  # 単位の後半が固有名詞の外
            out += _frontend_units(u["word"][se - b:])
    for u in out:
        u.pop("_span", None)
    return out


def _has_kanji(s: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" or c == "々" for c in s)


def tts_text(units: list[dict], kana_for_kanji: bool = False) -> str:
    """TTS に渡す文。固有名詞は読みのかなに置き換える。

    数字は漢数字（spoken）で渡す。kana_for_kanji=True では漢字を含む単語もすべて読み（pyopenjtalk の read）の
    かなにする。Qwen3-TTS は珍しい漢字を読み違える（承って、男体山、丑の日。M2 で 3 回作り直しても同じ誤り）ので、
    作り直しのときに使う。数字とその直後の単位（助数詞）のかなは、連濁と促音を反映した発音（pron。「いっぷん」
    「さんびゃく」）にする。
    """
    def piece(u, prev):
        if u["entity"]:
            return u["hira"]
        surface = u.get("spoken", u["word"])
        if kana_for_kanji and _has_kanji(surface):
            after_number = prev is not None and "spoken" in prev
            return u["hira"] if "spoken" in u or after_number else (u.get("read") or u["hira"])
        return surface
    return "".join(piece(u, units[i - 1] if i else None) for i, u in enumerate(units))


def merge_punct(words: list[dict]) -> list[dict]:
    """句読点の単語を直前の単語の表記にくっつける（時刻は直前の単語のまま）。先頭の句読点は捨てる。
    数字の単語の漢数字（spoken）にも同じ句読点を付ける。"""
    out: list[dict] = []
    for w in words:
        if w.get("punct"):
            if out:
                out[-1] = dict(out[-1], word=out[-1]["word"] + w["word"])
                if "spoken" in out[-1]:
                    out[-1]["spoken"] += w["word"]
            continue
        out.append({k: v for k, v in w.items() if k != "punct"})
    return out


def mora_count(units: list[dict]) -> int:
    """話速の上限（data_spec §5.3、9 モーラ/秒）の確認用。小さい「ゃゅょ」などは数えない。"""
    small = set("ゃゅょぁぃぅぇぉゎ")
    return sum(1 for u in units for c in u["hira"] if c not in small)
