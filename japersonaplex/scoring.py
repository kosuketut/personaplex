"""flip 評価の採点。重い依存を持たない（ASR 用の別環境からも使う）。"""
import re
import unicodedata

KANJI_DIGITS = {c: i for i, c in enumerate("〇一二三四五六七八九")}
KANJI_UNITS = {"十": 10, "百": 100, "千": 1000}


def normalize_numbers(text: str) -> str:
    """数の表記をそろえる。「十七時」「17時」「１７時」は "17時"、「千百円」「1,100円」「1千1百円」は "1100円" になる。"""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)

    def convert(match: re.Match) -> str:
        chars = match.group()
        if chars.isdigit():
            return chars
        total, digit = 0, None
        for c in chars:
            if c.isdigit() or c in KANJI_DIGITS:
                value = int(c) if c.isdigit() else KANJI_DIGITS[c]
                digit = value if digit is None else digit * 10 + value
            else:
                total += (1 if digit is None else digit) * KANJI_UNITS[c]
                digit = None
        return str(total + (digit or 0))
    return re.sub("[0-9〇一二三四五六七八九十百千]+", convert, text)


def mentions(expect: str, text: str) -> bool:
    """期待値が出ているか。「8時」が「18時」に一致しないよう、前に数字が続かないことを見る。"""
    return re.search(r"(?<!\d)" + re.escape(normalize_numbers(expect)), normalize_numbers(text)) is not None
