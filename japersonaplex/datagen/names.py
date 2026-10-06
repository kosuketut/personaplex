"""固有名詞（人名・店名）のプールを、読み付きの辞書から作る（data_spec §1.7）。

人名と地名は IPAdic（mecab-ipadic の Noun.name.csv、Noun.place.csv。NAIST のライセンスで、著作権表示と
免責の文を残せば改変・再配布してよい。`data/dict/ipadic/COPYING`）から取る。
- 同じ表記に読みが複数ある名前は使わない（data_spec §1.7-5 の案 A）。辞書に 1 つしか読みが載っていない
  表記は、現実に別の読みがあっても通ってしまう（例: 東海林は「しょうじ」だけ）
- 読みのハッシュで 10% を評価専用（H-name）に取り分ける（§6.1）
- 辞書のコスト（小さいほどよく使われる語）を頻度の目安にし、半分はよくある名前、半分は一様に引く

辞書の入手（リポジトリには入れない）:
    mkdir -p data/dict/ipadic && cd data/dict/ipadic && for f in Noun.name.csv Noun.place.csv COPYING; do
      curl -sfLO https://raw.githubusercontent.com/taku910/mecab/master/mecab-ipadic/$f; done
"""
from __future__ import annotations

import csv
import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .units import kata2hira

HELDOUT_FRACTION = 0.1
COMMON_TOP = 2000  # 「よくある名前」とみなすコスト順の上位


@dataclass(frozen=True)
class Name:
    surface: str
    reading: str  # ひらがな
    cost: int


def is_heldout(reading: str) -> bool:
    return int(hashlib.sha1(("name:" + reading).encode()).hexdigest()[:8], 16) % 1000 < HELDOUT_FRACTION * 1000


def _load(path: Path, keep) -> list[Name]:
    readings: dict[str, set[str]] = defaultdict(set)
    cost: dict[str, int] = {}
    with open(path, encoding="euc_jp", errors="replace") as f:
        for row in csv.reader(f):
            if len(row) < 12 or not keep(row):
                continue
            surface, reading = row[0], kata2hira(row[11])
            readings[surface].add(reading)
            cost[surface] = min(cost.get(surface, 10 ** 9), int(row[3]))
    return [Name(s, next(iter(r)), cost[s]) for s, r in readings.items()
            if len(r) == 1 and 1 <= len(s) <= 4 and "�" not in s]


def _is_katakana(s: str) -> bool:
    return all("ァ" <= c <= "ヶ" or c == "ー" for c in s)


class NamePool:
    """split="train" は学習用、"heldout" は評価専用（読みのハッシュで分ける）。"""

    FOREIGN_RATE = 0.1  # data_spec §1.7-2: 外国名のカタカナ表記 10%

    def __init__(self, ipadic_dir: str | Path, split: str = "train", tokenizer: str | Path | None = None):
        """tokenizer（llm-jp の SPM）を渡すと、語彙に無い文字を含む名前（約 1.5%。例: 龝山、中嶌）を除く。
        そうした文字はプロンプトでもテキストストリームでも未知語になり、名前を写す学習ができないため。"""
        d = Path(ipadic_dir)
        surnames = self._split(_load(d / "Noun.name.csv", lambda r: r[7] == "姓"), split)
        given = self._split(_load(d / "Noun.name.csv", lambda r: r[7] == "名"), split)
        places = self._split(_load(d / "Noun.place.csv", lambda r: r[6] == "地域" and r[7] == "一般"), split)
        if tokenizer is not None:
            import sentencepiece
            sp = sentencepiece.SentencePieceProcessor(str(tokenizer))
            ok = lambda n: sp.unk_id() not in sp.encode(n.surface)  # noqa: E731
            surnames, given, places = ([n for n in x if ok(n)] for x in (surnames, given, places))
        self.surnames = [n for n in surnames if not _is_katakana(n.surface)]
        self.foreign = [n for n in surnames if _is_katakana(n.surface)]
        self.given = [n for n in given if not _is_katakana(n.surface)]
        self.places = places

    @staticmethod
    def _split(names: list[Name], split: str) -> list[Name]:
        return sorted((n for n in names if is_heldout(n.reading) == (split == "heldout")), key=lambda n: n.cost)

    @staticmethod
    def _pick(rng: random.Random, names: list[Name]) -> Name:
        return rng.choice(names[:COMMON_TOP] if rng.random() < 0.5 else names)

    def surname(self, rng: random.Random) -> Name:
        return self._pick(rng, self.foreign if rng.random() < self.FOREIGN_RATE else self.surnames)

    def full_name(self, rng: random.Random) -> Name:
        if rng.random() < self.FOREIGN_RATE:
            return self._pick(rng, self.foreign)
        s, g = self._pick(rng, self.surnames), self._pick(rng, self.given)
        return Name(s.surface + g.surface, s.reading + g.reading, s.cost + g.cost)

    def place(self, rng: random.Random) -> Name:
        return self._pick(rng, self.places)
