"""ターンテイキングの細部（data_spec §3.3、§3.4）: 発話内の長い無音を詰める、聞き手の相槌を差し込む。

M2 の 110 対話（1.85 時間）の統計（stats.py）で、1 分あたりの Pause が 14.6 秒（日本語の実データは 3.5 秒）、
相槌が 0.5 回（仕様は 6〜10 回）、重なりが 0.4 秒（実データは 8.1 秒）と、仕様から大きく外れていたため。
"""
from __future__ import annotations

import random

import numpy as np

SR = 24000
# §3.4: agent の文の間は 0.15〜0.6 秒。これより長い発話内の無音は、この範囲の長さに詰める
MAX_PAUSE = 0.6
PAUSE_RANGE = (0.15, 0.6)
# §3.3: 相槌の回数（話し手の発話時間 1 分あたり）と語彙。キーは (場面, 相槌を打つ側)
BACKCHANNEL_RATE = {("service", "B"): 7.0, ("service", "A"): 6.0, ("qa", "B"): 7.0, ("qa", "A"): 4.0,
                    ("casual", "A"): 10.0, ("casual", "B"): 10.0}
BACKCHANNEL_WORDS = {
    "polite": ["はい", "ええ", "はいはい", "なるほど", "さようでございますか"],
    "casual": ["うん", "うんうん", "へえ", "そうなんだ", "なるほど", "ああ", "ふーん", "ほんとに"],
}
MIN_TURN_FOR_BC = 3.0  # この長さ以上の発話にだけ相槌を打つ
MIN_BC_SPACING = 1.5


def silent_runs(wav: np.ndarray, sr: int = SR, top_db: float = 40.0, frame: float = 0.01) -> list[tuple[float, float]]:
    """ピークから top_db 下を無音とみなした、無音区間 [(開始秒, 終了秒)]。"""
    n = int(frame * sr)
    if len(wav) < n:
        return []
    k = len(wav) // n
    rms = np.sqrt(np.mean(wav[: k * n].reshape(k, n) ** 2, axis=1) + 1e-12)
    quiet = 20 * np.log10(rms) < 20 * np.log10(rms.max()) - top_db
    runs, start = [], None
    for i, q in enumerate(quiet):
        if q and start is None:
            start = i
        elif not q and start is not None:
            runs.append((start * frame, i * frame))
            start = None
    if start is not None:
        runs.append((start * frame, k * frame))
    return runs


def compress_pauses(wav: np.ndarray, words: list[dict], rng: random.Random, sr: int = SR):
    """発話の内側の MAX_PAUSE を超える無音を PAUSE_RANGE の長さに詰める。単語の時刻もずらす。

    words は発話の先頭からの秒（align.py の出力。句読点を含む）。戻り値は (wav, words, 詰めた秒数)。
    無音の判定は音のエネルギーで行う（アライナの単語境界は無音の端を正確には与えないため）。
    """
    runs = [(a, b) for a, b in silent_runs(wav, sr) if b - a > MAX_PAUSE and a > 0.05 and b < len(wav) / sr - 0.05]
    if not runs:
        return wav, words, 0.0
    pieces, cuts, pos = [], [], 0
    for a, b in runs:
        keep = rng.uniform(*PAUSE_RANGE)
        cut_a = int((a + keep / 2) * sr)
        cut_b = int((b - keep / 2) * sr)
        pieces.append(wav[pos:cut_a])
        cuts.append((cut_a / sr, (cut_b - cut_a) / sr))
        pos = cut_b
    pieces.append(wav[pos:])

    def shift(t: float) -> float:
        removed = 0.0
        for at, dur in cuts:
            if t >= at + dur:
                removed += dur
            elif t > at:
                removed += t - at  # 削った区間の中の時刻は削った区間の始まりに寄せる
        return round(t - removed, 3)

    new_words = [dict(w, start=shift(w["start"]), end=shift(w["end"])) for w in words]
    return np.concatenate(pieces), new_words, float(sum(d for _, d in cuts))


def backchannel_slots(words: list[dict], start: float, end_limit: float) -> list[float]:
    """話し手の発話の中で相槌を置ける時刻（絶対秒）。句読点で終わる単語の終わりか、次の単語まで 150 ms 以上空く所。

    words は話し手の発話の単語（発話の先頭からの秒、句読点はくっつけたもの）。発話の最後の単語の後は除く
    （そこはターンの交替になるため）。
    """
    out = []
    for w, nxt in zip(words, words[1:]):
        t = start + w["end"]
        if t >= end_limit:
            break
        if w["word"][-1:] in "、。？！" or nxt["start"] - w["end"] >= 0.15:
            out.append(t)
    return out


def plan_backchannels(segments: list[dict], scenario: str, rng: random.Random) -> list[dict]:
    """segments: [{"speaker", "start", "end", "words", "slots"}]（相槌を打たれる側の発話、絶対秒）。
    戻り値: [{"speaker"（打つ側）, "start"}]。同じ聞き手の相槌同士は MIN_BC_SPACING 以上あける。
    聞き手がその時刻に自分で話していないかは呼び出し側で確かめる。"""
    out: list[dict] = []
    for seg in segments:
        dur = seg["end"] - seg["start"]
        if dur < MIN_TURN_FOR_BC or not seg["slots"]:
            continue
        listener = "B" if seg["speaker"] == "A" else "A"
        rate = BACKCHANNEL_RATE.get((scenario, listener), 6.0)
        n = np.random.default_rng(rng.randrange(2 ** 31)).poisson(rate * dur / 60)
        slots = sorted(rng.sample(seg["slots"], min(n, len(seg["slots"]))))
        last = -1e9
        for s in slots:
            t = s + rng.uniform(-0.1, 0.3)
            if t - last >= MIN_BC_SPACING:
                out.append({"speaker": listener, "start": t, "scenario": scenario})
                last = t
    return out
