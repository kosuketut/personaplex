"""datagen.assemble の間合いの決め方（data_spec §3）の不変条件。"""
import random

import numpy as np

from japersonaplex.datagen.assemble import bsi, plan_timeline


def make(turns, durs, words=None):
    d = {"id": "d", "scenario": "service",
         "turns": [{"idx": i, "speaker": s, "type": k, "text": x} for i, (s, k, x) in enumerate(turns)]}
    return d, {f"d/{i:02d}": dur for i, dur in enumerate(durs)}, words or {}


def channel_overlaps(placed):
    for ch in ("A", "B"):
        spans = sorted(((p["start"], p["cut"] or p["end"]) for p in placed if p["turn"]["speaker"] == ch))
        for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
            if b0 < a1:
                return True
    return False


def test_same_channel_never_overlaps():
    turns = [("A", "utterance", "x"), ("B", "utterance", "y"), ("A", "utterance", "z"), ("B", "utterance", "w")] * 5
    d, utts, words = make(turns, [2.0, 1.5, 3.0, 0.8] * 5)
    for seed in range(50):
        placed = plan_timeline(d, utts, words, random.Random(seed))
        assert not channel_overlaps(placed)
        assert [p["turn"]["idx"] for p in placed] == list(range(len(turns)))


def test_backchannel_is_overlaid_and_speaker_continues():
    turns = [("B", "utterance", "えっと、来週の土曜日なんですけど、"), ("A", "backchannel", "はい。"),
             ("B", "utterance", "二名で予約できますか。"), ("A", "utterance", "はい、承ります。")]
    d, utts, words = make(turns, [2.5, 0.3, 1.8, 1.5])
    for seed in range(30):
        placed = plan_timeline(d, utts, words, random.Random(seed))
        bc = [p for p in placed if p["backchannel"]]
        assert len(bc) == 1 and bc[0]["turn"]["idx"] == 1
        first, cont = placed[0], placed[2]
        # 相槌は前半の終わり付近（-100〜+300 ms）、続きは同じ話し手が短い間で話す
        assert -0.1 - 1e-9 <= bc[0]["start"] - first["end"] <= 0.3 + 1e-9
        assert 0.2 - 1e-9 <= cont["start"] - first["end"] <= 0.7 + 1e-9
        assert not channel_overlaps(placed)


def test_long_backchannel_label_is_treated_as_a_turn():
    turns = [("A", "utterance", "ご予約ですね。"), ("B", "backchannel", "あ、学習記録って、学校の成績表とかでいいんですかね。"),
             ("A", "utterance", "はい。")]
    d, utts, words = make(turns, [1.5, 3.0, 0.5])
    placed = plan_timeline(d, utts, words, random.Random(0))
    assert not any(p["backchannel"] for p in placed)


def test_interruption_cuts_agent_after_user_starts():
    turns = [("A", "utterance", "営業時間は七時から二十二時まで営業して"), ("B", "interruption", "あ、じゃあ"),
             ("A", "utterance", "はい、どうぞ。")]
    ws = [{"word": f"w{i}", "start": 0.4 * i, "end": 0.4 * i + 0.3, "punct": False} for i in range(10)]
    d, utts, _ = make(turns, [4.0, 1.5, 1.0])
    for seed in range(30):
        placed = plan_timeline(d, utts, {"d/00": ws}, random.Random(seed))
        agent, user = placed[0], placed[1]
        assert agent["cut"] is not None
        lag = agent["cut"] - user["start"]
        assert 0.3 - 1e-9 <= lag <= 0.8 + 1e-9 or agent["cut"] == agent["end"]
        assert 0.5 * 4.0 - 1e-9 <= user["start"] - agent["start"] <= 0.9 * 4.0 + 1e-9
        assert not channel_overlaps(placed)


def test_bsi_distribution_matches_spec():
    rng = random.Random(0)
    agent_to_user = np.array([bsi(rng, "service", "A", "B", 0.3) for _ in range(20000)])
    user_to_agent = np.array([bsi(rng, "service", "B", "A", 0.12) for _ in range(20000)])
    assert abs(np.mean(agent_to_user < 0) - 0.35) < 0.02
    assert abs(np.mean(user_to_agent < 0) - 0.10) < 0.01
    # agent は user の最後の 1 語（ここでは 0.12 秒）より深く重ならない
    assert user_to_agent.min() >= -0.12 - 1e-9
    gaps = user_to_agent[user_to_agent > 0]
    assert 0.2 < np.median(gaps) < 0.3 and gaps.max() <= 1.2


def test_compress_pauses_shortens_long_inner_silence_and_shifts_words():
    from japersonaplex.datagen.turntaking import MAX_PAUSE, PAUSE_RANGE, compress_pauses, silent_runs
    sr = 24000
    tone = lambda s: (0.3 * np.sin(np.arange(int(s * sr)) * 0.05)).astype(np.float32)  # noqa: E731
    wav = np.concatenate([tone(1.0), np.zeros(int(1.5 * sr), np.float32), tone(1.0)])
    words = [{"word": "a", "start": 0.1, "end": 0.9}, {"word": "b", "start": 2.6, "end": 3.4}]
    out, ws, removed = compress_pauses(wav, words, random.Random(0), sr)
    inner = [b - a for a, b in silent_runs(out, sr) if 0.05 < a and b < len(out) / sr - 0.05]
    assert len(inner) == 1 and PAUSE_RANGE[0] - 0.02 <= inner[0] <= PAUSE_RANGE[1] + 0.02
    assert abs(len(out) / sr - (len(wav) / sr - removed)) < 1e-3
    assert ws[0]["start"] == 0.1 and abs(ws[1]["start"] - (2.6 - removed)) < 1e-3
    # 短い無音は詰めない
    short = np.concatenate([tone(1.0), np.zeros(int((MAX_PAUSE - 0.1) * sr), np.float32), tone(1.0)])
    assert compress_pauses(short, words, random.Random(0), sr)[2] == 0.0


def test_backchannel_slots_at_commas_and_pauses_but_not_turn_end():
    from japersonaplex.datagen.turntaking import backchannel_slots
    ws = [{"word": "えっと、", "start": 0.0, "end": 0.4}, {"word": "来週", "start": 0.5, "end": 0.9},
          {"word": "の", "start": 0.9, "end": 1.0}, {"word": "土曜日", "start": 1.3, "end": 1.8},
          {"word": "です。", "start": 1.8, "end": 2.2}]
    assert backchannel_slots(ws, 10.0, 10.0 + 2.2 - 0.5) == [10.4, 11.0]
