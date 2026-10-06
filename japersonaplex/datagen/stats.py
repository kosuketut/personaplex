"""組み立てた対話のターンテイキングの統計（data_spec §3.2、§6.3）。単語時刻から求める。

    python -m japersonaplex.datagen.stats --out data/datagen/m2/run1

- 話者交替の間隔（BSI）: 話者ごとに、0.2 秒以上の無音で区切った発話区間（IPU、J-Moshi と同じ定義）を作り、
  話者が替わる境界の「次の開始 − 前の終了」。(a) 相槌を除いた交替だけ、(b) 相槌を含む全部、の 2 通り
- J-Moshi Table 3 の形式の 1 分あたりの合計秒数: IPU、Pause（同じ話者の IPU 間の無音）、Gap（交替の無音）、
  Overlap（両者が同時に話している時間）。日本語の実データは 59.7 / 3.5 / 4.0 / 8.1 秒。これだけは音声の
  エネルギーから求める（activity()）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

IPU_GAP = 0.2


def ipus(words: list[dict], speaker: str, with_bc: bool) -> list[tuple[float, float, bool]]:
    ws = sorted((w for w in words if w["speaker"] == speaker and (with_bc or not w.get("bc"))), key=lambda w: w["start"])
    out: list[list] = []
    for w in ws:
        if out and w["start"] - out[-1][1] < IPU_GAP and out[-1][2] == bool(w.get("bc")):
            out[-1][1] = max(out[-1][1], w["end"])
        else:
            out.append([w["start"], w["end"], bool(w.get("bc"))])
    return [tuple(x) for x in out]


def transitions(words: list[dict], with_bc: bool) -> list[float]:
    segs = sorted([(s, e, "A") for s, e, _ in ipus(words, "A", with_bc)] +
                  [(s, e, "B") for s, e, _ in ipus(words, "B", with_bc)])
    bsi, last = [], {}
    for s, e, spk in segs:
        other = "B" if spk == "A" else "A"
        prev_self, prev_other = last.get(spk), last.get(other)
        # 直前に話していたのが相手なら交替
        if prev_other is not None and (prev_self is None or prev_other[0] > prev_self[0]):
            bsi.append(s - prev_other[1])
        last[spk] = (s, e)
    return bsi


def activity(ch: np.ndarray, sr: int, frame: float = 0.01) -> np.ndarray:
    """音のエネルギーで話している区間を求める（10 ms ごとの真偽）。0.2 秒未満の無音は埋める（IPU の定義）。

    単語の時刻から作ると、CTC の単語の終わりが早めに出るせいで発話が細切れになり、Pause が多く出すぎた。
    閾値はチャネルのピークから 35 dB 下か −50 dBFS の高いほう（user 側の床ノイズ −65 dBFS 前後を拾わない）。
    """
    n = int(frame * sr)
    k = len(ch) // n
    db = 20 * np.log10(np.sqrt(np.mean(ch[: k * n].reshape(k, n) ** 2, axis=1)) + 1e-9)
    act = db > max(-50.0, db.max() - 35.0)
    gap = int(IPU_GAP / frame)
    idx = np.flatnonzero(act)
    for a, b in zip(idx, idx[1:]):
        if 1 < b - a <= gap:
            act[a:b] = True
    return act


def per_minute_audio(path: Path) -> dict:
    """J-Moshi Table 3 の形式（1 分あたりの IPU・Pause・Gap・Overlap の合計秒数）を、ステレオ wav から求める。"""
    import soundfile as sf
    wav, sr = sf.read(path, dtype="float32")
    a, b = activity(wav[:, 0], sr), activity(wav[:, 1], sr)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    either, both = a | b, a & b
    pause = gap = 0
    last_spk, i = None, 0
    while i < n:
        if either[i]:
            last_spk = "A" if (a[i] and not b[i]) else ("B" if (b[i] and not a[i]) else last_spk)
            i += 1
            continue
        j = i
        while j < n and not either[j]:
            j += 1
        if last_spk is not None and j < n:
            nxt = "A" if (a[j] and not b[j]) else ("B" if (b[j] and not a[j]) else last_spk)
            if nxt == last_spk:
                pause += j - i
            else:
                gap += j - i
        i = j
    m = n * 0.01 / 60
    return {"ipu_s": float(either.sum() * 0.01 / m), "pause_s": pause * 0.01 / m, "gap_s": gap * 0.01 / m,
            "overlap_s": float(both.sum() * 0.01 / m)}


def per_minute(words: list[dict], duration: float) -> dict:
    t = np.arange(0, duration, 0.01)
    act = {}
    for spk in ("A", "B"):
        a = np.zeros(len(t), dtype=bool)
        for s, e, _ in ipus(words, spk, True):
            a[int(s / 0.01):int(e / 0.01)] = True
        act[spk] = a
    both, either = act["A"] & act["B"], act["A"] | act["B"]
    # 無音区間を、前後が同じ話者なら Pause、違えば Gap に分ける
    pause = gap = 0.0
    segs = sorted([(s, e, k) for k in ("A", "B") for s, e, _ in ipus(words, k, True)])
    for (s0, e0, k0), (s1, e1, k1) in zip(segs, segs[1:]):
        silent = s1 - max(e0, max((e for s, e, k in segs if s <= s0), default=e0))
        if silent > 0:
            if k0 == k1:
                pause += silent
            else:
                gap += silent
    m = duration / 60
    return {"ipu_s": float(either.sum() * 0.01 / m), "pause_s": pause / m, "gap_s": gap / m,
            "overlap_s": float(both.sum() * 0.01 / m)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    rows = [json.loads(line) for line in open(out / "train.jsonl")]
    a, b, pm, bc_rate = [], [], [], []
    for r in rows:
        words = json.loads((out / r["words"]).read_text())
        a += transitions(words, with_bc=False)
        b += transitions(words, with_bc=True)
        pm.append(per_minute_audio(out / r["audio"]))
        bc_rate.append(sum(e["type"] == "backchannel" for e in r["events"]) / (r["duration"] / 60))
    a, b = np.array(a), np.array(b)

    def summ(x):
        return {"n": len(x), "overlap_frac": round(float(np.mean(x < 0)), 3),
                "median_ms": round(float(np.median(x)) * 1000), "abs_lt_500ms": round(float(np.mean(np.abs(x) < 0.5)), 3),
                "p5_p95_ms": [round(float(np.percentile(x, q)) * 1000) for q in (5, 95)]}
    res = {"dialogues": len(rows), "hours": round(sum(r["duration"] for r in rows) / 3600, 3),
           "bsi_turns_only": summ(a), "bsi_with_backchannels": summ(b),
           "per_minute": {k: round(float(np.mean([p[k] for p in pm])), 2) for k in pm[0]},
           "backchannels_per_min": round(float(np.mean(bc_rate)), 2),
           "spec": {"overlap_frac_turns_only": "0.20-0.30", "overlap_frac_with_bc": "0.30-0.40",
                    "jmoshi_real_per_minute": {"ipu_s": 59.7, "pause_s": 3.5, "gap_s": 4.0, "overlap_s": 8.1}}}
    (out / "stats.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
