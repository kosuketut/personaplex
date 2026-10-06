"""発話ごとの音声と単語時刻を、ターンテイキングの間合いをつけて 1 本の対話（ステレオ wav）に組み立てる。

リポジトリの .venv で動く（numpy、soundfile、pyloudnorm、sentencepiece）。

    python -m japersonaplex.datagen.assemble --out data/datagen/m1 \
        --tokenizer checkpoints/llm-jp-moshi-v1-pp16/tokenizer_spm_32k_3.model

入力は <out>/dialogues.jsonl、tts/utts.jsonl、tts/voices.jsonl、align/words.jsonl。出力:
    audio/<id>.wav    ステレオ 24 kHz（左 = agent、右 = user）
    words/<id>.json   [{"speaker", "word", "start", "end", "bc"}]。句読点は前の単語にくっつけてある
    train.jsonl       prepare.py 用のマニフェスト（data_spec §7.3 の拡張形式。prepare.py は余分なキーを無視する）
    qc.json           品質検査の集計。redo.txt は作り直す発話のキー（tts.py --redo に渡す）

間合いは data_spec §3 に従う（M1 で省いたもの: 相槌の事前合成、発話内のポーズの挿入、声の加工、残響）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pyloudnorm
import soundfile as sf

from .turntaking import backchannel_slots, compress_pauses, plan_backchannels, silent_runs
from .units import merge_punct

SR = 24000
FRAME_RATE = 12.5
MAX_DIALOGUE_S = 140.0  # data_spec §1.6
AGENT_LUFS = -24.0
# 一次の品質検査の閾値（data_spec §6.3 と bakeoff の観察から）
MAX_UNEXPLAINED_S = 0.3
# user は data_spec §6.3 の 0.15 から緩めた。user の音声は学習の正解ではなく、whisper が口語を書き換える
# （「〜ってどれくらいなんですかね」->「〜はどれくらいですか」）ことによる誤検出が user 側に多かったため
MAX_CER = {"A": 0.10, "B": 0.30}
MORA_PER_S = (3.0, 9.0)
BACKCHANNEL_S = (0.15, 0.8)
BACKCHANNEL_MAX_CHARS = 12

# data_spec §3.2: (重なりの確率, 重なり幅の中央値, 範囲), (ギャップの中央値, 95 パーセンタイル, 範囲)。秒
TIMING = {
    ("B", "A"): (0.10, 0.15, (0.08, 0.40), 0.25, 0.70, (0.08, 1.2)),   # user -> agent
    ("A", "B"): (0.35, 0.35, (0.08, 1.00), 0.35, 1.50, (0.08, 2.5)),   # agent -> user
    "casual": (0.35, 0.40, (0.08, 1.20), 0.25, 1.00, (0.08, 2.5)),
}


def lognormal(rng: random.Random, median: float, p95: float | None, lo: float, hi: float) -> float:
    sigma = np.log(p95 / median) / 1.645 if p95 else 0.5
    return float(np.clip(median * np.exp(rng.gauss(0.0, sigma)), lo, hi))


def bsi(rng: random.Random, scenario: str, prev: str, nxt: str, last_word_s: float) -> float:
    """話者交替の間隔（次の開始 − 前の終了）。負なら重なり。"""
    p_ov, ov_med, ov_rng, gap_med, gap_p95, gap_rng = TIMING["casual" if scenario == "casual" else (prev, nxt)]
    if rng.random() < p_ov:
        ov = lognormal(rng, ov_med, None, *ov_rng)
        if (prev, nxt) == ("B", "A") and scenario != "casual":
            ov = min(ov, last_word_s)  # agent は user の最後の 1 語の上でだけ話し始める
        return -ov
    return lognormal(rng, gap_med, gap_p95, *gap_rng)


def stable_seed(*parts) -> int:
    return int(hashlib.sha1("/".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def latest(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if path.exists():
        for line in open(path):
            r = json.loads(line)
            if r["key"] not in rows or r["attempt"] >= rows[r["key"]]["attempt"]:
                rows[r["key"]] = r
    return rows


def utterance_failures(utt: dict, al: dict | None, backchannel: bool) -> list[str]:
    if al is None or al["attempt"] != utt["attempt"]:
        return ["not_aligned"]
    qc, fails = al["qc"], []
    if utt["dur"] < 0.1:
        fails.append("empty")
    if qc["unexplained_speech_s"] > MAX_UNEXPLAINED_S:
        fails.append("unexplained_speech")
    if "cer" in qc and qc["cer"] > MAX_CER[utt["speaker"]]:
        fails.append("cer")
    if any(not e["ok"] for e in qc.get("entities", [])):
        fails.append("entity_reading")  # data_spec §6.3: 名前を含む発話は読みを照合する
    if utt["mora"] >= 6 and not MORA_PER_S[0] <= qc["mora_per_s"] <= MORA_PER_S[1]:
        fails.append("speech_rate")
    if backchannel and utt["dur"] < BACKCHANNEL_S[0]:
        fails.append("backchannel_length")  # 長すぎる相槌は普通のターンとして置くので落とさない（plan_timeline）
    return fails


def loudness_gain(wav: np.ndarray, target: float, meter: pyloudnorm.Meter) -> float:
    """target LUFS にする倍率。0.4 秒未満（pyloudnorm のブロック長）は RMS で代用する。"""
    if len(wav) >= int(0.4 * SR):
        lufs = meter.integrated_loudness(wav)
    else:
        lufs = 20 * np.log10(np.sqrt(np.mean(wav ** 2)) + 1e-9) - 0.691
    if not np.isfinite(lufs):
        return 1.0
    return float(10 ** ((target - lufs) / 20))


def speech_end(words: list[dict], dur: float) -> float:
    """発話の中で最後の単語が終わる時刻（発話の先頭から）。wav の末尾の無音を除いた、聞こえる終わり。"""
    ends = [w["end"] for w in words if not w.get("punct")]
    return min(dur, max(ends)) if ends else dur


def plan_timeline(d: dict, durs: dict[str, float], words: dict[str, list], rng: random.Random) -> list[dict]:
    """各ターンの開始時刻を決める。戻り値は [{turn, start, end, speech_end, cut, backchannel}]（秒、絶対時刻）。

    話者交替の間隔は、前の話者の最後の単語の終わりから測る（wav の末尾の無音からではない）。cut は割り込みで止める時刻。
    """
    turns = d["turns"]
    key = lambda t: f"{d['id']}/{t['idx']:02d}"  # noqa: E731

    def place(t, start, bc=False):
        k = key(t)
        return {"turn": t, "start": start, "end": start + durs[k], "speech_end": start + speech_end(words.get(k, []), durs[k]),
                "cut": None, "backchannel": bc}

    placed: list[dict] = []
    channel_end = {"A": 0.0, "B": 0.0}
    prev_main = None  # 直前の（相槌でない）ターンの placed 要素
    i = 0
    while i < len(turns):
        t = turns[i]
        nxt = turns[i + 1] if i + 1 < len(turns) else None
        is_bc = (t["type"] == "backchannel" and len(t["text"]) <= BACKCHANNEL_MAX_CHARS and prev_main is not None
                 and prev_main["turn"]["speaker"] != t["speaker"] and durs[key(t)] <= BACKCHANNEL_S[1])
        if is_bc:
            # 話し手の発話の切れ目（前半の終わり）の −100〜+300 ms に置く。話し手は止まらない
            p = place(t, max(prev_main["speech_end"] + rng.uniform(-0.1, 0.3), channel_end[t["speaker"]] + 0.05), True)
            placed.append(p)
            channel_end[t["speaker"]] = p["end"]
            if nxt is not None and nxt["speaker"] == prev_main["turn"]["speaker"]:
                # 相槌の後は同じ話し手が続ける。間は発話内のポーズ程度
                start2 = max(prev_main["speech_end"] + rng.uniform(0.2, 0.7), channel_end[nxt["speaker"]] + 0.05)
                p = place(nxt, start2)
                placed.append(p)
                channel_end[nxt["speaker"]] = p["end"]
                prev_main = p
                i += 2
                continue
            i += 1
            continue
        if prev_main is None:
            start = rng.uniform(0.3, 1.2)
        elif t["type"] == "interruption" and prev_main["turn"]["speaker"] == "A" and t["speaker"] == "B":
            # 割り込み: agent ターンの 50〜90% の位置にある単語の境界で user が話し始め、agent は 0.3〜0.8 秒後に止まる
            ws = [w for w in words.get(key(prev_main["turn"]), []) if not w["punct"]]
            pdur = prev_main["speech_end"] - prev_main["start"]
            cands = [w["start"] for w in ws if 0.5 * pdur <= w["start"] <= 0.9 * pdur]
            at = rng.choice(cands) if cands else 0.7 * pdur
            start = prev_main["start"] + at
            prev_main["cut"] = min(prev_main["end"], start + rng.uniform(0.3, 0.8))
            prev_main["speech_end"] = min(prev_main["speech_end"], prev_main["cut"])
            channel_end["A"] = prev_main["cut"]
        else:
            last = [w for w in words.get(key(prev_main["turn"]), []) if not w["punct"]]
            last_word_s = (last[-1]["end"] - last[-1]["start"]) if last else 0.2
            start = prev_main["speech_end"] + bsi(rng, d["scenario"], prev_main["turn"]["speaker"], t["speaker"],
                                                  last_word_s)
            start = max(start, prev_main["start"] + 0.3)
        start = max(start, channel_end[t["speaker"]] + 0.05)  # 同じチャネルの発話は重ねない
        p = place(t, start)
        placed.append(p)
        channel_end[t["speaker"]] = p["end"]
        prev_main = p
        i += 1
    return placed


def _busy(intervals: list[tuple[float, float]], a: float, b: float, margin: float = 0.2) -> bool:
    return any(a < e + margin and s - margin < b for s, e in intervals)


def assemble_one(d: dict, utts: dict, aligned: dict, voices: list[dict], bcs: list[dict], out: Path,
                 rng: random.Random, compress: bool = True, add_backchannels: bool = True):
    key = lambda t: f"{d['id']}/{t['idx']:02d}"  # noqa: E731
    wavs, words_rel, durs, removed = {}, {}, {}, 0.0
    for t in d["turns"]:
        k = key(t)
        wav, sr = sf.read(out / utts[k]["wav"], dtype="float32")
        assert sr == SR, f"{utts[k]['wav']}: {sr} Hz"
        ws = aligned[k]["words"]
        if compress:
            wav, ws, cut = compress_pauses(wav, ws, rng)  # §3.4: 発話内の長すぎる無音を詰める
            removed += cut
        wavs[k], words_rel[k], durs[k] = wav, ws, len(wav) / SR
    placed = plan_timeline(d, durs, words_rel, rng)
    # 長さの上限: はみ出すターン以降を落とす
    placed = [p for p in placed if (p["cut"] or p["end"]) <= MAX_DIALOGUE_S]
    total = max((p["cut"] or p["end"]) for p in placed) + rng.uniform(0.5, 1.5)
    stereo = np.zeros((int(total * SR) + 1, 2), dtype=np.float32)
    meter = pyloudnorm.Meter(SR)
    user_gain_db = rng.uniform(-10.0, 6.0) if rng.random() < 0.5 else 0.0
    base_level = {"A": AGENT_LUFS + rng.gauss(0.0, 1.5), "B": AGENT_LUFS + user_gain_db}
    words_out, events = [], []
    busy = {"A": [], "B": []}
    for p in placed:
        t = p["turn"]
        wav = wavs[key(t)]
        ch = 0 if t["speaker"] == "A" else 1
        level = base_level[t["speaker"]] + (rng.gauss(0.0, 1.0) if ch == 0 else 0.0)
        if p["backchannel"]:
            level -= rng.gauss(6.0, 3.0)  # 相槌はターンより小さく
        wav = wav * loudness_gain(wav, level, meter)
        if p["cut"] is not None:
            n = max(0, int((p["cut"] - p["start"]) * SR))
            fade = min(n, int(0.03 * SR))
            wav = wav[:n].copy()
            if fade:
                wav[n - fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
            events.append({"type": "interrupt", "speaker": "B", "agent_cut": round(p["cut"], 3)})
        a = int(p["start"] * SR)
        stereo[a:a + len(wav), ch] += wav[: len(stereo) - a]
        busy[t["speaker"]].append((p["start"], p["cut"] or p["end"]))
        if p["backchannel"]:
            events.append({"type": "backchannel", "speaker": t["speaker"], "start": round(p["start"], 3),
                           "end": round(p["end"], 3), "source": "script"})
        for w in merge_punct(words_rel[key(t)]):
            s = p["start"] + w["start"]
            if p["cut"] is not None and s >= p["cut"]:
                continue  # 止めた後に始まる単語はテキストからも消す（data_spec §3.5）
            words_out.append({"speaker": t["speaker"], "word": w["word"], "start": round(s, 3),
                              "end": round(min(p["start"] + w["end"], p["cut"] or 1e9), 3),
                              "bc": p["backchannel"], **({"spoken": w["spoken"]} if "spoken" in w else {})})
    # §3.3: 聞き手の相槌を、話し手の 3 秒以上の発話の切れ目に差し込む（事前に作った相槌の音声を使い回す）
    if add_backchannels and bcs:
        segments = []
        for p in placed:
            if p["backchannel"]:
                continue
            end = p["cut"] or p["speech_end"]
            ws = merge_punct(words_rel[key(p["turn"])])
            segments.append({"speaker": p["turn"]["speaker"], "start": p["start"], "end": end,
                             "slots": backchannel_slots(ws, p["start"], end - 0.5)})
        for b in plan_backchannels(segments, d["scenario"], rng):
            spk = b["speaker"]
            voice = d["agent_voice"] if spk == "A" else d["user_voice"]
            clips = [c for c in bcs if c["voice"] == voice and c["role"] == spk and c["style"] == d["voice_style"]]
            if not clips:
                continue
            c = rng.choice(clips)
            s0, s1 = b["start"], b["start"] + c["dur"]
            if s1 >= total - 0.1 or _busy(busy[spk], s0, s1):
                continue  # 聞き手が自分で話している間には打たない
            wav, _ = sf.read(out / c["path"], dtype="float32")
            wav = wav * loudness_gain(wav, base_level[spk] - rng.gauss(6.0, 3.0), meter)
            a = int(s0 * SR)
            stereo[a:a + len(wav), 0 if spk == "A" else 1] += wav[: len(stereo) - a]
            busy[spk].append((s0, s1))
            events.append({"type": "backchannel", "speaker": spk, "start": round(s0, 3), "end": round(s1, 3),
                           "source": "inserted", "word": c["word"]})
            # agent の相槌は実際に話しているのでテキストストリームに入れる（§3.3、§5.2）。開始は前の無音を除いた位置
            runs = silent_runs(wav)
            lead = runs[0][1] if runs and runs[0][0] == 0.0 else 0.0
            words_out.append({"speaker": spk, "word": c["word"], "start": round(s0 + lead, 3), "end": round(s1, 3),
                              "bc": True})
    # user チャネルの床ノイズ（推論時はマイク入力）。-65 ± 5 dBFS の白色雑音
    noise_db = rng.uniform(-70.0, -60.0)
    stereo[:, 1] += np.random.default_rng(rng.randrange(2 ** 31)).standard_normal(len(stereo)).astype(np.float32) \
        * 10 ** (noise_db / 20)
    peak = np.abs(stereo).max(axis=0)
    stereo /= np.maximum(peak / 10 ** (-1 / 20), 1.0)  # ピークを -1 dBFS に制限
    sf.write(out / "audio" / f"{d['id']}.wav", stereo, SR, subtype="PCM_16")
    words_out.sort(key=lambda w: (w["start"], w["speaker"]))
    (out / "words" / f"{d['id']}.json").write_text(json.dumps(words_out, ensure_ascii=False))
    vp = [v for v in voices if v["voice"] == d["agent_voice"] and v["style"] == d["voice_style"]]
    return {
        "id": d["id"], "split": "train", "kind": d["scenario"], "audio": f"audio/{d['id']}.wav",
        "words": f"words/{d['id']}.json", "text_prompt": d["text_prompt"],
        "voice_prompt": rng.choice(vp)["path"] if vp else None, "agent_voice": d["agent_voice"],
        "user_voice": d["user_voice"], "facts": d.get("facts"),
        "entities": [{"surface": s, "reading": r, "in_prompt": s in d["text_prompt"]} for s, r in d["entities"].items()],
        "events": events, "duration": round(total, 2), "n_turns_used": len(placed), "n_turns": len(d["turns"]),
        "pause_removed_s": round(removed, 2), "gen": d.get("gen"),
    }


def token_overflow(words: list[dict], tokenizer) -> dict:
    """data_spec §5.3: 単語ごとに「トークン数 ≤ 単語のフレーム数 + 2」か、前の単語とぶつかって後ろへずれた量。"""
    from ..sequence import encode_word
    over, lag_gt3, cursor, n, unk = 0, 0, 0, 0, []
    for w in sorted((w for w in words if w["speaker"] == "A"), key=lambda w: w["start"]):
        if tokenizer.unk_id() in tokenizer.encode(w["word"]):
            unk.append(w["word"])  # 語彙に無い文字は encode_word が捨てるので、テキストから抜ける
        ids = encode_word(tokenizer, w["word"])
        if not ids:
            continue
        n += 1
        frames = max(1, int(round((w["end"] - w["start"]) * FRAME_RATE)))
        over += len(ids) > frames + 2
        start = int(w["start"] * FRAME_RATE)
        lag_gt3 += max(cursor, start) - start > 3
        cursor = max(cursor, start) + len(ids)
    return {"n_words": n, "tokens_gt_frames_plus2": over, "lag_gt_3_frames": lag_gt3,
            "lag_gt_3_frac": round(lag_gt3 / max(n, 1), 4), "unk_words": unk}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", help="指定するとテキストトークンのあふれ（§5.3）も集計する")
    ap.add_argument("--keep-failed", action="store_true", help="品質検査に落ちた発話を含む対話もマニフェストに入れる")
    ap.add_argument("--no-compress", action="store_true", help="発話内の長い無音を詰めない（比較用）")
    ap.add_argument("--no-backchannels", action="store_true", help="聞き手の相槌を差し込まない（比較用）")
    ap.add_argument("--max-attempts", type=int, default=3,
                    help="1 発話を合成する回数の上限（data_spec §7.2 の k）。超えても落ちる発話を含む対話は捨てる")
    args = ap.parse_args()
    out = Path(args.out)
    for sub in ("audio", "words"):
        (out / sub).mkdir(exist_ok=True)
    dialogues = [json.loads(line) for line in open(out / "dialogues.jsonl")]
    utts = latest(out / "tts" / "utts.jsonl")
    aligned = latest(out / "align" / "words.jsonl")
    voices = [json.loads(line) for line in open(out / "tts" / "voices.jsonl")]
    bc_path = out / "tts" / "backchannels.jsonl"
    bcs = [json.loads(line) for line in open(bc_path)] if bc_path.exists() else []
    tokenizer = None
    if args.tokenizer:
        import sentencepiece
        tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)

    redo, manifest, report = [], [], []
    for d in dialogues:
        fails = {}
        for t in d["turns"]:
            k = f"{d['id']}/{t['idx']:02d}"
            is_bc = t["type"] == "backchannel" and len(t["text"]) <= BACKCHANNEL_MAX_CHARS  # plan_timeline と同じ判定
            f = utterance_failures(utts[k], aligned.get(k), is_bc)
            if f:
                fails[k] = f
        if any(f == ["not_aligned"] for f in fails.values()):
            report.append({"id": d["id"], "status": "not_aligned"})
            continue
        retry = [k for k in fails if utts[k]["attempt"] + 1 < args.max_attempts]
        redo += retry
        if fails and not args.keep_failed:
            report.append({"id": d["id"], "status": "failed_qc" if retry else "dropped", "fails": fails})
            continue
        if not any(v["voice"] == d["agent_voice"] and v["style"] == d["voice_style"] for v in voices):
            report.append({"id": d["id"], "status": "no_voice_prompt"})  # tts.py --fill-voices で作れる
            continue
        row = assemble_one(d, utts, aligned, voices, bcs, out, random.Random(stable_seed("assemble", d["id"])),
                           compress=not args.no_compress, add_backchannels=not args.no_backchannels)
        row["qc"] = {"failed_utterances": fails,
                     "max_unexplained_s": max(aligned[f"{d['id']}/{t['idx']:02d}"]["qc"]["unexplained_speech_s"]
                                              for t in d["turns"]),
                     "cer": {s: [aligned[f"{d['id']}/{t['idx']:02d}"]["qc"].get("cer") for t in d["turns"]
                                 if t["speaker"] == s] for s in ("A", "B")}}
        if tokenizer is not None:
            row["qc"]["tokens"] = token_overflow(json.loads((out / row["words"]).read_text()), tokenizer)
            if row["qc"]["tokens"]["lag_gt_3_frac"] > 0.02:  # data_spec §5.3
                report.append({"id": d["id"], "status": "token_overflow", "tokens": row["qc"]["tokens"]})
                continue
        manifest.append(row)
        report.append({"id": d["id"], "status": "ok", "duration": row["duration"], "fails": fails})
    with open(out / "train.jsonl", "w") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "redo.txt").write_text("".join(k + "\n" for k in sorted(set(redo))))
    summary = {"dialogues": len(dialogues), "assembled": len(manifest), "redo_utterances": len(set(redo)),
               "dropped": sum(r["status"] == "dropped" for r in report),
               "no_voice_prompt": sum(r["status"] == "no_voice_prompt" for r in report),
               "hours": round(sum(r["duration"] for r in manifest) / 3600, 3)}
    (out / "qc.json").write_text(json.dumps({"summary": summary, "dialogues": report}, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
