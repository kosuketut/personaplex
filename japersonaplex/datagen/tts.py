"""台本（LLM の出力 jsonl）の各発話と声プロンプトを Qwen3-TTS で合成する。

環境は data/envs/tts（torch 2.8.0+cu128、qwen_tts、pyopenjtalk）。リポジトリの .venv では動かない。

    PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 data/envs/tts/bin/python -m japersonaplex.datagen.tts \
        --scripts scripts.jsonl --voices data/voicebank/refs_v0/manifest.jsonl --out data/datagen/m1

出力（--out 以下）:
    dialogues.jsonl   対話ごとの情報（台本、声の割り当て、固有名詞と読み）
    tts/utts.jsonl    発話ごとの情報。作り直したら同じ key の行を追記する（後の段は attempt が最大の行を使う）
    tts/utt/<対話>/<番号>_<attempt>.wav
    tts/voices.jsonl、tts/voices/<声>_<k>.wav   声プロンプト（agent と同じ声で、対話に出てこない中立的な文）

固有名詞は TTS に読みのかなで渡す（units.tts_text）。読みは台本の "readings"（scripts_llm.py が IPAdic から引いたもの）。
--redo にキーの一覧（1 行 1 キー）を渡すと、その発話だけ seed を変えて作り直す。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pyopenjtalk
import soundfile as sf
import torch

from .units import kata2hira, mora_count, tts_text, utterance_units

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
SPEAKER = {"agent": "A", "user": "B"}
HELDOUT_VOICE_FRACTION = 0.1  # data_spec §6.1 H-voice
# 声プロンプト用の中立的な文（data_spec §4.4: 話題を持たない、役割に合う文体 70%）
# 長さの違う文を混ぜてある（話速の遅い声・速い声でも 3.5〜5.5 秒に入る文があるように）
VOICE_PROMPT_TEXTS = {
    "polite": ["お電話ありがとうございます。ご用件をお伺いいたします。", "本日はご来店いただき、誠にありがとうございます。",
               "少々お待ちください。ただいま確認いたします。", "ご不明な点がございましたら、お気軽にお尋ねください。",
               "かしこまりました。確認いたしますね。", "お待たせいたしました。ご案内いたします。",
               "いつもご利用いただき、誠にありがとうございます。担当の者がご用件をお伺いいたします。",
               "恐れ入りますが、もう一度お名前とご用件をお伺いしてもよろしいでしょうか。"],
    "casual": ["うん、ちょっと聞いてほしいことがあるんだけど、いいかな。", "そうそう、それでね、この前の話の続きなんだけど。",
               "えっと、どこから話せばいいかな。とりあえず聞いてくれる？", "なるほどね。じゃあ、次はこっちの話をしようか。",
               "あ、それいいね。ちょっと考えてみる。", "うんうん、わかるよ。そういうことあるよね。",
               "えー、そうなんだ。全然知らなかったな。もうちょっと詳しく教えてくれる？",
               "そういえばさ、最近ちょっと気になってることがあって、聞いてもらってもいいかな。"],
}
STYLE = {"service": "polite", "qa": "polite", "casual": "casual"}


def stable_seed(*parts) -> int:
    return int(hashlib.sha1("/".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def default_reading(surface: str) -> str:
    return kata2hira("".join(f["read"] for f in pyopenjtalk.run_frontend(surface)))


def entities_of(script: dict) -> dict[str, str]:
    """台本の固有名詞（店名・担当者名・客の名前）と読み。scripts_llm.py の台本は辞書の読みを "readings" に持つ。
    持たない台本（bakeoff の 12 本）は pyopenjtalk の推定で代用する。"""
    if script.get("readings"):
        return dict(script["readings"])
    spec = script.get("spec", {})
    names = {spec[k] for k in ("store_name", "agent_name", "user_name") if spec.get(k)}
    return {n: default_reading(n) for n in names}


def load_voices(path: str) -> tuple[list[dict], list[dict]]:
    voices = [json.loads(line) for line in open(path) if line.strip()]
    for v in voices:
        v["id"] = Path(v["path"]).stem.split("_")[0]
        if not Path(v["path"]).exists():  # 別のマシンの絶対パスなら、manifest と同じディレクトリから読む
            v["path"] = str(Path(path).parent / Path(v["path"]).name)
    heldout = [v for v in voices if stable_seed("voice", v["speaker_id"]) % 1000 < HELDOUT_VOICE_FRACTION * 1000]
    train = [v for v in voices if v not in heldout]
    return train, heldout


def plan_dialogues(scripts: list[dict], voices: list[dict]) -> list[dict]:
    plans = []
    for s in scripts:
        rng = random.Random(stable_seed("plan", s["id"]))
        agent = rng.choice(voices)
        user = rng.choice([v for v in voices if v["speaker_id"] != agent["speaker_id"]])
        readings = entities_of(s)
        turns = []
        for i, t in enumerate(s["turns"]):
            kind = t["type"] if not (t["type"] == "interruption" and t["speaker"] != "user") else "utterance"
            units = utterance_units(t["text"], readings)
            turns.append({"idx": i, "speaker": SPEAKER[t["speaker"]], "type": kind, "text": t["text"], "units": units})
        plans.append({"id": s["id"], "scenario": s["scenario"], "text_prompt": s["system_prompt"], "facts": s.get("facts"),
                      "spec": s.get("spec"), "agent_voice": agent["id"], "user_voice": user["id"],
                      "voice_style": STYLE.get(s["scenario"], "polite"), "entities": readings, "turns": turns,
                      "gen": {"llm": s.get("model"), "tts": MODEL}})
    return plans


def max_tokens(units: list[dict]) -> int:
    """12 Hz の codec トークン数の上限。繰り返しで止まらない失敗の被害と、バッチの待ち時間を抑える。"""
    return int(12 * (mora_count(units) / 4 + 3))


def model_path() -> str:
    """TTS の重みの場所。キャッシュにあればそのローカルパスを使う（計算ノードから HF に出られない場合に備える。
    HF_HUB_OFFLINE=1 だと qwen_tts の読み込みが model_info を呼んで失敗した）。環境変数 TTS_MODEL で上書きできる。"""
    if os.environ.get("TTS_MODEL"):
        return os.environ["TTS_MODEL"]
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(MODEL, local_files_only=True)
    except Exception:  # noqa: BLE001  キャッシュに無ければ HF から取る
        return MODEL


class Synth:
    def __init__(self, voices: dict[str, dict], device: str = "cuda:0"):
        from qwen_tts import Qwen3TTSModel
        self.model = Qwen3TTSModel.from_pretrained(model_path(), device_map=device, dtype=torch.bfloat16,
                                                   attn_implementation="sdpa")
        self.voices = voices
        self.prompts: dict[str, object] = {}

    def prompt(self, voice_id: str):
        if voice_id not in self.prompts:
            v = self.voices[voice_id]
            self.prompts[voice_id] = self.model.create_voice_clone_prompt(
                ref_audio=v["path"], ref_text=v["text"], x_vector_only_mode=False)[0]
        return self.prompts[voice_id]

    def run(self, jobs: list[dict], batch: int, seed: int, token_budget: int = 6000) -> list[tuple[np.ndarray, int]]:
        """jobs: [{"voice", "text", "max_new_tokens"}]。長さの近いものをまとめてバッチにする。

        バッチの大きさは batch 以下で、かつ「件数 × 最長の max_new_tokens」が token_budget 以下。長い発話ばかりの
        バッチで codec の復号がメモリを使い切った（48 GB の GPU、64 件で OOM）ための制限。
        """
        order = sorted(range(len(jobs)), key=lambda i: jobs[i]["max_new_tokens"])
        groups, cur = [], []
        for i in order:
            if cur and (len(cur) >= batch or (len(cur) + 1) * jobs[i]["max_new_tokens"] > token_budget):
                groups.append(cur)
                cur = []
            cur.append(i)
        if cur:
            groups.append(cur)
        out: list = [None] * len(jobs)
        done = 0
        for g, idx in enumerate(groups):
            torch.manual_seed(seed + g)
            wavs, sr = self.model.generate_voice_clone(
                text=[jobs[i]["text"] for i in idx], language=["Japanese"] * len(idx),
                voice_clone_prompt=[self.prompt(jobs[i]["voice"]) for i in idx],
                max_new_tokens=max(jobs[i]["max_new_tokens"] for i in idx))
            for i, w in zip(idx, wavs):
                out[i] = (np.asarray(w, dtype=np.float32), sr)
            done += len(idx)
            print(f"tts {done}/{len(order)}", flush=True)
        return out


def make_voice_prompts(synth: "Synth", need: list[tuple[str, str]], have: list[dict], out: Path, per_voice: int,
                       batch: int, token_budget: int) -> list[dict]:
    """声プロンプト（data_spec §4.4）: agent の声と文体ごとに per_voice 本。3.5〜5.5 秒に入るまで文を替えて作り直す。"""
    rows = list(have)
    for attempt in range(len(VOICE_PROMPT_TEXTS["polite"])):
        jobs = []
        for voice, style in need:
            n = sum(1 for r in rows if r["voice"] == voice and r["style"] == style)
            texts = VOICE_PROMPT_TEXTS[style]
            for k in range(n, per_voice):
                units = utterance_units(texts[(k + attempt) % len(texts)])
                jobs.append({"voice": voice, "style": style, "k": k, "text": tts_text(units),
                             "max_new_tokens": max_tokens(units)})
        if not jobs:
            break
        for job, (wav, sr) in zip(jobs, synth.run(jobs, batch, stable_seed("vp", attempt), token_budget)):
            wav = trim(wav, sr)
            if 3.5 <= len(wav) / sr <= 5.5:
                path = out / "tts" / "voices" / f"{job['voice']}_{job['style']}_{job['k']}.wav"
                sf.write(path, wav, sr, subtype="PCM_16")
                rows.append({"voice": job["voice"], "style": job["style"], "path": str(path.relative_to(out)),
                             "text": job["text"], "dur": round(len(wav) / sr, 2)})
    missing = [n for n in need if not any(r["voice"] == n[0] and r["style"] == n[1] for r in rows)]
    if missing:
        print(f"WARNING: 声プロンプトが 3.5〜5.5 秒に入らなかった声: {missing}（この声の対話は組み立てで捨てる）", flush=True)
    return rows


# 相槌の語彙（data_spec §3.3）。キーは (役割, 場面の文体)。0.8 秒を超えやすい長い相槌は入れていない
BACKCHANNEL_VOCAB = {
    ("A", "polite"): ["はい", "ええ", "はいはい", "なるほど", "そうですか"],
    ("B", "polite"): ["はい", "ええ", "うん", "なるほど", "へえ"],
    ("A", "casual"): ["うん", "うんうん", "へえ", "そうなんだ", "なるほど", "ああ", "ふーん", "ほんとに"],
    ("B", "casual"): ["うん", "うんうん", "へえ", "そうなんだ", "なるほど", "ああ", "ふーん", "ほんとに"],
}


def make_backchannels(synth: "Synth", dialogues: list[dict], out: Path, variants: int, batch: int,
                      token_budget: int) -> list[dict]:
    """声ごとに相槌の音声を先に作っておく（§3.3: 短い発話は TTS が失敗しやすいので、長さで選別して使い回す）。"""
    need = sorted({(d["agent_voice"], "A", d["voice_style"]) for d in dialogues} |
                  {(d["user_voice"], "B", d["voice_style"]) for d in dialogues})
    (out / "tts" / "bc").mkdir(parents=True, exist_ok=True)
    jobs = []
    for voice, role, style in need:
        for word in BACKCHANNEL_VOCAB[(role, style)]:
            units = utterance_units(word + "。")
            for k in range(variants):
                jobs.append({"voice": voice, "role": role, "style": style, "word": word, "k": k,
                             "text": tts_text(units), "max_new_tokens": 36})
    rows = []
    for job, (wav, sr) in zip(jobs, synth.run(jobs, batch, stable_seed("bc"), token_budget)):
        wav = trim(wav, sr, pad=0.02)
        dur = len(wav) / sr
        if not 0.15 <= dur <= 0.8:
            continue
        path = out / "tts" / "bc" / f"{job['voice']}_{job['role']}_{job['style']}_{job['word']}_{job['k']}.wav"
        sf.write(path, wav, sr, subtype="PCM_16")
        rows.append({k: job[k] for k in ("voice", "role", "style", "word")} | {"path": str(path.relative_to(out)),
                                                                              "dur": round(dur, 3)})
    print(f"backchannels: {len(rows)}/{len(jobs)} clips within 0.15-0.8 s", flush=True)
    return rows


def trim(wav: np.ndarray, sr: int, top_db: float = 40.0, pad: float = 0.05) -> np.ndarray:
    """前後の無音を落とす（ピークから top_db 下を無音とみなす）。"""
    frame = int(0.01 * sr)
    if len(wav) < frame:
        return wav
    rms = np.sqrt(np.convolve(wav ** 2, np.ones(frame) / frame, mode="same") + 1e-12)
    loud = np.flatnonzero(20 * np.log10(rms) > 20 * np.log10(rms.max()) - top_db)
    if len(loud) == 0:
        return wav[:0]
    a, b = max(0, loud[0] - int(pad * sr)), min(len(wav), loud[-1] + int(pad * sr))
    return wav[a:b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scripts", required=True)
    ap.add_argument("--voices", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--token-budget", type=int, default=6000, help="1 バッチの「件数 × 最長の codec トークン数」の上限")
    ap.add_argument("--prompts-per-voice", type=int, default=2)
    ap.add_argument("--redo", help="作り直す発話のキーを 1 行 1 つ書いたファイル")
    ap.add_argument("--fill-voices", action="store_true", help="声プロンプトが足りない声の分だけ作って終わる")
    ap.add_argument("--backchannels", action="store_true", help="相槌の音声（声と役割ごと）を作って終わる")
    ap.add_argument("--voice-split", choices=["train", "heldout"], default="train",
                    help="heldout は評価専用の声（data_spec §6.1 の H-voice、声バンクの約 10%%）だけを使う")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "tts" / "utt").mkdir(parents=True, exist_ok=True)
    (out / "tts" / "voices").mkdir(parents=True, exist_ok=True)

    train_voices, heldout_voices = load_voices(args.voices)
    use_voices = heldout_voices if args.voice_split == "heldout" else train_voices
    all_voices = {v["id"]: v for v in use_voices}
    utts_path = out / "tts" / "utts.jsonl"

    if args.fill_voices:
        dialogues = [json.loads(line) for line in open(out / "dialogues.jsonl")]
        need = sorted({(d["agent_voice"], d["voice_style"]) for d in dialogues})
        vpath = out / "tts" / "voices.jsonl"
        have = [json.loads(line) for line in open(vpath)] if vpath.exists() else []
        missing = [n for n in need if not any(r["voice"] == n[0] and r["style"] == n[1] for r in have)]
        if not missing:
            return
        vrows = make_voice_prompts(Synth(all_voices), missing, have, out, args.prompts_per_voice, args.batch,
                                   args.token_budget)
        with open(vpath, "w") as f:
            for r in vrows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return
    synth = Synth(all_voices)
    if args.backchannels:
        dialogues = [json.loads(line) for line in open(out / "dialogues.jsonl")]
        rows = make_backchannels(synth, dialogues, out, 3, args.batch, args.token_budget)
        with open(out / "tts" / "backchannels.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return
    if args.redo:
        keys = {k.strip() for k in open(args.redo) if k.strip()}
        latest: dict[str, dict] = {}
        for line in open(utts_path):
            row = json.loads(line)
            latest[row["key"]] = row
        # 単位列は作り直すたびに台本と読みから作り直す（units.py の変更を古い行にも反映するため）
        readings = {d["id"]: d["entities"] for d in map(json.loads, open(out / "dialogues.jsonl"))}
        rows = [dict(latest[k], attempt=latest[k]["attempt"] + 1,
                     units=utterance_units(latest[k]["text"], readings[latest[k]["did"]])) for k in sorted(keys)]
    else:
        scripts = [json.loads(line) for line in open(args.scripts) if line.strip()]
        plans = plan_dialogues(scripts, use_voices)
        with open(out / "dialogues.jsonl", "w") as f:
            for p in plans:
                f.write(json.dumps({k: v for k, v in p.items() if k != "turns"} | {
                    "turns": [{k: v for k, v in t.items() if k != "units"} for t in p["turns"]]}, ensure_ascii=False) + "\n")
        rows = [{"key": f"{p['id']}/{t['idx']:02d}", "did": p["id"], **t, "voice": p["agent_voice" if t["speaker"] == "A"
                 else "user_voice"], "attempt": 0} for p in plans for t in p["turns"]]

        need = sorted({(p["agent_voice"], p["voice_style"]) for p in plans})
        vrows = make_voice_prompts(synth, need, [], out, args.prompts_per_voice, args.batch, args.token_budget)
        with open(out / "tts" / "voices.jsonl", "w") as f:
            for r in vrows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 作り直し（attempt >= 1）では漢字を読みのかなにして渡す（漢字の読み違いが同じ発話で繰り返されたため）
    jobs = [{"voice": r["voice"], "text": tts_text(r["units"], kana_for_kanji=r["attempt"] >= 1),
             "max_new_tokens": max_tokens(r["units"])} for r in rows]
    results = synth.run(jobs, args.batch, stable_seed("utt", len(rows), rows[0]["attempt"] if rows else 0),
                        args.token_budget)
    with open(utts_path, "a" if args.redo else "w") as f:
        for r, (wav, sr) in zip(rows, results):
            wav = trim(wav, sr)
            d = out / "tts" / "utt" / r["did"]
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"{r['idx']:02d}_{r['attempt']}.wav"
            sf.write(path, wav, sr, subtype="PCM_16")
            r = dict(r, tts_text=tts_text(r["units"], kana_for_kanji=r["attempt"] >= 1), wav=str(path.relative_to(out)), sr=sr,
                     dur=round(len(wav) / sr, 3), mora=mora_count(r["units"]))
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
