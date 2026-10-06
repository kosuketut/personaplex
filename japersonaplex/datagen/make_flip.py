"""組み立て済みの対話から flip 評価のセット（eval_flip.py の flip.jsonl 形式）を作る。

pilot の flip（data/pilot/flip.jsonl）は pyopenjtalk の単一話者の声で作ってあり、pilot40 で学習したモデルには
未知の声になる。こちらは評価専用の名前・業種・声の対話（H-name、H-domain、H-voice）の実音声から作る。

1 件の作り方:
- 対話の単語時刻から、user の発話のあとに agent が数字の入った事実で答えている組を探す（最初の 3 回の質問まで）
- user の音声は、その質問の終わりまで（右チャンネル）に TAIL 秒の無音を足したもの。モデルは自分で挨拶してから答える
- プロンプト A は対話のプロンプトそのもの、B は答えに出た事実の数字を 1 か所だけ変えたもの。期待値は「数字＋単位」

    python -m japersonaplex.datagen.make_flip --out data/datagen/pilot40/flip40 \
        --set h_name=data/datagen/pilot40/h_name --set h_domain=data/datagen/pilot40/h_domain \
        --set h_voice=data/datagen/pilot40/h_voice --set valid=data/datagen/pilot40/main:valid.jsonl --n 24

--direct（v2）: 上の作り方では、元の台本の質問が事実を直接聞いていないことが多く（「明日って営業してますかね」）、
妥当な答えでも不正解になった（v1 の step 800 で 156 回答中 51 が「触れていない」）。--direct では、数字の入った事実を
1 つ選び、「{項目}を教えてもらえますか。」のような直接の質問を対話の user の声で TTS し、元の user の音声の
最初の発話の直前（agent の挨拶のあいだ）につなぐ。TTS を使うので data/envs/tts で GPU を使って動かす。

--from DIR --before-user N: --direct で作ったセット DIR の質問の音声をそのまま使い、元の user の N 番目の発話の直前に
つなぎ直す（それまでの元の user の発話は残す。TTS は使わない）。学習台本では数字の事実を最初の user の発話の直後に
答える組が少ないので、質問の位置が外れの原因かを見る。--set は DIR を作ったときと同じものを渡す。

--from DIR --questions FILE --form NAME: DIR と同じ対話・プロンプト・期待値で、質問の文だけを FILE（flip_questions.py が
台本 LLM で作る。項目名を言わない言い換え paraphrase と、事情を添えて聞く situational）の NAME 列に替えて TTS し直す。
flip40 v2 の質問は項目名で直接聞く形だけで、v3 の台本の形に近いため（docs/ja-pilot40-train.md）。
"""
import argparse
import json
import random
import re
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from ..scoring import mentions, normalize_numbers

# 質問のあとの無音。v1 と flip40 v2 の最初の版は 9 秒で、v1 の step 800 の外れの 35% が答えの途中で切れていたので 15 秒にした
TAIL_S = 15.0
MAX_QUESTIONS = 3
NUMBER = re.compile(r"\d+")
DIRECT_TEMPLATES = ["{key}を教えてもらえますか。", "すみません、{key}を知りたいんですけど。", "あの、{key}って、どうなってますか。",
                    "えっと、{key}について聞きたいんですけど。"]


def utterances(words: list[dict]) -> list[dict]:
    """相槌を除いた単語を、話者が替わるところで発話にまとめる。"""
    out = []
    for w in words:
        if w["bc"]:
            continue
        if out and out[-1]["speaker"] == w["speaker"]:
            out[-1]["text"] += w["word"]
            out[-1]["end"] = w["end"]
        else:
            out.append({"speaker": w["speaker"], "text": w["word"], "start": w["start"], "end": w["end"]})
    return out


def perturb(n: int, unit: str, others: set[int], rng: random.Random) -> int | None:
    """事実の数字 n を、同じ単位でありそうな別の値にする。"""
    if unit == "時":
        cands = [m for m in range(max(1, n - 3), min(23, n + 3) + 1) if m != n]
    elif unit == "円":
        step = 10 if n < 1000 else 100
        cands = [round(n * f / step) * step for f in (0.5, 0.6, 0.7, 0.8, 1.2, 1.3, 1.4, 1.5)]
    else:
        d = max(2, n // 3)
        cands = [m for m in range(max(1, n - d), n + d + 1) if m != n]
    cands = [m for m in cands if m > 0 and m != n and m not in others]
    return rng.choice(cands) if cands else None


def flip_fact(item: dict, key: str, value: str, rng: random.Random) -> dict | None:
    """事実 key の値の数字を 1 か所だけ変えたプロンプト B と期待値。変えられなければ None。"""
    if "万" in value:  # 「6万5千円」は数の正規化が扱えない
        return None
    others = {int(x) for x in NUMBER.findall(normalize_numbers(item["text_prompt"]))}
    for m in NUMBER.finditer(value):
        unit = value[m.end(): m.end() + 1]
        if not unit or unit.isdigit():
            continue
        new = perturb(int(m.group()), unit, others, rng)
        if new is None:
            continue
        value_b = value[: m.start()] + str(new) + value[m.end():]
        prompt_b = item["text_prompt"].replace(f"{key}は{value}", f"{key}は{value_b}", 1)
        if prompt_b != item["text_prompt"]:
            return {"kind": key, "prompts": [item["text_prompt"], prompt_b], "expect": [f"{m.group()}{unit}", f"{new}{unit}"]}
    return None


def make_direct_case(item: dict, root: Path, rng: random.Random) -> dict | None:
    """数字の入った事実を 1 つ選び、それを直接聞く質問の文と、元の user の最初の発話の開始時刻を返す。"""
    if item["kind"] != "service" or not item.get("facts"):
        return None
    first_user = next((u for u in utterances(json.loads((root / item["words"]).read_text())) if u["speaker"] == "B"), None)
    if first_user is None:
        return None
    facts = list(item["facts"].items())
    rng.shuffle(facts)
    for key, value in facts:
        case = flip_fact(item, key, value, rng)
        if case:
            question = rng.choice(DIRECT_TEMPLATES).format(key=key)
            return {"question": question, "question_start": first_user["start"], **case}
    return None


def make_case(item: dict, root: Path, rng: random.Random) -> dict | None:
    if item["kind"] != "service" or not item.get("facts"):
        return None
    utts = utterances(json.loads((root / item["words"]).read_text()))
    questions = 0
    for i, u in enumerate(utts[:-1]):
        if u["speaker"] != "B":
            continue
        questions += 1
        if questions > MAX_QUESTIONS:
            return None
        answer = utts[i + 1]
        if answer["speaker"] != "A":
            continue
        for key, value in item["facts"].items():
            if "万" in value:  # 「6万5千円」は数の正規化が扱えない
                continue
            for m in NUMBER.finditer(value):
                unit = value[m.end(): m.end() + 1]
                if not unit or unit.isdigit():
                    continue
                expect_a = f"{m.group()}{unit}"
                if not mentions(expect_a, answer["text"]):
                    continue
                others = {int(x) for x in NUMBER.findall(normalize_numbers(item["text_prompt"]))}
                new = perturb(int(m.group()), unit, others, rng)
                if new is None:
                    continue
                value_b = value[: m.start()] + str(new) + value[m.end():]
                prompt_b = item["text_prompt"].replace(f"{key}は{value}", f"{key}は{value_b}", 1)
                if prompt_b == item["text_prompt"]:
                    continue
                return {"question": u["text"], "answer": answer["text"], "question_end": u["end"],
                        "kind": key, "prompts": [item["text_prompt"], prompt_b], "expect": [expect_a, f"{new}{unit}"]}
    return None


def synth_questions(cases: list, voices_manifest: str, seed: int) -> list[tuple[np.ndarray, int]]:
    """直接の質問を対話の user の声で TTS する。0.6〜6 秒に入らないものは seed を変えて 2 回まで作り直す。"""
    from .tts import Synth, load_voices, max_tokens
    from .units import tts_text, utterance_units

    train, heldout = load_voices(voices_manifest)
    synth = Synth({v["id"]: v for v in train + heldout})
    jobs = []
    for _, _, _, item, case in cases:
        units = utterance_units(case["question"])
        jobs.append({"voice": item["user_voice"], "text": tts_text(units), "max_new_tokens": max_tokens(units)})
    out = synth.run(jobs, batch=64, seed=seed)
    for attempt in (1, 2):
        bad = [i for i, (w, sr) in enumerate(out) if not 0.6 <= len(w) / sr <= 6.0]
        if not bad:
            break
        print(f"retry {len(bad)} questions (attempt {attempt})", flush=True)
        for i, r in zip(bad, synth.run([jobs[i] for i in bad], batch=64, seed=seed + 1000 * attempt)):
            out[i] = r
    return out


def check_questions(out: Path) -> None:
    """--direct で作った質問の音声を whisper（CPU）で書き起こし、質問の文との一致度を flip.jsonl に書き足す。
    faster-whisper が要るので data/envs/align で動かす。"""
    from difflib import SequenceMatcher

    from faster_whisper import WhisperModel

    def norm(s: str) -> str:
        return re.sub(r"[\s、。？！?!,.]", "", normalize_numbers(s))

    model = WhisperModel("medium", device="cpu", compute_type="int8")
    rows = [json.loads(line) for line in open(out / "flip.jsonl")]
    for r in rows:
        wav, sr = sf.read(out / r["user_audio"], dtype="float32")
        end = len(wav) - int(r.get("tail_s", TAIL_S) * sr)
        seg = resample_poly(wav[end - int(r["question_s"] * sr): end], 16000, sr).astype(np.float32)
        text = "".join(s.text for s in model.transcribe(seg, language="ja", beam_size=5)[0])
        r["question_asr"], r["question_asr_ratio"] = text, round(SequenceMatcher(None, norm(r["question"]), norm(text)).ratio(), 3)
    with open(out / "flip.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    low = [r for r in rows if r["question_asr_ratio"] < 0.7]
    print(f"question ASR: {len(rows) - len(low)}/{len(rows)} with ratio >= 0.7", flush=True)
    for r in low:
        print(f"  {r['id']} {r['question_asr_ratio']} {r['question']} -> {r['question_asr']}", flush=True)


def splice_question(user: np.ndarray, start: int, q: np.ndarray, q_sr: int, sr: int) -> tuple[np.ndarray, float]:
    """TTS した質問を user の音声の start（サンプル）につなぎ、TAIL_S 秒の無音を足す。質問の長さ（秒）も返す。"""
    if q_sr != sr:
        q = resample_poly(q, sr, q_sr).astype(np.float32)
    # 元の user の最初の発話と同じ大きさにする（その区間がほぼ無音なら TTS のまま）
    ref_rms = float(np.sqrt(np.mean(user[start: start + len(q)] ** 2)))
    if ref_rms > 1e-3:
        q = q * (ref_rms / max(float(np.sqrt(np.mean(q ** 2))), 1e-6))
    return np.concatenate([user[:start], q, np.zeros(int(TAIL_S * sr), dtype=np.float32)]), round(len(q) / sr, 2)


def resynth_questions(src: Path, out: Path, sets: dict[str, tuple[Path, list[dict]]], questions: str, form: str,
                      voices: str, seed: int) -> None:
    """src（--direct のセット）と同じ対話・プロンプト・期待値で、質問の文だけを questions の form 列に替えて TTS し直す。"""
    texts = {r["id"]: r[form] for r in map(json.loads, open(questions)) if r.get(form)}
    rows, cases = [], []
    for r in map(json.loads, open(src / "flip.jsonl")):
        if r["id"] not in texts:
            continue
        root, items = sets[r["set"]]
        item = next(it for it in items if it["id"] == r["source"])
        cases.append((r["id"], r["set"], root, item, {"question": texts[r["id"]]}))
        rows.append(r)
    audio = synth_questions(cases, voices, seed)
    with open(out / "flip.jsonl", "w") as f:
        for r, (_, _, root, item, case), (q, q_sr) in zip(rows, cases, audio):
            pcm, sr = sf.read(root / item["audio"], dtype="float32")
            first_user = next(u for u in utterances(json.loads((root / item["words"]).read_text())) if u["speaker"] == "B")
            user, question_s = splice_question(pcm[:, 1].copy(), int(first_user["start"] * sr), q, q_sr, sr)
            sf.write(out / r["user_audio"], user, sr, subtype="PCM_16")
            shutil.copy(src / r["voice_prompt"], out / r["voice_prompt"])
            keep = {k: v for k, v in r.items() if not k.startswith("question")}
            f.write(json.dumps({**keep, "question": case["question"], "question_form": form, "question_s": question_s,
                                "tail_s": TAIL_S}, ensure_ascii=False) + "\n")
    print(f"{len(rows)} questions re-synthesized ({form})", flush=True)


def move_questions(src: Path, out: Path, sets: dict[str, tuple[Path, list[dict]]], before_user: int) -> None:
    """src（--direct のセット）の質問の音声を、元の user の before_user 番目の発話の直前につなぎ直す。"""
    rows = []
    for r in map(json.loads, open(src / "flip.jsonl")):
        root, items = sets[r["set"]]
        item = next(it for it in items if it["id"] == r["source"])
        users = [u for u in utterances(json.loads((root / item["words"]).read_text())) if u["speaker"] == "B"]
        if len(users) < before_user:
            continue
        old, sr = sf.read(src / r["user_audio"], dtype="float32")
        q = old[int(users[0]["start"] * sr): len(old) - int(r["tail_s"] * sr)]  # --direct は最初の発話の開始につないだ
        pcm, pcm_sr = sf.read(root / item["audio"], dtype="float32")
        assert pcm_sr == sr
        user = np.concatenate([pcm[: int(users[before_user - 1]["start"] * sr), 1], q, np.zeros(int(TAIL_S * sr), np.float32)])
        sf.write(out / r["user_audio"], user, sr, subtype="PCM_16")
        shutil.copy(src / r["voice_prompt"], out / r["voice_prompt"])
        rows.append({**r, "tail_s": TAIL_S, "before_user": before_user, "question_s": round(len(q) / sr, 2)})
    with open(out / "flip.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} cases moved before user utterance {before_user}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="NAME=DIR[:MANIFEST]",
                    help="組み立て済みの対話のディレクトリ。マニフェストの既定は train.jsonl")
    ap.add_argument("--check", action="store_true", help="--direct で作った質問の音声を whisper で確かめるだけ")
    ap.add_argument("--n", type=int, default=24, help="セットごとの最大件数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--direct", action="store_true", help="直接の質問を TTS して使う（v2）")
    ap.add_argument("--voices", default="data/voicebank/refs_v0/manifest.jsonl", help="--direct の user の声")
    ap.add_argument("--from", dest="src", help="--direct で作ったセット。質問の音声をつなぎ直す（--before-user）")
    ap.add_argument("--before-user", type=int, default=2, help="--from: 元の user の何番目の発話の直前につなぐか")
    ap.add_argument("--questions", help="--from: 質問の文の jsonl（flip_questions.py）。文を替えて TTS し直す")
    ap.add_argument("--form", help="--questions の列（paraphrase、situational）")
    args = ap.parse_args()

    out = Path(args.out)
    if args.check:
        check_questions(out)
        return
    for sub in ("user", "voices"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    sets = {}
    for spec in args.set:
        name, rest = spec.split("=", 1)
        directory, _, manifest = rest.partition(":")
        sets[name] = (Path(directory), [json.loads(line) for line in open(Path(directory) / (manifest or "train.jsonl"))])
    if args.src and args.questions:
        resynth_questions(Path(args.src), out, sets, args.questions, args.form, args.voices, args.seed)
        return
    if args.src:
        move_questions(Path(args.src), out, sets, args.before_user)
        return
    cases = []  # (id, セット名, root, 対話, case)
    for name, (root, items) in sets.items():
        items = list(items)
        rng = random.Random(f"{args.seed}:{name}")
        rng.shuffle(items)
        made = 0
        for item in items:
            if made >= args.n:
                break
            case = (make_direct_case if args.direct else make_case)(item, root, rng)
            if case is None:
                continue
            cases.append((f"{name}_{made:03d}", name, root, item, case))
            made += 1
        print(f"{name}: {made} cases from {len(items)} dialogues", flush=True)
    questions = synth_questions(cases, args.voices, args.seed) if args.direct else None

    rows = []
    for k, (cid, name, root, item, case) in enumerate(cases):
        pcm, sr = sf.read(root / item["audio"], dtype="float32")
        user = pcm[:, 1].copy()  # 左が agent、右が user（prepare.py と同じ）
        tail = np.zeros(int(TAIL_S * sr), dtype=np.float32)
        if args.direct:
            user, case["question_s"] = splice_question(user, int(case.pop("question_start") * sr), *questions[k], sr)
        else:
            end = int(case.pop("question_end") * sr) + int(0.2 * sr)
            user = np.concatenate([user[:end], tail])
        sf.write(out / "user" / f"{cid}.wav", user, sr, subtype="PCM_16")
        voice = Path(item["voice_prompt"])
        shutil.copy(root / voice, out / "voices" / f"{cid}_{voice.name}")
        rows.append({"id": cid, "set": name, "source": item["id"], "user_audio": f"user/{cid}.wav",
                     "voice_prompt": f"voices/{cid}_{voice.name}", "tail_s": TAIL_S, **case})
    with open(out / "flip.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
